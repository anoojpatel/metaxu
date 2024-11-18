from tape_vm import Opcode, Instruction

class VMToCCompiler:
    def __init__(self, instructions):
        self.instructions = instructions
        self.c_code = ""
        self.indent_level = 0
        self.vars = set()
        self.stack = []
        self.current_scope = None
        self.scope_counter = 0
        self.thread_local_vars = set()
        self.cuda_vars = set()
        self.message_queues = set()

    def compile(self):
        # Include necessary headers
        self.c_code += "#include <stdio.h>\n"
        self.c_code += "#include <stdlib.h>\n"
        self.c_code += "#include <pthread.h>\n"  # For multithreading
        self.c_code += "#include <cuda_runtime.h>\n"  # For CUDA operations
        self.c_code += "#include <setjmp.h>\n"  # For continuation handling
        self.c_code += "\n"
        
        # Simple continuation structure
        self.c_code += "typedef struct {\n"
        self.c_code += "    jmp_buf env;\n"
        self.c_code += "    void* result;\n"
        self.c_code += "} continuation_t;\n\n"

        # Add thread-local storage and message queue types
        self.c_code += "typedef struct {\n"
        self.c_code += "    pthread_mutex_t mutex;\n"
        self.c_code += "    pthread_cond_t cond;\n"
        self.c_code += "    void* data;\n"
        self.c_code += "    int size;\n"
        self.c_code += "    int capacity;\n"
        self.c_code += "} message_queue_t;\n\n"

        # Add thread context structure
        self.c_code += "typedef struct {\n"
        self.c_code += "    message_queue_t* queues;\n"
        self.c_code += "    void* locals;\n"
        self.c_code += "} thread_context_t;\n\n"

        # Add effect handler table
        self.c_code += "typedef struct {\n"
        self.c_code += "    int effect_type;\n"
        self.c_code += "    void (*handler)(continuation_state_t*);\n"
        self.c_code += "} effect_handler_t;\n\n"

        # Add effect handler registration
        self.c_code += "effect_handler_t effect_handlers[256];\n"
        self.c_code += "int num_effect_handlers = 0;\n\n"

        self.c_code += "void register_effect_handler(int effect_type, void (*handler)(continuation_state_t*)) {\n"
        self.c_code += "    effect_handlers[num_effect_handlers].effect_type = effect_type;\n"
        self.c_code += "    effect_handlers[num_effect_handlers].handler = handler;\n"
        self.c_code += "    num_effect_handlers++;\n"
        self.c_code += "}\n\n"

        # Generate message queue operations
        self._generate_message_queue_operations()
        
        self.c_code += "int main() {\n"
        self.indent_level += 1
        self.translate_instructions()
        self.indent_level -= 1
        self.c_code += "    return 0;\n"
        self.c_code += "}\n"
        return self.c_code

    def translate_instructions(self):
        idx = 0
        while idx < len(self.instructions):
            instr = self.instructions[idx]
            self.translate_instruction(instr)
            idx += 1

    def translate_instruction(self, instr):
        opcode = instr.opcode
        indent = '    ' * self.indent_level

        if opcode == Opcode.CREATE_FUNC:
            self.current_scope = instr.operands[0]
            self.c_code += f"{indent}{{\n"
            self.indent_level += 1
            # Initialize thread-local storage if needed
            if self.thread_local_vars:
                for var in self.thread_local_vars:
                    self.c_code += f"{indent}__thread int {var};\n"

        elif opcode == Opcode.SPAWN_THREAD:
            func_name = self.stack_pop()
            context = self.get_temp_var()
            thread = self.get_temp_var()
            
            # Create thread context
            self.c_code += f"{indent}thread_context_t* {context} = malloc(sizeof(thread_context_t));\n"
            self.c_code += f"{indent}{context}->queues = create_message_queue(10);\n"
            self.c_code += f"{indent}{context}->locals = NULL;\n"
            
            # Create and start thread
            self.c_code += f"{indent}pthread_t {thread};\n"
            self.c_code += f"{indent}pthread_create(&{thread}, NULL, (void* (*)(void*)){func_name}, {context});\n"
            self.stack_push(thread)

        elif opcode == Opcode.TO_DEVICE:
            var = self.stack_pop()
            size = instr.operands[0] if instr.operands else 1
            device_var = self.get_temp_var()
            
            self.c_code += f"{indent}int* {device_var}_d;\n"
            self.c_code += f"{indent}cudaMalloc((void**)&{device_var}_d, {size} * sizeof(int));\n"
            self.c_code += f"{indent}cudaMemcpy({device_var}_d, {var}, {size} * sizeof(int), cudaMemcpyHostToDevice);\n"
            self.cuda_vars.add(device_var)
            self.stack_push(f"{device_var}_d")

        elif opcode == Opcode.FROM_DEVICE:
            device_var = self.stack_pop()
            size = instr.operands[0] if instr.operands else 1
            host_var = self.get_temp_var()
            
            self.c_code += f"{indent}int* {host_var} = (int*)malloc({size} * sizeof(int));\n"
            self.c_code += f"{indent}cudaMemcpy({host_var}, {device_var}, {size} * sizeof(int), cudaMemcpyDeviceToHost);\n"
            self.c_code += f"{indent}cudaFree({device_var});\n"
            self.stack_push(host_var)

        elif opcode == Opcode.SEND_MESSAGE:
            queue = self.stack_pop()
            msg = self.stack_pop()
            self.c_code += f"{indent}send_message({queue}, (void*){msg});\n"

        elif opcode == Opcode.RECEIVE_MESSAGE:
            queue = self.stack_pop()
            result = self.get_temp_var()
            self.c_code += f"{indent}int {result} = (int)receive_message({queue});\n"
            self.stack_push(result)

        elif opcode == Opcode.CREATE_CONTINUATION:
            # Create a continuation point
            cont_var = self.get_temp_var()
            self.c_code += f"{indent}continuation_t* {cont_var} = malloc(sizeof(continuation_t));\n"
            self.c_code += f"{indent}if (setjmp({cont_var}->env) == 0) {{\n"
            self.indent_level += 1
            self.c_code += f"{indent}{cont_var}->result = NULL;\n"
            self.indent_level -= 1
            self.c_code += f"{indent}}}\n"
            self.stack_push(cont_var)

        elif opcode == Opcode.RESUME:
            # Resume a continuation with a result
            cont_var = self.stack_pop()
            result_var = self.stack_pop() if len(self.stack) > 0 else "NULL"
            self.c_code += f"{indent}{cont_var}->result = (void*){result_var};\n"
            self.c_code += f"{indent}longjmp({cont_var}->env, 1);\n"

        elif opcode == Opcode.RETURN:
            # End the current scope
            if self.current_scope:
                self.indent_level -= 1
                self.c_code += f"{indent}}}\n"
                self.current_scope = None
            value = self.stack_pop()
            self.c_code += f"{indent}return {value};\n"

        elif opcode == Opcode.CALL_FUNC:
            # Create a new scope for function call
            func_name = instr.operands[0]
            arg_count = instr.operands[1] if len(instr.operands) > 1 else 0
            args = [self.stack_pop() for _ in range(arg_count)][::-1]
            temp_var = self.get_temp_var()
            self.c_code += f"{indent}{{\n"
            self.indent_level += 1
            self.c_code += f"{indent}int {temp_var} = {func_name}({', '.join(args)});\n"
            self.indent_level -= 1
            self.c_code += f"{indent}}}\n"
            self.stack_push(temp_var)

        elif opcode == Opcode.LOAD_CONST:
            value = instr.operands[0]
            temp_var = self.get_temp_var()
            if isinstance(value, int):
                self.c_code += f"{indent}int {temp_var} = {value};\n"
            elif isinstance(value, float):
                self.c_code += f"{indent}float {temp_var} = {value};\n"
            elif isinstance(value, str):
                self.c_code += f'{indent}char* {temp_var} = "{value}";\n'
            else:
                raise Exception(f"Unsupported constant type: {type(value)}")
            self.stack_push(temp_var)

        elif opcode == Opcode.LOAD_VAR:
            var_name = instr.operands[0]
            self.stack_push(var_name)

        elif opcode == Opcode.STORE_VAR:
            var_name = instr.operands[0]
            value = self.stack_pop()
            if var_name not in self.vars:
                var_type = 'int'  # Assume int for simplicity; adjust as needed
                self.c_code += f"{indent}{var_type} {var_name} = {value};\n"
                self.vars.add(var_name)
            else:
                self.c_code += f"{indent}{var_name} = {value};\n"

        elif opcode == Opcode.ADD:
            right = self.stack_pop()
            left = self.stack_pop()
            temp_var = self.get_temp_var()
            self.c_code += f"{indent}int {temp_var} = {left} + {right};\n"
            self.stack_push(temp_var)

        elif opcode == Opcode.SUB:
            right = self.stack_pop()
            left = self.stack_pop()
            temp_var = self.get_temp_var()
            self.c_code += f"{indent}int {temp_var} = {left} - {right};\n"
            self.stack_push(temp_var)

        elif opcode == Opcode.MUL:
            right = self.stack_pop()
            left = self.stack_pop()
            temp_var = self.get_temp_var()
            self.c_code += f"{indent}int {temp_var} = {left} * {right};\n"
            self.stack_push(temp_var)

        elif opcode == Opcode.DIV:
            right = self.stack_pop()
            left = self.stack_pop()
            temp_var = self.get_temp_var()
            self.c_code += f"{indent}int {temp_var} = {left} / {right};\n"
            self.stack_push(temp_var)

        elif opcode == Opcode.JUMP:
            label = instr.operands[0]
            self.c_code += f"{indent}goto {label};\n"

        elif opcode == Opcode.LABEL:
            label = instr.operands[0]
            self.c_code += f"{label}:\n"

        elif opcode == Opcode.CREATE_STRUCT:
            # Get struct name and field names from operands
            struct_name = instr.operands[0]
            field_names = instr.operands[1]
            
            # Pop field values from stack in reverse order
            field_values = [self.stack_pop() for _ in range(len(field_names))][::-1]
            
            # Create struct instance
            temp_var = self.get_temp_var()
            self.c_code += f"{indent}struct {struct_name} {temp_var} = {{\n"
            for name, value in zip(field_names, field_values):
                self.c_code += f"{indent}    .{name} = {value},\n"
            self.c_code += f"{indent}}};\n"
            
            self.stack_push(temp_var)

        elif opcode == Opcode.ACCESS_FIELD:
            # Get field name from operands
            field_name = instr.operands[0]
            
            # Pop struct variable from stack
            struct_var = self.stack_pop()
            
            # Access field and store in temp var
            temp_var = self.get_temp_var()
            self.c_code += f"{indent}int {temp_var} = {struct_var}.{field_name};\n"
            
            self.stack_push(temp_var)

        elif opcode == Opcode.DUP:
            value = self.stack[-1]
            self.stack_push(value)
        elif opcode == Opcode.POP:
            self.stack_pop()
        elif opcode == Opcode.PRINT:
            value = self.stack_pop()
            self.c_code += f"{indent}printf(\"%d\\n\", {value});\n"
        else:
            raise Exception(f"Unsupported opcode: {opcode}")

    def get_temp_var(self):
        temp_var = f"t{len(self.vars)}"
        self.vars.add(temp_var)
        return temp_var

    def stack_push(self, value):
        self.stack.append(value)

    def stack_pop(self):
        if not self.stack:
            raise Exception("Stack underflow")
        return self.stack.pop()

    def translate_instructions(self):
        for instr in self.instructions:
            self.translate_instruction(instr)
