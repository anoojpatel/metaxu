from tape_vm import Opcode, Instruction

class VMToCCompiler:
    def __init__(self, instructions):
        self.instructions = instructions
        self.c_code = ""
        self.indent_level = 0
        self.label_counter = 0
        self.label_mapping = {}
        self.vars = set()  # Keep track of variable declarations
        self.stack = []

    def compile(self):
        self.c_code += "#include <stdio.h>\n"
        self.c_code += "#include <stdlib.h>\n"
        self.c_code += "#include <pthread.h>\n"  # For multithreading
        self.c_code += "\n"
        self.c_code += "typedef void* (*thread_func_t)(void*);\n"
        self.c_code += "\n"
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

        if opcode == Opcode.LOAD_CONST:
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
        elif opcode == Opcode.CALL_FUNC:
            func_name = instr.operands[0]
            arg_count = instr.operands[1] if len(instr.operands) > 1 else 0
            args = [self.stack_pop() for _ in range(arg_count)][::-1]
            temp_var = self.get_temp_var()
            self.c_code += f"{indent}int {temp_var} = {func_name}({', '.join(args)});\n"
            self.stack_push(temp_var)
        elif opcode == Opcode.RETURN:
            value = self.stack_pop()
            self.c_code += f"{indent}return {value};\n"
        elif opcode == Opcode.MOVE_VAR:
            # For C code, moving variables is not directly applicable
            var_name = instr.operands[0]
            value = self.stack_pop()
            self.c_code += f"{indent}// MOVE_VAR {var_name} (handled)\n"
            # Invalidate the original variable if necessary
        elif opcode == Opcode.VEC_ADD:
            # Implement vector addition using loops or intrinsic functions
            pass  # Placeholder
        elif opcode == Opcode.CREATE_VECTOR:
            # Implement vector creation
            pass  # Placeholder
        elif opcode == Opcode.CREATE_STRUCT:
            # Implement struct creation
            pass  # Placeholder
        elif opcode == Opcode.ACCESS_FIELD:
            field_name = instr.operands[0]
            struct_var = self.stack_pop()
            temp_var = self.get_temp_var()
            self.c_code += f"{indent}int {temp_var} = {struct_var}.{field_name};\n"
            self.stack_push(temp_var)
        elif opcode == Opcode.SPAWN_THREAD:
            func_name = self.stack_pop()
            temp_var = self.get_temp_var()
            self.c_code += f"{indent}pthread_t {temp_var};\n"
            self.c_code += f"{indent}pthread_create(&{temp_var}, NULL, (thread_func_t){func_name}, NULL);\n"
            self.stack_push(temp_var)
        elif opcode == Opcode.SEND_MESSAGE:
            # Implement message passing between threads
            pass  # Placeholder
        elif opcode == Opcode.RECEIVE_MESSAGE:
            # Implement message receiving
            pass  # Placeholder
        elif opcode == Opcode.PERFORM_EFFECT:
            # Implement effect handling
            pass  # Placeholder
        elif opcode == Opcode.SET_HANDLER:
            # Implement setting effect handler
            pass  # Placeholder
        elif opcode == Opcode.UNSET_HANDLER:
            # Implement unsetting effect handler
            pass  # Placeholder
        elif opcode == Opcode.RESUME:
            # Implement resuming from an effect
            pass  # Placeholder
        elif opcode == Opcode.TO_DEVICE:
            # Implement data transfer to device (GPU)
            pass  # Placeholder
        elif opcode == Opcode.FROM_DEVICE:
            # Implement data transfer from device (GPU)
            pass  # Placeholder
        elif opcode == Opcode.JUMP_IF_FALSE:
            label = instr.operands[0]
            condition = self.stack_pop()
            self.c_code += f"{indent}if (!{condition}) goto {label};\n"
        elif opcode == Opcode.JUMP:
            label = instr.operands[0]
            self.c_code += f"{indent}goto {label};\n"
        elif opcode == Opcode.LABEL:
            label = instr.operands[0]
            self.c_code += f"{label}:\n"
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

