from metaxu.tape_vm import Opcode, Instruction
from metaxu.effect_map import EffectParser
from typing import List, Optional, Dict

class EffectMapper:
    def __init__(self):
        self.effect_map = {}
        self.c_effect_types = set()

    def register_effect_mapping(self, metaxu_effect: str, c_effect: str, return_type: str, arg_types: List[str]):
        """Register a mapping from Metaxu effect to C runtime effect"""
        key = (metaxu_effect, c_effect)
        self.effect_map[key] = {
            'return_type': return_type,
            'arg_types': arg_types
        }
        self.c_effect_types.add(c_effect)

    def get_effect_mapping(self, metaxu_effect: str, c_effect: str) -> Optional[Dict]:
        """Get the mapping info for a Metaxu-to-C effect pair"""
        return self.effect_map.get((metaxu_effect, c_effect))

class VMToCCompiler:
    def __init__(self, instructions, linker, options=None):
        self.instructions = instructions
        self.linker = linker
        self.options = options
        self.c_code = ""
        self.indent_level = 0
        self.vars = set()
        self.stack = []
        self.type_stack = []
        self.temp_var_count = 0
        self.label_count = 0
        self.current_scope = None
        self.scope_counter = 0
        self.thread_local_vars = set()
        self.cuda_enabled = False  # Default to no CUDA support
        self.message_queues = set()
        self.drop_tracker = DropTracker()
        self.effect_parser = EffectParser()
        self.effect_mapper = EffectMapper()

    def register_effect_mapping(self, metaxu_effect: str, c_effect: str, return_type: str, arg_types: List[str]):
        """Register a mapping from Metaxu effect to C runtime effect"""
        self.effect_mapper.register_effect_mapping(metaxu_effect, c_effect, return_type, arg_types)

    def compile_effect_operation(self, effect_op: str, args: List[str]) -> str:
        """Compile an effect operation with C runtime mapping"""
        mapping = self.effect_mapper.get_effect_mapping(effect_op, self.effect_parser.get_c_effect(effect_op))
        if not mapping:
            raise Exception(f"No mapping found for effect {effect_op}")
                
        # Generate C function call
        return f"runtime_{self.effect_parser.get_c_effect(effect_op).lower()}({', '.join(args)})"

    def compile(self):
        """Generate C code from VM instructions"""
        # Set CUDA support based on options
        if self.options and hasattr(self.options, 'cuda_enabled'):
            self.cuda_enabled = self.options.cuda_enabled

        self._generate_prelude()
        self._generate_runtime()
        
        # For libraries, generate exported symbols
        if self.options and self.options.link_mode == "library":
            self._generate_exports()
        
        # Generate main function for executables
        if not self.options or self.options.link_mode != "library":
            self.c_code += "int main(int argc, char *argv[]) {\n"
            self.indent_level += 1
            self._generate_init()
        
        # Generate instruction implementations
        for instr in self.instructions:
            self._compile_instruction(instr)
            
        # Close main for executables
        if not self.options or self.options.link_mode != "library":
            self.c_code += "    return 0;\n"
            self.c_code += "}\n"
            
        return self.c_code

    def _generate_prelude(self):
        """Generate includes and common declarations"""
        # Standard includes
        self.c_code += "#include <stdio.h>\n"
        self.c_code += "#include <stdlib.h>\n"
        self.c_code += "#include <string.h>\n"
        self.c_code += "#include <stdbool.h>\n"  # For bool type
        self.c_code += "#include <pthread.h>\n"  # For multithreading
        self.c_code += "#include <setjmp.h>\n"   # For continuation handling
        
        # Include effect system
        self.c_code += "#include \"effects.h\"\n"
        
        # Only include CUDA if explicitly enabled
        if self.cuda_enabled:
            self.c_code += "#include <cuda_runtime.h>\n"
        
        self.c_code += "\n"
        
        # Forward declarations
        self.c_code += "// Forward declarations\n"
        self.c_code += "struct continuation_state;\n"
        self.c_code += "struct thread_pool;\n"
        self.c_code += "struct future;\n"
        self.c_code += "struct queue;\n"
        self.c_code += "struct task;\n"
        self.c_code += "struct domain;\n\n"
        
        # Type aliases
        self.c_code += "// Type definitions\n"
        self.c_code += "typedef struct continuation_state continuation_state_t;\n"
        self.c_code += "typedef struct thread_pool thread_pool_t;\n"
        self.c_code += "typedef struct future future_t;\n"
        self.c_code += "typedef struct queue queue_t;\n"
        self.c_code += "typedef struct task task_t;\n"
        self.c_code += "typedef struct domain domain_t;\n\n"

    def _generate_exports(self):
        """Generate exported symbols for library mode"""
        if not self.options.library_name:
            raise ValueError("Library name must be specified for library builds")
            
        # Generate library initialization function
        self.c_code += f"EXPORT void {self.options.library_name}_init(void) {{\n"
        self._generate_init()
        self.c_code += "}\n\n"
        
        # Generate cleanup function
        self.c_code += f"EXPORT void {self.options.library_name}_cleanup(void) {{\n"
        self.c_code += "    // TODO: Add cleanup code\n"
        self.c_code += "}\n\n"

    def _generate_runtime(self):
        """Generate runtime support code"""
        self._generate_effect_table()
        self.c_code += "\n// Runtime initialization\n"
        self.c_code += "static mx_runtime_t* runtime = NULL;\n\n"
        self.c_code += "static void init_runtime() {\n"
        self.c_code += "    if (runtime) return;\n"
        self.c_code += "    runtime = malloc(sizeof(mx_runtime_t));\n"
        self.c_code += "    runtime->effect_system = mx_effect_system_create();\n"
        self.c_code += "    mx_init_static_effects(runtime->effect_system);\n"
        self.c_code += "}\n\n"

        # Basic structures
        self.c_code += "// Runtime structures\n"
        
        # Continuation structure
        self.c_code += "struct continuation_state {\n"
        self.c_code += "    jmp_buf env;\n"
        self.c_code += "    void* result;\n"
        self.c_code += "    void (*handler)(continuation_state_t*);\n"
        self.c_code += "};\n\n"
        
        # Domain structure for ownership tracking
        self.c_code += "typedef struct domain {\n"
        self.c_code += "    pthread_mutex_t mutex;\n"
        self.c_code += "    int domain_id;\n"
        self.c_code += "    void* data;\n"
        self.c_code += "    void (*drop_value)(void*);\n"
        self.c_code += "    bool value_needs_drop;\n"
        self.c_code += "} domain_t;\n\n"
        
        # Message queue structure
        self.c_code += "struct queue {\n"
        self.c_code += "    void** items;\n"
        self.c_code += "    size_t capacity;\n"
        self.c_code += "    size_t size;\n"
        self.c_code += "    size_t head;\n"
        self.c_code += "    size_t tail;\n"
        self.c_code += "    pthread_mutex_t mutex;\n"
        self.c_code += "    pthread_cond_t not_empty;\n"
        self.c_code += "    pthread_cond_t not_full;\n"
        self.c_code += "};\n\n"
        
        # Task structure
        self.c_code += "struct task {\n"
        self.c_code += "    void (*function)(void*);\n"
        self.c_code += "    void* arg;\n"
        self.c_code += "    bool detached;\n"
        self.c_code += "    future_t* future;\n"
        self.c_code += "};\n\n"
        
        # Thread pool structure
        self.c_code += "struct thread_pool {\n"
        self.c_code += "    pthread_t* threads;\n"
        self.c_code += "    size_t num_threads;\n"
        self.c_code += "    queue_t* tasks;\n"
        self.c_code += "    bool should_stop;\n"
        self.c_code += "    pthread_mutex_t mutex;\n"
        self.c_code += "    pthread_cond_t condition;\n"
        self.c_code += "};\n\n"
        
        # Future structure
        self.c_code += "struct future {\n"
        self.c_code += "    void* result;\n"
        self.c_code += "    bool is_ready;\n"
        self.c_code += "    pthread_mutex_t mutex;\n"
        self.c_code += "    pthread_cond_t condition;\n"
        self.c_code += "};\n\n"
        
        # Function prototypes
        self.c_code += "// Function prototypes\n"
        self.c_code += "bool queue_empty(queue_t* queue);\n"
        self.c_code += "void* queue_pop(queue_t* queue);\n"
        self.c_code += "void free_message_queue(queue_t* queue);\n"
        self.c_code += "void free_thread_pool(thread_pool_t* pool);\n"
        self.c_code += "void free_future(future_t* future);\n\n"
        
        # Implementation of utility functions
        self.c_code += "// Utility function implementations\n"
        self._generate_utility_functions()

    def _generate_utility_functions(self):
        """Generate implementations of utility functions"""
        # Queue operations
        self.c_code += "bool queue_empty(queue_t* queue) {\n"
        self.c_code += "    return queue->size == 0;\n"
        self.c_code += "}\n\n"
        
        self.c_code += "void* queue_pop(queue_t* queue) {\n"
        self.c_code += "    if (queue_empty(queue)) return NULL;\n"
        self.c_code += "    void* item = queue->items[queue->head];\n"
        self.c_code += "    queue->head = (queue->head + 1) % queue->capacity;\n"
        self.c_code += "    queue->size--;\n"
        self.c_code += "    return item;\n"
        self.c_code += "}\n\n"
        
        # Memory management functions
        self.c_code += "void free_message_queue(queue_t* queue) {\n"
        self.c_code += "    if (queue) {\n"
        self.c_code += "        if (queue->items) free(queue->items);\n"
        self.c_code += "        pthread_mutex_destroy(&queue->mutex);\n"
        self.c_code += "        pthread_cond_destroy(&queue->not_empty);\n"
        self.c_code += "        pthread_cond_destroy(&queue->not_full);\n"
        self.c_code += "        free(queue);\n"
        self.c_code += "    }\n"
        self.c_code += "}\n\n"
        
        self.c_code += "void free_thread_pool(thread_pool_t* pool) {\n"
        self.c_code += "    if (pool) {\n"
        self.c_code += "        if (pool->threads) free(pool->threads);\n"
        self.c_code += "        if (pool->tasks) {\n"
        self.c_code += "            while (!queue_empty(pool->tasks)) {\n"
        self.c_code += "                task_t* task = queue_pop(pool->tasks);\n"
        self.c_code += "                if (task) free(task);\n"
        self.c_code += "            }\n"
        self.c_code += "            free_message_queue(pool->tasks);\n"
        self.c_code += "        }\n"
        self.c_code += "        pthread_mutex_destroy(&pool->mutex);\n"
        self.c_code += "        pthread_cond_destroy(&pool->condition);\n"
        self.c_code += "        free(pool);\n"
        self.c_code += "    }\n"
        self.c_code += "}\n\n"
        
        self.c_code += "void free_future(future_t* future) {\n"
        self.c_code += "    if (future) {\n"
        self.c_code += "        pthread_mutex_destroy(&future->mutex);\n"
        self.c_code += "        pthread_cond_destroy(&future->condition);\n"
        self.c_code += "        free(future);\n"
        self.c_code += "    }\n"
        self.c_code += "}\n\n"

    def _generate_effect_table(self):
        """Generate static effect handler table"""
        self.c_code += "\n// Static effect handler table\n"
        self.c_code += "static const struct {\n"
        self.c_code += "    const char* name;\n"
        self.c_code += "    mx_effect_handler_t handler;\n"
        self.c_code += "} mx_static_effects[] = {\n"
        
        # Add all registered effects
        for metaxu_effect, c_effect in self.effect_mapper.effect_map.items():
            handler_name = f"handle_{c_effect[1].lower()}"
            self.c_code += f'    {{"{metaxu_effect[0]}::{metaxu_effect[1]}", {handler_name}}},\n'
        
        self.c_code += "    {NULL, NULL}\n"
        self.c_code += "};\n\n"
        
        # Generate static initialization function
        self.c_code += "static void mx_init_static_effects(mx_effect_system_t* system) {\n"
        self.c_code += "    for (int i = 0; mx_static_effects[i].name != NULL; i++) {\n"
        self.c_code += "        mx_register_effect(system, mx_static_effects[i].name, mx_static_effects[i].handler);\n"
        self.c_code += "    }\n"
        self.c_code += "}\n\n"

    def _generate_init(self):
        """Generate initialization code"""
        self.c_code += "    // Initialize effect system\n"
        self.c_code += "    init_effect_system();\n\n"
        
        # Initialize thread pool if needed
        if self._uses_threads():
            self.c_code += "    // Initialize thread pool\n"
            self.c_code += "    thread_pool_t* global_pool = create_thread_pool(4);\n"
        
        # Initialize global domain if needed
        if self._uses_domains():
            self.c_code += "    // Initialize global domain\n"
            self.c_code += "    domain_t* global_domain = create_domain();\n"

    def _compile_instruction(self, instr):
        """Compile a single instruction"""
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
            
            # Create spawn request
            self.c_code += f"{indent}spawn_request_t req = {{.function = {func_name}}};\n"
            self.c_code += f"{indent}thread_t* {thread} = invoke_effect(EFFECT_SPAWN, &req);\n"
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
            
            # Create send request
            self.c_code += f"{indent}message_t req = {{.queue = {queue}, .data = {msg}}};\n"
            self.c_code += f"{indent}invoke_effect(EFFECT_SEND, &req);\n"

        elif opcode == Opcode.RECEIVE_MESSAGE:
            queue = self.stack_pop()
            result = self.get_temp_var()
            
            # Create receive request
            self.c_code += f"{indent}void* {result} = invoke_effect(EFFECT_RECEIVE, {queue});\n"
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
                self.c_code += f"{indent}float {temp_var} = {value}f;\n"
            elif isinstance(value, str):
                # Escape special characters in string
                escaped_value = value.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
                self.c_code += f'{indent}const char* {temp_var} = "{escaped_value}";\n'
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

        elif opcode == Opcode.END:
            # Clean up any CUDA variables in this scope
            for var in self.cuda_vars:
                self.c_code += f"{indent}cudaFree({var}_d);\n"
            
            # Close the current scope block
            if self.current_scope:
                self.indent_level -= 1
                self.c_code += f"{indent}}}\n"
                self.current_scope = None
                
        elif opcode == Opcode.CREATE_DOMAIN:
            # Get the data to wrap in a domain
            data = self.stack_pop()
            drop_value = instr.operands[0]
            value_needs_drop = instr.operands[1]
            temp_var = self.get_temp_var()
            self.c_code += f"{indent}domain_t* {temp_var} = create_domain((void*){data}, {drop_value}, {value_needs_drop});\n"
            self.stack_push(temp_var)

        elif opcode == Opcode.ACQUIRE_DOMAIN:
            # Get domain and acquire exclusive access
            domain = self.stack_pop()
            temp_var = self.get_temp_var()
            self.c_code += f"{indent}void* {temp_var} = domain_acquire({domain});\n"
            self.stack_push(temp_var)

        elif opcode == Opcode.RELEASE_DOMAIN:
            # Release domain after use
            domain = self.stack_pop()
            self.c_code += f"{indent}domain_release({domain});\n"

        elif opcode == Opcode.TRANSFER_DOMAIN:
            # Transfer domain ownership to another thread
            thread = self.stack_pop()
            domain = self.stack_pop()
            self.c_code += f"{indent}transfer_domain({domain}, {thread});\n"

        elif opcode == Opcode.LOAD_INDIRECT:
            # Load value through pointer
            addr = self.stack_pop()
            value = self.get_temp_var()
            self.c_code += f"{indent}int {value} = *({addr});\n"
            self.stack_push(value)

        elif opcode == Opcode.STORE_INDIRECT:
            # Store value through pointer
            value = self.stack_pop()
            addr = self.stack_pop()
            self.c_code += f"{indent}*({addr}) = {value};\n" 

        elif opcode == Opcode.DUP:
            value = self.stack[-1]
            self.stack_push(value)
        elif opcode == Opcode.POP:
            self.stack_pop()
        elif opcode == Opcode.PRINT: # Technically effectful but useful to debug
            value = self.stack_pop()
            # Check if value is a string constant
            if any(f'const char* {value} =' in line or f'char* {value} =' in line for line in self.c_code.split('\n')):
                self.c_code += f"{indent}printf(\"%s\\n\", {value});\n"
            else:
                self.c_code += f"{indent}printf(\"%d\\n\", {value});\n"
        elif opcode == Opcode.MALLOC:
            size = self.stack_pop()
            ptr = self.get_temp_var()
            self.c_code += f"{indent}void* {ptr} = malloc({size});\n"
            self.stack_push(ptr)
            
        elif opcode == Opcode.FREE:
            ptr = self.stack_pop()
            self.c_code += f"{indent}free({ptr});\n"
            
        elif opcode == Opcode.TRACK_DROP:
            var_name = instr.operands[0]
            needs_drop = instr.operands[1]
            if needs_drop:
                self.drop_tracker.track_var(var_name, needs_drop=True)
                
        elif opcode == Opcode.MOVE:
            src = instr.operands[0]
            dst = instr.operands[1]
            if src in self.drop_tracker.drop_flags:
                self.drop_tracker.move_var(src)
                self.drop_tracker.track_var(dst, needs_drop=True)
            
        elif opcode == Opcode.LOAD_INDIRECT:
            # Load value through pointer with type safety
            addr_type = self.type_stack.pop()
            if not addr_type.is_pointer():
                raise TypeError("LOAD_INDIRECT requires pointer operand")
            
            value_type = addr_type.pointed_type
            ptr = self.stack_pop()
            value = self.get_temp_var()
            self.c_code += f"{indent}{value_type.c_name} {value} = *({value_type.c_name}*){ptr};\n"
            self.stack_push(value)
            self.type_stack.append(value_type)
            
        elif opcode == Opcode.STORE_INDIRECT:
            # Store value through pointer with type safety
            value_type = self.type_stack.pop()
            addr_type = self.type_stack.pop()
            if not addr_type.is_pointer():
                raise TypeError("STORE_INDIRECT requires pointer operand")
                
            if not value_type.can_assign_to(addr_type.pointed_type):
                raise TypeError(f"Cannot store {value_type} to pointer of {addr_type.pointed_type}")
                
            value = self.stack_pop()
            ptr = self.stack_pop()
            self.c_code += f"{indent}*({addr_type.pointed_type.c_name}*){ptr} = {value};\n"
        elif opcode == Opcode.EXCLAVE:
            # Move value from stack to caller's region and return
            value = self.stack_pop()
            self.c_code += f"{indent}// Move value to caller's region\n"
            self.c_code += f"{indent}{value} = move_to_caller_region({value});\n"
            self.c_code += f"{indent}return {value};\n"
        
        elif opcode == Opcode.JOIN_THREAD:
            thread = self.stack_pop()
            result = self.get_temp_var()
            
            # Create join request
            self.c_code += f"{indent}join_request_t req = {{.thread = {thread}}};\n"
            self.c_code += f"{indent}void* {result} = invoke_effect(EFFECT_JOIN, &req);\n"
            self.stack_push(result)

        elif opcode == Opcode.ALLOC_DOMAIN:
            size = self.stack_pop()
            domain = self.get_temp_var()
            
            # Create alloc request
            self.c_code += f"{indent}alloc_request_t req = {{.size = {size}}};\n"
            self.c_code += f"{indent}domain_t* {domain} = invoke_effect(EFFECT_ALLOC, &req);\n"
            self.stack_push(domain)

        elif opcode == Opcode.FREE_DOMAIN:
            domain = self.stack_pop()
            self.c_code += f"{indent}invoke_effect(EFFECT_FREE, {domain});\n"

        elif opcode == Opcode.MOVE_DOMAIN:
            from_domain = self.stack_pop()
            to_domain = self.stack_pop()
            
            # Create move request
            self.c_code += f"{indent}move_request_t req = {{.from = {from_domain}, .to = {to_domain}}};\n"
            self.c_code += f"{indent}invoke_effect(EFFECT_MOVE, &req);\n"

        elif opcode == Opcode.BORROW_DOMAIN:
            domain = self.stack_pop()
            result = self.get_temp_var()
            
            # Create borrow request
            self.c_code += f"{indent}void* {result} = invoke_effect(EFFECT_BORROW, {domain});\n"
            self.stack_push(result)

    def get_temp_var(self):
        temp_var = f"t{self.temp_var_count}"
        self.temp_var_count += 1
        self.vars.add(temp_var)
        return temp_var

    def stack_push(self, value):
        self.stack.append(value)

    def stack_pop(self):
        if not self.stack:
            raise Exception("Stack underflow")
        return self.stack.pop()

    def gen_thread_types(self):
        """Generate C code for thread-related types"""
        self.c_code += """
// Thread handle
typedef struct {
    pthread_t thread;
    bool detached;
} thread_t;

// Task representation
typedef struct {
    void* (*f)(void*);
    void* arg;
    future_t* future;
} task_t;

// Queue operations
typedef struct queue_node {
    void* data;
    struct queue_node* next;
} queue_node_t;

typedef struct {
    queue_node_t* head;
    queue_node_t* tail;
    pthread_mutex_t mutex;
} queue_t;

queue_t* queue_create() {
    queue_t* queue = malloc(sizeof(queue_t));
    queue->head = NULL;
    queue->tail = NULL;
    pthread_mutex_init(&queue->mutex, NULL);
    return queue;
}

bool queue_empty(queue_t* queue) {
    return queue->head == NULL;
}

void queue_push(queue_t* queue, void* data) {
    queue_node_t* node = malloc(sizeof(queue_node_t));
    node->data = data;
    node->next = NULL;
    
    if (queue->tail == NULL) {
        queue->head = node;
        queue->tail = node;
    } else {
        queue->tail->next = node;
        queue->tail = node;
    }
}

void* queue_pop(queue_t* queue) {
    if (queue->head == NULL) {
        return NULL;
    }
    
    queue_node_t* node = queue->head;
    void* data = node->data;
    queue->head = node->next;
    
    if (queue->head == NULL) {
        queue->tail = NULL;
    }
    
    free(node);
    return data;
}

// Continuation type for effect handlers
typedef struct {
    jmp_buf env;
    void* result;
} continuation_t;

void resume(continuation_t* k, void* result) {
    k->result = result;
    longjmp(k->env, 1);
}

// Destructor for domain values
void free_domain(domain_t* domain) {
    if (domain) {
        // First clean up the contained value if needed
        if (domain->value_needs_drop) {
            domain->drop_value(domain->value);
        }
        free(domain);
    }
}

// Destructor for thread context
void free_thread_context(thread_context_t* ctx) {
    if (ctx) {
        // Clean up message queues
        if (ctx->queues) {
            free_message_queue(ctx->queues);
        }
        // Clean up thread-local storage
        if (ctx->locals) {
            free(ctx->locals);
        }
        free(ctx);
    }
}

// Destructor for thread pool
void free_thread_pool(thread_pool_t* pool) {
    if (pool) {
        // Clean up threads
        if (pool->threads) {
            free(pool->threads);
        }
        // Clean up task queue
        if (pool->tasks) {
            while (!queue_empty(pool->tasks)) {
                task_t* task = queue_pop(pool->tasks);
                if (task) {
                    free(task);
                }
            }
            free_message_queue(pool->tasks);
        }
        free(pool);
    }
}

// Destructor for future
void free_future(future_t* future) {
    if (future) {
        // Clean up result if needed
        if (future->result_needs_drop && future->result) {
            future->drop_result(future->result);
        }
        free(future);
    }
}
"""

    def gen_thread_handlers(self):
        """Generate C code for thread effect handlers"""
        self.c_code += """
// Thread pool implementation
typedef struct {
    pthread_t* threads;
    int size;
    queue_t* task_queue;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    bool shutdown;
} thread_pool_t;

// Future implementation
typedef struct {
    void* result;
    bool completed;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    void (*drop_result)(void*);
    bool result_needs_drop;
} future_t;

// Thread effect handlers
static thread_local continuation_t* thread_k;

void* thread_spawn_handler(void* (*f)(void*), void* arg) {
    pthread_t thread;
    pthread_create(&thread, NULL, f, arg);
    thread_t* result = malloc(sizeof(thread_t));
    result->thread = thread;
    result->detached = false;
    resume(thread_k, result);
}

void thread_join_handler(thread_t* thread) {
    if (!thread->detached) {
        pthread_join(thread->thread, NULL);
    }
    free(thread);
    resume(thread_k, NULL);
}

void thread_yield_handler() {
    sched_yield();
    resume(thread_k, NULL);
}

void thread_detach_handler(thread_t* thread) {
    if (!thread->detached) {
        pthread_detach(thread->thread);
        thread->detached = true;
    }
    resume(thread_k, NULL);
}

thread_t* thread_current_handler() {
    thread_t* result = malloc(sizeof(thread_t));
    result->thread = pthread_self();
    result->detached = false;
    resume(thread_k, result);
}

// Thread pool effect handlers
static thread_pool_t* global_pool = NULL;
static thread_local continuation_t* pool_k;

void* pool_worker(void* arg) {
    thread_pool_t* pool = (thread_pool_t*)arg;
    while (true) {
        pthread_mutex_lock(&pool->mutex);
        while (queue_empty(pool->task_queue) && !pool->shutdown) {
            pthread_cond_wait(&pool->cond, &pool->mutex);
        }
        if (pool->shutdown && queue_empty(pool->task_queue)) {
            pthread_mutex_unlock(&pool->mutex);
            break;
        }
        task_t* task = queue_pop(pool->task_queue);
        pthread_mutex_unlock(&pool->mutex);
        
        // Execute task
        task->f(task->arg);
        free(task);
    }
    return NULL;
}

void pool_set_size_handler(int size) {
    if (global_pool == NULL) {
        global_pool = malloc(sizeof(thread_pool_t));
        global_pool->threads = malloc(sizeof(pthread_t) * size);
        global_pool->size = size;
        global_pool->task_queue = queue_create();
        pthread_mutex_init(&global_pool->mutex, NULL);
        pthread_cond_init(&global_pool->cond, NULL);
        global_pool->shutdown = false;
        
        // Start worker threads
        for (int i = 0; i < size; i++) {
            pthread_create(&global_pool->threads[i], NULL, pool_worker, global_pool);
        }
    }
    resume(pool_k, NULL);
}

future_t* pool_submit_handler(void* (*f)(void*), void* arg) {
    if (global_pool == NULL) {
        pool_set_size_handler(4); // Default size
    }
    
    future_t* future = malloc(sizeof(future_t));
    future->completed = false;
    future->result = NULL;
    future->result_needs_drop = false;
    pthread_mutex_init(&future->mutex, NULL);
    pthread_cond_init(&future->cond, NULL);
    
    task_t* task = malloc(sizeof(task_t));
    task->f = f;
    task->arg = arg;
    task->future = future;
    
    pthread_mutex_lock(&global_pool->mutex);
    queue_push(global_pool->task_queue, task);
    pthread_cond_signal(&global_pool->cond);
    pthread_mutex_unlock(&global_pool->mutex);
    
    resume(pool_k, future);
}

void* pool_await_handler(future_t* future) {
    pthread_mutex_lock(&future->mutex);
    while (!future->completed) {
        pthread_cond_wait(&future->cond, &future->mutex);
    }
    void* result = future->result;
    pthread_mutex_unlock(&future->mutex);
    
    // Cleanup future
    pthread_mutex_destroy(&future->mutex);
    pthread_cond_destroy(&future->cond);
    free(future);
    
    resume(pool_k, result);
}
"""

    def install_thread_handlers(self):
        """Install handlers for thread effects"""
        self.c_code += """
// Install thread effect handlers
void install_thread_handlers() {
    continuation_t k;
    if (setjmp(k.env) == 0) {
        thread_k = &k;
        
        // Install Thread effect handlers
        register_effect_handler("Thread.spawn", thread_spawn_handler);
        register_effect_handler("Thread.join", thread_join_handler);
        register_effect_handler("Thread.yield", thread_yield_handler);
        register_effect_handler("Thread.detach", thread_detach_handler);
        register_effect_handler("Thread.current", thread_current_handler);
        
        // Install ThreadPool effect handlers
        register_effect_handler("ThreadPool.submit", pool_submit_handler);
        register_effect_handler("ThreadPool.await", pool_await_handler);
        register_effect_handler("ThreadPool.set_pool_size", pool_set_size_handler);
    }
}

// Effect handler registry
typedef struct {
    const char* name;
    void* (*handler)(void*);
} effect_handler_t;

#define MAX_HANDLERS 100
static effect_handler_t handler_registry[MAX_HANDLERS];
static int num_handlers = 0;

void register_effect_handler(const char* name, void* (*handler)(void*)) {
    if (num_handlers < MAX_HANDLERS) {
        handler_registry[num_handlers].name = name;
        handler_registry[num_handlers].handler = handler;
        num_handlers++;
    }
}

void* handle_effect(const char* effect_name, void* arg) {
    for (int i = 0; i < num_handlers; i++) {
        if (strcmp(handler_registry[i].name, effect_name) == 0) {
            return handler_registry[i].handler(arg);
        }
    }
    fprintf(stderr, "No handler found for effect: %s\\n", effect_name);
    exit(1);
}
"""

class DropFlag:
    def __init__(self, name, needs_drop=False):
        self.name = name
        self.needs_drop = needs_drop
        self.moved = False

    def mark_moved(self):
        self.moved = True

    def is_valid(self):
        return not self.moved

class DropTracker:
    def __init__(self):
        self.drop_flags = {}
        self.scope_stack = []

    def enter_scope(self):
        self.scope_stack.append(set())

    def exit_scope(self):
        scope = self.scope_stack.pop()
        for var in scope:
            if var in self.drop_flags and self.drop_flags[var].needs_drop:
                # Generate drop code for variable
                pass

    def track_var(self, name, needs_drop=False):
        self.drop_flags[name] = DropFlag(name, needs_drop)
        if self.scope_stack:
            self.scope_stack[-1].add(name)

    def move_var(self, name):
        if name in self.drop_flags:
            self.drop_flags[name].mark_moved()

    def is_valid(self, name):
        return name in self.drop_flags and self.drop_flags[name].is_valid()
