# tape_vm.py

from enum import Enum, auto
import threading
from queue import Queue

class Opcode(Enum):
    # Basic opcodes
    LOAD_CONST = auto()
    LOAD_VAR = auto()
    STORE_VAR = auto()
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    CREATE_FUNC = auto()
    CREATE_CLOSURE = auto()
    CALL_FUNC = auto()
    RETURN = auto()
    # Ownership
    MOVE_VAR = auto()
    # SIMD Operations
    VEC_ADD = auto()
    VEC_SUB = auto()
    VEC_MUL = auto()
    VEC_DIV = auto()
    CREATE_VECTOR = auto()
    # Structs and Enums
    CREATE_STRUCT = auto()
    ACCESS_FIELD = auto()
    CREATE_VARIANT = auto()
    MATCH_VARIANT = auto()
    # Multithreading
    SPAWN_THREAD = auto()
    SEND_MESSAGE = auto()
    RECEIVE_MESSAGE = auto()
    # Effects
    CREATE_CONTINUATION = auto()
    CALL_EFFECT_HANDLER = auto()
    INVOKE_CONTINUATION = auto()
    SET_HANDLER = auto()
    UNSET_HANDLER = auto()
    RESUME = auto()
    # Device Operations
    TO_DEVICE = auto()
    FROM_DEVICE = auto()
    # Control flow
    JUMP_IF_FALSE = auto()
    JUMP = auto()
    LABEL = auto()
    DUP = auto()
    POP = auto()

class Instruction:
    def __init__(self, opcode, *operands):
        self.opcode = opcode
        self.operands = operands

    def __repr__(self):
        return f"{self.opcode.name} {', '.join(map(str, self.operands))}"

class Frame:
    def __init__(self, return_address, scope_name, environment=None):
        self.return_address = return_address
        self.scope_name = scope_name
        self.local_vars = {}
        self.resources = []  # List of resources to deallocate
        if environment:
            self.local_vars.update(environment)

class FunctionObject:
    def __init__(self, code_label, environment):
        self.code_label = code_label  # Label to jump to
        self.environment = environment  # Captured variables

class TapeVM:
    """ An Instruction TapeVM with a Stack"""
    def __init__(self, instructions, thread_id=None, message_queue=None):
        self.instructions = instructions
        self.pc = 0  # Program counter
        self.stack = []
        self.vars = {}
        self.functions = {}
        self.labels = {}
        self.call_stack = []
        self.frames = []
        self.effect_handlers = {}
        self.thread_id = thread_id or generate_thread_id()
        self.message_queue = message_queue or Queue()
        register_vm(self)

    def run(self):
        self.preprocess_labels()
        while self.pc < len(self.instructions):
            instr = self.instructions[self.pc]
            self.execute(instr)
            self.pc += 1

    def preprocess_labels(self):
        for idx, instr in enumerate(self.instructions):
            if instr.opcode == Opcode.LABEL:
                label_name = instr.operands[0]
                self.labels[label_name] = idx

    def execute(self, instr):
        opcode = instr.opcode

        if opcode == Opcode.LOAD_CONST:
            self.stack.append(instr.operands[0])
        elif opcode == Opcode.LOAD_VAR:
            name = instr.operands[0]
            for frame in reversed(self.frames):
                if name in frame.local_vars:
                    self.stack.append(frame.local_vars[name])
                    break
            else:
                raise Exception(f"Variable '{name}' not found")            
        elif opcode == Opcode.STORE_VAR:
            name = instr.operands[0]
            self.frames[-1].local_vars[name] = self.stack.pop()
            self.pc += 1
            self.vars[instr.operands[0]] = value
        elif opcode == Opcode.ADD:
            right = self.stack.pop()
            left = self.stack.pop()
            self.stack.append(left + right)
        elif opcode == Opcode.SUB:
            right = self.stack.pop()
            left = self.stack.pop()
            self.stack.append(left - right)
        elif opcode == Opcode.MUL:
            right = self.stack.pop()
            left = self.stack.pop()
            self.stack.append(left * right)
        elif opcode == Opcode.DIV:
            right = self.stack.pop()
            left = self.stack.pop()
            self.stack.append(left // right)
        if opcode == OpCode.CREATE_FUNC:
            label = instr.operands[0]
            func_obj = FunctionObject(label, {})
            self.stack.append(func_obj)
            self.pc += 1

        elif opcode == OpCode.CREATE_CLOSURE:
            label = instr.operands[0]
            num_captured = instr.operands[1]
            captured_vars = [self.stack.pop() for _ in range(num_captured)][::-1]
            environment = {f'var_{i}': val for i, val in enumerate(captured_vars)}
            func_obj = FunctionObject(label, environment)
            self.stack.append(func_obj)
            self.pc += 1
            
        elif opcode == Opcode.CALL_FUNC:
            num_args = instr.operands[0]
            args = [self.stack.pop() for _ in range(num_args)][::-1]
            func_obj = self.stack.pop()
            if not isinstance(func_obj, FunctionObject):
                raise Exception("Attempted to call a non-function object")
            # Save current execution state
            return_address = self.pc + 1
            self.frames.append(Frame(return_address, func_obj.code_label, func_obj.environment))
            # Set up local variables for the function
            frame = self.frames[-1]
            # Store arguments as local variables
            for i, arg in enumerate(args):
                frame.local_vars[f'arg_{i}'] = arg
            # Jump to function code
            self.pc = self.labels[func_obj.code_label]
        elif opcode == Opcode.RETURN:
            # Restore previous state
            self.pc, self.vars, self.stack = self.call_stack.pop()
        elif opcode == Opcode.MOVE_VAR:
            var_name = instr.operands[0]
            value = self.vars.pop(var_name, None)
            self.stack.append(value)
        elif opcode == Opcode.CREATE_STRUCT:
            struct_name = instr.operands[0]
            field_values = instr.operands[1]
            struct_instance = {'__struct_name__': struct_name, **field_values}
            self.stack.append(struct_instance)

        elif opcode == Opcode.ACCESS_FIELD:
            field_name = instr.operands[0]
            struct_instance = self.stack.pop()
            if field_name in struct_instance:
                self.stack.append(struct_instance[field_name])
            else:
                raise Exception(f"Field '{field_name}' does not exist.")
        elif opcode == Opcode.VEC_ADD:
            right = self.stack.pop()
            left = self.stack.pop()
            result = [l + r for l, r in zip(left, right)]
            self.stack.append(result)
        elif opcode == Opcode.CREATE_VECTOR:
            base_type = instr.operands[0]
            size = instr.operands[1]
            elements = [self.stack.pop() for _ in range(size)][::-1]
            self.stack.append(elements)
        elif opcode == Opcode.SPAWN_THREAD:
            func = self.stack.pop()
            if not isinstance(func, str):
                raise Exception("Attempting to spawn a non-function")
            func_instructions = self.functions.get(func)
            if func_instructions is None:
                raise Exception(f"Function '{func}' not defined")
            # Create a new VM instance
            new_thread_vm = TapeVM(
                func_instructions,
                message_queue=Queue(),
                functions=self.functions,
                labels=self.labels,
                effect_handlers=self.effect_handlers.copy()
            )
            threading.Thread(target=new_thread_vm.run).start()
            # Push thread ID onto the stack
            self.stack.append(new_thread_vm.thread_id)
        # TODO: Remove explicit effect handling in favor of compiled jumps to labels
        if opcode == Opcode.CREATE_CONTINUATION:
                continuation_label = instr.operands[0]
                # Save the continuation (instructions after current pc)
                continuation_instructions = self.instructions[self.pc + 1:]
                self.continuations[continuation_label] = continuation_instructions
                self.pc += 1  # Move to next instruction
        # TODO: Remove explicit effect handling in favor of compiled jumps to labels
        elif opcode == Opcode.CALL_EFFECT_HANDLER:
            effect_name = instr.operands[0]
            arg_count = instr.operands[1]
            continuation_label = instr.operands[2]
            args = [self.stack.pop() for _ in range(arg_count)][::-1]
            # Retrieve the effect handler
            handler = self.effect_handlers.get(effect_name)
            if handler is None:
                raise Exception(f"Unknown effect: {effect_name}")
            # Call the effect handler, passing the continuation
            handler(self, *args, continuation_label)
            # Effect handler is responsible for invoking continuation
            break  # Stop execution; effect handler resumes as needed
        elif opcode == Opcode.LABEL:
            # No action needed; labels are markers
            self.pc += 1
        elif opcode == Opcode.INVOKE_CONTINUATION:
            continuation_label = instr.operands[0]
            continuation_instructions = self.continuations.get(continuation_label)
            if continuation_instructions is None:
                raise Exception(f"Unknown continuation: {continuation_label}")
            # Replace instructions and reset pc
            self.instructions = continuation_instructions
            self.pc = 0
        if opcode == 'JUMP':
            label = instr.operands[0]
            if label not in self.labels:
                raise Exception(f"Unknown label: {label}")
            return_address = self.pc + 1
            scope_name = label  # Use label as scope name
            self.frames.append(Frame(return_address, scope_name))
            self.pc = self.labels[label]
        elif opcode == 'END':
            # Pop the current frame and return if frames are used
            if not self.frames:
                raise Exception("Frame stack underflow!")
            frame = self.frames.pop()
            self.pc = self.frame.return_address
        elif opcode == 'PUSH':
            value = self.stack.pop()
            self.stack.append(value)
            self.pc += 1
        elif opcode == 'POP':
            self.stack.pop()
            self.pc += 1        
        # TODO: Implement other opcodes...
        else:
            raise Exception(f"Unknown opcode {opcode}")

    def create_continuation(self):
        # Capture current state
        continuation = {
            'pc': self.pc,
            'instructions': self.instructions,
            'vars': self.vars.copy(),
            'stack': self.stack.copy(),
            'functions': self.functions,
            'labels': self.labels,
            'effect_handlers': self.effect_handlers.copy(),
            'call_stack': self.call_stack.copy(),
        }
        return continuation

    def resume_continuation(self, continuation, value):
        # Restore state
        self.pc = continuation['pc']
        self.instructions = continuation['instructions']
        self.vars = continuation['vars']
        self.stack = continuation['stack']
        self.functions = continuation['functions']
        self.labels = continuation['labels']
        self.effect_handlers = continuation['effect_handlers']
        self.call_stack = continuation['call_stack']
        self.stack.append(value)

    def deallocate_frame(self, frame):
        # Clear local variables
        frame.local_vars.clear()
        # Deallocate resources
        for resource in frame.resources:
            resource.deallocate()
        # Additional cleanup if necessary

vm_registry = {}
thread_id_counter = 0
vm_registry_lock = threading.Lock()

def generate_thread_id():
    global thread_id_counter
    with vm_registry_lock:
        thread_id = thread_id_counter
        thread_id_counter += 1
    return thread_id

def register_vm(vm):
    with vm_registry_lock:
        vm_registry[vm.thread_id] = vm

def get_vm_by_thread_id(thread_id):
    with vm_registry_lock:
        return vm_registry.get(thread_id)

