# tape_vm.py

from enum import Enum, auto
import threading
from queue import Queue
from vm_manager import register_vm, get_vm_by_thread_id

class Opcode(Enum):
    # Basic opcodes
    LOAD_CONST = auto()
    LOAD_VAR = auto()
    STORE_VAR = auto()
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
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
    PERFORM_EFFECT = auto()
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

class TapeVM:
    def __init__(self, instructions, thread_id=None, message_queue=None):
        self.instructions = instructions
        self.pc = 0  # Program counter
        self.stack = []
        self.vars = {}
        self.functions = {}
        self.labels = {}
        self.call_stack = []
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
            value = self.vars.get(instr.operands[0], 0)
            self.stack.append(value)
        elif opcode == Opcode.STORE_VAR:
            value = self.stack.pop()
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
        elif opcode == Opcode.CALL_FUNC:
            func_name = instr.operands[0]
            func_instructions = self.functions.get(func_name)
            if func_instructions is None:
                raise Exception(f"Function '{func_name}' not defined")
            # Save current state
            self.call_stack.append((self.pc, self.vars.copy(), self.stack.copy()))
            # Set up new state
            self.pc = -1
            self.instructions = func_instructions
            self.vars = {}
            self.stack = []
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
        elif opcode == Opcode.PERFORM_EFFECT:
            effect_name = instr.operands[0]
            arg_count = instr.operands[1]
            args = [self.stack.pop() for _ in range(arg_count)][::-1]
            if effect_name in self.effect_handlers:
                continuation = self.create_continuation()
                handler_instructions = self.effect_handlers[effect_name]
                handler_vm = TapeVM(
                    handler_instructions,
                    message_queue=self.message_queue,
                    functions=self.functions,
                    labels=self.labels,
                    effect_handlers=self.effect_handlers.copy()
                )
                handler_vm.stack = args + [continuation]
                handler_vm.vars = self.vars.copy()
                handler_vm.run()
                result = handler_vm.stack.pop()
                self.stack.append(result)
            else:
                raise Exception(f"No handler for effect '{effect_name}'")
        # Implement other opcodes...
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

