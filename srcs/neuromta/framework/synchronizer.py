from typing import Any, Callable
from neuromta.framework.logger import logger
from neuromta.framework.memory_handle import Pointer


__all__ = ["VariableHandle", "FIFOBufferHandle"]
    

class VariableHandle:
    class ActionCondition:
        def __init__(self, condition: Callable[[int], bool]):
            self.condition = condition
            self.action_id = condition.__condition_method_name if hasattr(condition, "__condition_method_name") else id(condition)

        def __call__(self, x: int):
            if isinstance(x, VariableHandle):
                x = x.value
            return self.condition(x)
        
        @property
        def signature(self):
            return self.action_id

        def __repr__(self):
            return f"ActionCondition({self.action_id})"
    
    def __init__(self, handle_name: str, initial_value: int=0):
        self.handle_name = handle_name
        
        self._value: int = initial_value
        self._value_conditional_action_methods: dict[Any, tuple[VariableHandle.ActionCondition, list[Callable]]] = {}

    def add_conditional_action(self, condition: 'VariableHandle.ActionCondition', action: Callable):
        if condition.signature not in self._value_conditional_action_methods:
            self._value_conditional_action_methods[condition.signature] = (condition, [])
        self._value_conditional_action_methods[condition.signature][1].append(action)
        
    def _run_actions(self):
        for signature in list(self._value_conditional_action_methods.keys()):
            condition, actions = self._value_conditional_action_methods[signature]
            if condition(self._value):
                for action in actions:
                    action()
                del self._value_conditional_action_methods[signature]

    def atomic_update(self, value: int):
        self._value = value
        self._run_actions()
    
    def atomic_compare_and_swap(self, cmp_value: int, new_value: int, callback: Callable = None):
        if self._value == cmp_value:
            self._value = new_value
            if callback is not None:
                callback()
            self._run_actions()
        else:
            def _action():
                self._value = new_value
                if callback is not None:
                    callback()
                self._run_actions()
                
            self.add_conditional_action(self.equals_to(cmp_value), _action)
            
    def atomic_wait(self, expected_value: int, callback: Callable):
        if self._value == expected_value:
            callback()
        else:
            self.add_conditional_action(self.equals_to(expected_value), callback)
            
    def atomic_wait_conditional(self, condition: 'VariableHandle.ActionCondition', callback: Callable):
        if not isinstance(condition, VariableHandle.ActionCondition):
            raise Exception(f"Condition for atomic_wait_conditional must be an instance of VariableHandle.ActionCondition, but got {type(condition).__name__}.")
        
        if condition(self._value):
            callback()
        else:
            self.add_conditional_action(condition, callback)
    
    def atomic_increase(self, increment: int, callback: Callable = None):
        self._value += increment
        if callback is not None:
            callback()
        self._run_actions()
        
    @property
    def value(self) -> int:
        return self._value
    
    @value.setter
    def value(self, new_value: int):
        self.atomic_update(new_value)
    
    def __repr__(self):
        return f"VariableHandle(name={self.handle_name}, value={self._value})"
    
    def __str__(self):
        return f"VariableHandle(name={self.handle_name}, value={self._value})"
    
    @classmethod
    def tmp(cls, initial_value: int=0) -> "VariableHandle":
        return cls(handle_name="tmp", initial_value=initial_value)
    
    def equals_to(self, value: int) -> 'VariableHandle.ActionCondition':
        method = lambda x: x == (value.value if isinstance(value, VariableHandle) else value)
        method.__condition_method_name = f"equals_to_{value}"
        return self.ActionCondition(method)
    
    def greater_equal(self, value: int) -> 'VariableHandle.ActionCondition':
        method = lambda x: x >= (value.value if isinstance(value, VariableHandle) else value)
        method.__condition_method_name = f"greater_equal_{value}"
        return self.ActionCondition(method)

    def less_equal(self, value: int) -> 'VariableHandle.ActionCondition':
        method = lambda x: x <= (value.value if isinstance(value, VariableHandle) else value)
        method.__condition_method_name = f"less_equal_{value}"
        return self.ActionCondition(method)

    def greater_than(self, value: int) -> 'VariableHandle.ActionCondition':
        method = lambda x: x > (value.value if isinstance(value, VariableHandle) else value)
        method.__condition_method_name = f"greater_than_{value}"
        return self.ActionCondition(method)

    def less_than(self, value: int) -> 'VariableHandle.ActionCondition':
        method = lambda x: x < (value.value if isinstance(value, VariableHandle) else value)
        method.__condition_method_name = f"less_than_{value}"
        return self.ActionCondition(method)
    
    
class FIFOBufferHandle:
    def __init__(self, handle_name: str, depth: int, entry_size: int):
        self.handle_name = handle_name
        self.depth = depth
        self.entry_size = entry_size
        self._mem_ptr: Pointer = Pointer()  # empty pointer, to be allocated by the core
        
        self._ref_counts: list[VariableHandle] = [VariableHandle(f"{handle_name}_ref_counter_{i}", initial_value=0) for i in range(depth)]
        self._global_counter: VariableHandle = VariableHandle(f"{handle_name}_global_counter", initial_value=0)
        
        self._entry_vacant_action_methods: dict[int, list[Callable]] = {}
        self._entry_valid_action_methods: dict[int, list[Callable]] = {}
        
    def _run_actions(self):
        for entry_id in list(self._entry_vacant_action_methods.keys()):
            if self._entry_vacant_condition(self, entry_id):
                for action in self._entry_vacant_action_methods[entry_id]:
                    action()
                del self._entry_vacant_action_methods[entry_id]
                
        for entry_id in list(self._entry_valid_action_methods.keys()):
            if self._entry_valid_condition(self, entry_id):
                for action in self._entry_valid_action_methods[entry_id]:
                    action()
                del self._entry_valid_action_methods[entry_id]
            
    @staticmethod
    def _entry_vacant_condition(buf: 'FIFOBufferHandle', entry_id: int) -> bool:
        entry_idx = entry_id % buf.depth
        return buf._ref_counts[entry_idx].value == 0
    
    @staticmethod
    def _entry_valid_condition(buf: 'FIFOBufferHandle', entry_id: int) -> bool:
        if entry_id >= buf._global_counter.value:
            return False
        entry_idx = entry_id % buf.depth
        return buf._ref_counts[entry_idx].value > 0
    
    def wait_until_vacant(self, entry_id: int, callback: Callable):
        self._run_actions()
        
        if self._entry_vacant_condition(self, entry_id):
            callback()
        else:
            if entry_id not in self._entry_vacant_action_methods:
                self._entry_vacant_action_methods[entry_id] = []
            self._entry_vacant_action_methods[entry_id].append(callback)

    def wait_until_valid(self, entry_id: int, callback: Callable):
        self._run_actions()
        
        if self._entry_valid_condition(self, entry_id):
            callback()
        else:
            if entry_id not in self._entry_valid_action_methods:
                self._entry_valid_action_methods[entry_id] = []
            self._entry_valid_action_methods[entry_id].append(callback)

    def write_entry(self, entry_id: int, ref_count: int):
        if not self._entry_vacant_condition(self, entry_id):
            raise Exception(f"Attempting to write to a non-vacant entry (entry_id={entry_id}) in FIFO buffer '{self.handle_name}'.")
        
        entry_idx = entry_id % self.depth
        self._ref_counts[entry_idx].value = ref_count
        self._global_counter.atomic_update(max(entry_id + 1, self._global_counter.value))
        
        self._run_actions()
        
    def read_entry(self, entry_id: int):
        if not self._entry_valid_condition(self, entry_id):
            raise Exception(f"Attempting to read from a non-valid entry (entry_id={entry_id}) in FIFO buffer '{self.handle_name}'.")
        
        entry_idx = entry_id % self.depth
        self._ref_counts[entry_idx].atomic_increase(-1)
        
        self._run_actions()
        
    @property
    def mem_ptr(self) -> Pointer:
        return self._mem_ptr
       
    def get_ptr(self, entry_id: int) -> Pointer:
        entry_id = entry_id % self.depth
        return Pointer(self.mem_ptr.addr + entry_id * self.entry_size)
    
    def get_entry_idx(self, ptr: Pointer) -> int:
        entry_idx = (ptr.addr - self.mem_ptr.addr) // self.entry_size
        if entry_idx < 0 or entry_idx >= self.depth:
            raise Exception(f"Pointer {ptr} is out of range for the FIFO buffer '{self.handle_name}'.")
        return entry_idx
    
    def get_ref_count(self, entry_id: int) -> VariableHandle:
        entry_id = entry_id % self.depth
        return self._ref_counts[entry_id]
    
    def __repr__(self):
        valid_slots = [i for i in range(self.depth) if self._ref_counts[i].value > 0]
        return f"FIFOBufferHandle(name={self.handle_name}, depth={self.depth}, entry_size={self.entry_size}, valid_slots={valid_slots})"
