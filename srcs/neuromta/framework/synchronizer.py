from typing import Any, Callable
from neuromta.framework.logger import logger
from neuromta.framework.memory_handle import Pointer


__all__ = ["VariableHandle", "FIFOBufferHandle"]
    

class VariableHandle:
    class ActionCondition:
        @classmethod
        def equals_to(cls, value: int) -> Callable[[int], bool]:
            return lambda x: x == value
        
        @classmethod
        def greater_equal(cls, value: int) -> Callable[[int], bool]:
            return lambda x: x >= value
        
        @classmethod
        def less_equal(cls, value: int) -> Callable[[int], bool]:
            return lambda x: x <= value

        @classmethod
        def greater_than(cls, value: int) -> Callable[[int], bool]:
            return lambda x: x > value
        
        @classmethod
        def less_than(cls, value: int) -> Callable[[int], bool]:
            return lambda x: x < value
    
    def __init__(self, handle_name: str, initial_value: int=0):
        self.handle_name = handle_name
        
        self._value: int = initial_value
        self._action_methods: list[tuple[Callable[[int], bool], Callable]] = []
    
    def _run_actions(self):
        processed = []
        for i, (condition, action) in enumerate(self._action_methods):
            if condition(self._value):
                action()
                processed.append(i)
                
        for i in reversed(processed):
            del self._action_methods[i]
            
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
                
            self._action_methods.append((self.ActionCondition.equals_to(cmp_value), _action))
            
    def atomic_wait(self, expected_value: int, callback: Callable):
        if self._value == expected_value:
            callback()
        else:
            self._action_methods.append((self.ActionCondition.equals_to(expected_value), callback))
            
    def atimic_wait_conditional(self, condition: Callable[[int], bool], callback: Callable):
        if condition(self._value):
            callback()
        else:
            self._action_methods.append((condition, callback))
    
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
    
    
class FIFOBufferHandle:
    def __init__(self, handle_name: str, depth: int, entry_size: int):
        self.handle_name = handle_name
        self.depth = depth
        self.entry_size = entry_size
        self._mem_ptr: Pointer = Pointer()  # empty pointer, to be allocated by the core
        
        self._rd_offset = 0
        self._wr_offset = 0
        self._ref_counts: list[VariableHandle] = [VariableHandle(f"{handle_name}_ref_counter_{i}", initial_value=0) for i in range(depth)]
        self._action_methods: list[tuple[Callable[[FIFOBufferHandle], bool], Callable]] = []
        self._is_full = False
        self._is_empty = True
        self._global_counter: VariableHandle = VariableHandle(f"{handle_name}_global_counter", initial_value=0)
        
    def _run_actions(self):
        processed = []
        for i, (condition, action) in enumerate(self._action_methods):
            if condition(self):
                action()
                processed.append(i)
                
        for i in reversed(processed):
            del self._action_methods[i]
            
    @staticmethod
    def _entry_vacant_condition(buf: 'FIFOBufferHandle', entry_id: VariableHandle) -> bool:
        entry_idx = entry_id.value % buf.depth
        if entry_idx != buf._wr_offset:
            return False
        return buf._ref_counts[entry_idx].value == 0
    
    @staticmethod
    def _entry_valid_condition(buf: 'FIFOBufferHandle', entry_id: VariableHandle) -> bool:
        entry_idx = entry_id.value % buf.depth
        if entry_id.value >= buf._global_counter.value:
            return False
        if buf._is_full:
            return buf._ref_counts[entry_idx].value > 0
        if buf._rd_offset <= entry_idx < buf._wr_offset:
            return buf._ref_counts[entry_idx].value > 0
        if buf._wr_offset < buf._rd_offset:
            if entry_idx >= buf._rd_offset or entry_idx < buf._wr_offset:
                return buf._ref_counts[entry_idx].value > 0
        return False
        
    def wait_until_vacant(self, entry_id: VariableHandle, callback: Callable):
        self.pop_all_evictable()
        
        if self._entry_vacant_condition(self, entry_id):
            callback()
        else:
            self._action_methods.append((lambda buf: self._entry_vacant_condition(buf, entry_id), callback))

    def wait_until_valid(self, entry_id: VariableHandle, callback: Callable):
        self.pop_all_evictable()
        
        if self._entry_valid_condition(self, entry_id):
            callback()
        else:
            self._action_methods.append((lambda buf: self._entry_valid_condition(buf, entry_id), callback))

    def write_entry(self, entry_id: VariableHandle, ref_count: int):
        if not self._entry_vacant_condition(self, entry_id):
            raise Exception(f"Attempting to write to a non-vacant entry (entry_id={entry_id.value}) in FIFO buffer '{self.handle_name}'.")
        
        entry_idx = entry_id.value % self.depth
        self._ref_counts[entry_idx].value = ref_count
        self._global_counter.atomic_increase(1)
        self._wr_offset = (self._wr_offset + 1) % self.depth
        
        self._is_empty = False
        if self._rd_offset == self._wr_offset:
            self._is_full = True
        
        self.pop_all_evictable()
        
    def read_entry(self, entry_id: VariableHandle):
        if not self._entry_valid_condition(self, entry_id):
            raise Exception(f"Attempting to read from a non-valid entry (entry_id={entry_id.value}) in FIFO buffer '{self.handle_name}'.")
        
        entry_idx = entry_id.value % self.depth
        self._ref_counts[entry_idx].atomic_increase(-1)
        
        self.pop_all_evictable()
        
    def pop_all_evictable(self):
        if self._is_empty:
            return
        
        _cnt = 0
        while self._ref_counts[self._rd_offset].value == 0:
            self._rd_offset = (self._rd_offset + 1) % self.depth
            _cnt += 1
            if self._rd_offset == self._wr_offset:
                break
        
        if _cnt > 0:
            self._is_full = False
            if self._rd_offset == self._wr_offset:
                self._is_empty = True
        
        self._run_actions()
        
    @property
    def mem_ptr(self) -> Pointer:
        return self._mem_ptr
       
    def get_ptr(self, entry_id: VariableHandle | int) -> Pointer:
        entry_id = entry_id.value if isinstance(entry_id, VariableHandle) else entry_id
        entry_id = entry_id % self.depth
        return Pointer(self.mem_ptr.addr + entry_id * self.entry_size)
    
    def get_entry_idx(self, ptr: Pointer) -> int:
        entry_idx = (ptr.addr - self.mem_ptr.addr) // self.entry_size
        if entry_idx < 0 or entry_idx >= self.depth:
            raise Exception(f"Pointer {ptr} is out of range for the FIFO buffer '{self.handle_name}'.")
        return entry_idx
    
    def get_ref_count(self, entry_id: VariableHandle | int) -> VariableHandle:
        entry_id = entry_id.value if isinstance(entry_id, VariableHandle) else entry_id
        entry_id = entry_id % self.depth
        return self._ref_counts[entry_id]
