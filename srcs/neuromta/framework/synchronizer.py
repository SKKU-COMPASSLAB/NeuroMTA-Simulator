from typing import Any, Callable


__all__ = ["LockHandle", "VariableHandle"]


class LockHandle:
    def __init__(self):
        self._owner: Any = None
        self._history: list[tuple[Any, Callable]] = []
        
    def acquire(self, owner: Any, callback: Callable):
        if self._owner is None:
            self._owner = owner
            callback()
        else:
            self._history.append((owner, callback))
            
    def release(self, owner: Any):
        if self._owner != owner:
            raise RuntimeError("Cannot release lock: not the owner")
        
        if len(self._history) > 0:
            next_owner, callback = self._history.pop(0)
            self._owner = next_owner
            callback()
        else:
            self._owner = None
            

class VariableHandle:
    def __init__(self, initial_value: int):
        self._value: int = initial_value
        self._actions: dict[int, list[Callable]] = {}
    
    def _run_actions(self):
        if self._value in self._actions:
            for action in self._actions[self._value]:
                action()
            del self._actions[self._value]
            
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
            if cmp_value not in self._actions:
                self._actions[cmp_value] = []
            
            def _action():
                self._value = new_value
                if callback is not None:
                    callback()
                self._run_actions()
                
            self._actions[cmp_value].append(_action)
            
    def atomic_wait(self, expected_value: int, callback: Callable):
        if self._value == expected_value:
            callback()
        else:
            if expected_value not in self._actions:
                self._actions[expected_value] = []
            self._actions[expected_value].append(callback)
    
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
