import abc
import enum
import functools
import itertools
from typing import Callable, Sequence, Any

from neuromta.framework.logger import logger


__all__ = [
    # "SerializedCoreObjectState",
    "SerializableCoreObject",
    "Serializer",
    "new_global_serializer",
    "get_global_serializer",
    "check_global_serializer",
    "clear_global_serializer",
]


class Core: ...

class SerializedCoreObjectState:
    def __init__(self, state_type: type, state_data: dict):
        if not isinstance(state_type, type):
            state_type = type(state_type)
            
        self.state_type = state_type
        self.state_data = state_data
        
    def restore(self, core: 'Core'):
        restore_method = getattr(self.state_type, "from_state", None)
        if restore_method is None:
            return self.state_data
        return restore_method(core, self.state_data)
        
class SerializableCoreObject(abc.ABC):
    def get_state(self) -> dict:
        return self.__dict__.copy()
    
    @classmethod
    @abc.abstractmethod
    def from_state(cls, core: 'Core', state: dict) -> 'SerializableCoreObject':
        raise NotImplementedError("from_state method must be implemented by subclasses of SerializableCoreObject")
    
class SerializedCoreStateID(str):
    def __init__(self, state_id: str):
        self.state_id = state_id

class Serializer:
    def __init__(self):
        self.state_hub: dict[SerializedCoreStateID, SerializedCoreObjectState] = {}
        self.obj_hub: dict[SerializedCoreStateID, Any] = {}
        self._obj_id_to_state_id: dict[tuple[type, int], SerializedCoreStateID] = {}
        
    def pack(self, obj: Any):
        if isinstance(obj, dict):
            new_obj = {}
            for k, v in obj.items():
                if isinstance(v, SerializableCoreObject):
                    new_obj[k] = self.add_obj(v)
                else:
                    new_obj[k] = self.pack(v)
        elif isinstance(obj, list):
            new_obj = []
            for i, v in enumerate(obj):
                if isinstance(v, SerializableCoreObject):
                    new_obj.append(self.add_obj(v))
                else:
                    new_obj.append(self.pack(v))
        elif isinstance(obj, tuple):
            new_obj = []
            for v in obj:
                if isinstance(v, SerializableCoreObject):
                    new_obj.append(self.add_obj(v))
                else:
                    new_obj.append(self.pack(v))
            obj = tuple(new_obj)
        elif isinstance(obj, set):
            new_state_data = set()
            for v in obj:
                if isinstance(v, SerializableCoreObject):
                    new_state_data.add(self.add_obj(v))
                else:
                    new_state_data.add(self.pack(v))
            obj = new_state_data
        elif isinstance(obj, SerializableCoreObject):
            new_obj = self.add_obj(obj)
        else:
            new_obj = obj
        
        return new_obj
    
    def unpack(self, core: 'Core', obj: Any):
        if isinstance(obj, dict):
            new_obj = {}
            for k, v in obj.items():
                new_obj[k] = self.get_obj(core, v) if isinstance(v, SerializedCoreStateID) else self.unpack(core, v)
        elif isinstance(obj, list):
            new_obj = []
            for v in obj:
                new_obj.append(self.get_obj(core, v) if isinstance(v, SerializedCoreStateID) else self.unpack(core, v))
        elif isinstance(obj, tuple):
            new_obj = []
            for v in obj:
                new_obj.append(self.get_obj(core, v) if isinstance(v, SerializedCoreStateID) else self.unpack(core, v))
            new_obj = tuple(new_obj)
        elif isinstance(obj, set):
            new_obj = set()
            for v in obj:
                new_obj.add(self.get_obj(core, v) if isinstance(v, SerializedCoreStateID) else self.unpack(core, v))
        elif isinstance(obj, SerializedCoreStateID):
            new_obj = self.get_obj(core, obj)
        else:
            new_obj = obj
        
        return new_obj
    
    def add_obj(self, obj: Any):
        obj_id = (type(obj), id(obj))
        
        if obj_id in self._obj_id_to_state_id:
            return self._obj_id_to_state_id[obj_id]
        
        state_id = SerializedCoreStateID(f"{type(obj).__name__}.{len(self.state_hub)}")
        self._obj_id_to_state_id[obj_id] = state_id
        
        if isinstance(obj, SerializableCoreObject):
            state = SerializedCoreObjectState(type(obj), self.pack(obj.get_state()))
        else:
            state = SerializedCoreObjectState(type(obj), self.pack(obj))
        
        self.state_hub[state_id] = state
        
        return state_id
    
    def get_obj(self, core: 'Core', state_id: SerializedCoreStateID):
        if state_id in self.obj_hub:
            return self.obj_hub[state_id]
        
        state = self.state_hub.get(state_id, None)
        if state is None:
            raise Exception(f"State with ID {state_id} not found in state hub")
        
        if isinstance(state, SerializedCoreObjectState):
            state.state_data = self.unpack(core, state.state_data)
            obj = state.restore(core)
        elif isinstance(state, SerializedCoreStateID):
            obj = self.get_obj(core, state)
        else:
            obj = self.unpack(core, state)
            
        self.obj_hub[state_id] = obj
        
        return obj

_global_serializer = None
    
def new_global_serializer() -> Serializer:
    global _global_serializer
    _global_serializer = Serializer()
    return _global_serializer

def get_global_serializer() -> Serializer:
    global _global_serializer
    return _global_serializer

def check_global_serializer() -> bool:
    global _global_serializer
    return _global_serializer is not None

def clear_global_serializer():
    global _global_serializer
    _global_serializer = None