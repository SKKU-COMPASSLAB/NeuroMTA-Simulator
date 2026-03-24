import os
import dill
import zstandard as zstd
import types
import io
import _thread

from neuromta.framework.core import *
from neuromta.framework.device import Device
from neuromta.framework.logger import logger


__all__ = ["DumpSaver"]

class DumpSaver:
    def __init__(self, device: Device):
        self._device = device
        
        if not self._device.is_initialized:
            raise Exception("Device is not initialized. Please call initialize() before creating a dump.")
        
    def dump(self) -> dict:
        if check_global_serializer():
            logger.info("A global serializer already exists. Reusing the existing serializer for dumping.")
            serializer = get_global_serializer()
        else:
            logger.info("No global serializer found. Creating a new global serializer for dumping.")
            serializer = new_global_serializer()

        core_dump: dict = {}
        
        for core_id, core in self._device.initialized_cores.items():
            core_state = core.all_dump_core_states()
            core_dump[core_id] = core_state
            
        clear_global_serializer()
        
        return {
            "core_dump": core_dump,
            "state_hub": serializer.state_hub
        }
    
    def load(self, dump_data: dict):
        core_dump = dump_data.get("core_dump", {})
        state_hub = dump_data.get("state_hub", {})
        
        if check_global_serializer():
            logger.info("A global serializer already exists. Reusing the existing serializer for loading.")
            serializer = get_global_serializer()
            serializer.state_hub.update(state_hub)
        else:
            logger.info("No global serializer found. Creating a new global serializer for loading.")
            serializer = new_global_serializer()
            serializer.state_hub.update(state_hub)
            
        for core_id, core in self._device.initialized_cores.items():
            state = core_dump.get(core_id, None)
            if state is not None:
                core.all_load_core_states(state)
            else:
                logger.warning(f"Core with ID {core_id} in device does not have a corresponding state in dump data. Skipping loading states for this core.")
                
    def dump_to_file(self, output_dir: str, file_name: str):
        if not file_name.endswith(".nmta"):
            file_name += ".nmta"
            
        try:
            os.makedirs(output_dir, exist_ok=True)
            file_path = os.path.join(output_dir, file_name)
            dump_data = self.dump()
            
            inspector = SerializationInspector()
            flag = inspector.run(dump_data, target_name="dump_data")
            
            if not flag:
                raise Exception("Dump data contains unserializable objects. Aborting dump to file.")
            
            packed_data = dill.dumps(dump_data)
            compressor = zstd.ZstdCompressor()
            compressed_data = compressor.compress(packed_data)
            with open(file_path, "wb") as f:
                f.write(compressed_data)
                
            logger.info(f"Successfully saved dump to file: {file_path}")
        except Exception as e:
            logger.error(f"Error occurred while dumping to file {file_path}: {e}")
            raise e
            
    def load_from_file(self, output_dir: str, file_name: str):
        if not file_name.endswith(".nmta"):
            file_name += ".nmta"
        file_path = os.path.join(output_dir, file_name)
        
        try:
            with open(file_path, "rb") as f:
                compressed_data = f.read()
            decompressor = zstd.ZstdDecompressor()
            packed_data = decompressor.decompress(compressed_data)
            dump_data = dill.loads(packed_data)
            self.load(dump_data)
            logger.info(f"Successfully loaded dump from file: {file_path}")
        except FileNotFoundError as e:
            logger.error(f"File not found: {file_path}")
            raise e
        except Exception as e:
            logger.error(f"Error occurred while loading dump from file {file_path}: {e}")
            raise e

class SerializationInspector:
    def __init__(self):
        self.visited = set()
        self.unsafe_type_names = {'PyCapsule', 'SwigPyObject'}
        self.found_issues = False

    def is_unsafe(self, obj):
        type_name = type(obj).__name__
        if type_name in self.unsafe_type_names or 'capsule' in type_name.lower():
            return True
        
        if isinstance(obj, io.IOBase):
            return True
            
        if type(obj) is _thread.LockType:
            return True
            
        return False

    def check_safety(self, obj, path="root"):
        obj_id = id(obj)
        if obj_id in self.visited:
            return
        self.visited.add(obj_id)

        if self.is_unsafe(obj):
            logger.warning(f"unsafe object detected: {path}")
            logger.warning(f"   -> type: {type(obj).__name__}, value: {obj}")
            self.found_issues = True
            return

        if isinstance(obj, dict):
            for k, v in obj.items():
                self.check_safety(k, f"{path}[key:{k}]")
                self.check_safety(v, f"{path}['{k}']")

        elif isinstance(obj, (list, tuple, set, frozenset)):
            for i, item in enumerate(obj):
                self.check_safety(item, f"{path}[{i}]")

        elif isinstance(obj, types.FunctionType):
            if obj.__closure__:
                for i, cell in enumerate(obj.__closure__):
                    self.check_safety(cell.cell_contents, f"{path}.__closure__[{i}]")

        elif hasattr(obj, '__dict__'):
            for k, v in vars(obj).items():
                self.check_safety(v, f"{path}.{k}")

        elif hasattr(obj, '__slots__'):
            for slot in obj.__slots__:
                if hasattr(obj, slot):
                    self.check_safety(getattr(obj, slot), f"{path}.{slot}")

    def run(self, target_obj, target_name="TargetObject"):
        logger.debug(f"testing serialization safety of '{target_name}' ...")
        self.visited.clear()
        self.found_issues = False
        
        self.check_safety(target_obj, path=target_name)
        
        if self.found_issues:
            return False
        else:
            return True

# if __name__ == "__main__":
#     import threading

#     class DummyPyCapsule:
#         pass
#     DummyPyCapsule.__name__ = 'PyCapsule'

#     dangerous_pointer = DummyPyCapsule()
#     lock = threading.Lock()

#     def simulator_decorator(func):
#         engine_capsule = dangerous_pointer 
        
#         def wrapper(*args, **kwargs):
#             return func(*args, **kwargs)
#         return wrapper

#     @simulator_decorator
#     def profile_target():
#         pass

#     complex_data = {
#         "metadata": {"version": 1.0, "status": "running"},
#         "history": [1, 2, 3],
#         "state": {
#             "lock": lock,
#             "func": profile_target
#         }
#     }

#     inspector = SerializationInspector()
#     inspector.run(complex_data, target_name="complex_data")