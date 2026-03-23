import os
import json
import time
import queue
import threading
import enum
import pandas as pd
import numpy as np


__all__ = ["ProfilerBase", "GroupedProfilerBase", "ProfilerFileSaver", "ProfilerFileSaverHub", "ProfilerFileLoader"]


try:
    from neuromta_monitor.profiler import ProfilerBase, GroupedProfilerBase, ProfilerFileSaver, ProfilerFileSaverHub, ProfilerFileLoader
except ImportError as e:
    class ProfilerBase:
        class Type(enum.Enum):
            QUANTITY     = 0     # simple summation of quantities
            AVG_QUANTITY = 1
            UTILIZATION  = 2     # overlapped time intervals
            BANDWIDTH    = 3     # summation of (total quantity / overlapped interval)
        
        class Entry:
            def __init__(self, issue_time: int, commit_time: int, quantity: int=0):
                self.issue_time = issue_time
                self.commit_time = commit_time
                self.quantity = quantity
                
            def can_merge_with(self, other: 'ProfilerBase.Entry') -> bool:
                if self.issue_time <= other.issue_time <= self.commit_time:
                    return True
                if self.issue_time <= other.commit_time <= self.commit_time:
                    return True
                if other.issue_time <= self.issue_time <= other.commit_time:
                    return True
                if other.issue_time <= self.commit_time <= other.commit_time:
                    return True
                return False
            
            def merge_with(self, other: 'ProfilerBase.Entry') -> 'ProfilerBase.Entry':
                new_issue_time = min(self.issue_time, other.issue_time)
                new_commit_time = max(self.commit_time, other.commit_time)
                new_quantity = self.quantity + other.quantity
                return ProfilerBase.Entry(new_issue_time, new_commit_time, new_quantity)
                
        
        def __init__(self, metric_name: str, metric_unit: str, n_max_entries: int, profiler_type: Type):
            self.metric_name = metric_name
            self.metric_unit = metric_unit
            self.n_max_entries = n_max_entries
            self.profiler_type = profiler_type
            
            self.entries: list[ProfilerBase.Entry] = []
            self._is_dirty = False  # whether there are new entries since last get_profile()
            
            self._file_saver = None
            
        def add_entry(self, issue_time: int, commit_time: int, quantity: int=0):
            if issue_time == commit_time:
                commit_time += 1
                
            self.entries = sorted(self.entries, key=lambda e: (e.commit_time, e.issue_time))
            
            entry = ProfilerBase.Entry(issue_time, commit_time, quantity)
            
            if self._file_saver is not None:
                self._file_saver.add_entry(entry)
            
            is_merged = False
            for i in range(len(self.entries)-1, -1, -1):
                if self.entries[i].can_merge_with(entry):
                    entry = self.entries[i].merge_with(entry)
                    self.entries.pop(i)
                    is_merged = True
                elif is_merged:
                    break
            
            if len(self.entries) >= self.n_max_entries:
                self.entries.pop(0)
            self.entries.append(entry)
                
            self._is_dirty = True
        
        def commit(self):
            self._is_dirty = False

        def get_profile(self) -> float:
            if self.profiler_type == ProfilerBase.Type.QUANTITY:
                return self._get_quantity_profile()
            elif self.profiler_type == ProfilerBase.Type.AVG_QUANTITY:
                return self._get_average_profile()
            elif self.profiler_type == ProfilerBase.Type.UTILIZATION:
                return self._get_utilization_profile()
            elif self.profiler_type == ProfilerBase.Type.BANDWIDTH:
                return self._get_bandwidth_profile()
            else:
                raise ValueError(f"Unsupported profiler type: {self.profiler_type}")
        
        def _get_quantity_profile(self) -> float:
            return sum(entry.quantity for entry in self.entries)
        
        def _get_average_profile(self) -> float:
            if not self.entries:
                return 0.0
            
            total_commit_time = self.entries[0].commit_time
            total_issue_time = self.entries[0].issue_time
            total_quantity = 0
            
            for entry in self.entries:
                total_commit_time = max(entry.commit_time, total_commit_time)
                total_issue_time = min(entry.issue_time, total_issue_time)
                total_quantity += entry.quantity
                
            total_interval = total_commit_time - total_issue_time
            
            return 0.0 if total_interval == 0 else (total_quantity / total_interval)
        
        def _get_utilization_profile(self) -> float:
            if not self.entries:
                return 0.0
            
            # Sort entries by commit time
            sorted_entries = sorted(self.entries, key=lambda e: (e.commit_time, -e.issue_time), reverse=True)
            
            # Check total overlapped intervals
            current_issue_time = sorted_entries[0].issue_time
            current_commit_time = sorted_entries[0].commit_time
            total_issue_time = current_issue_time
            total_commit_time = current_commit_time
            total_overlapped_interval = 0
            
            for entry in sorted_entries:
                if entry.commit_time < current_issue_time:
                    total_overlapped_interval += current_commit_time - current_issue_time
                    current_issue_time = entry.issue_time
                    current_commit_time = entry.commit_time
                else:
                    current_issue_time = min(current_issue_time, entry.issue_time)
                    current_commit_time = max(current_commit_time, entry.commit_time)
                    
                total_issue_time = min(total_issue_time, entry.issue_time)
                total_commit_time = max(total_commit_time, entry.commit_time)
                
            total_overlapped_interval += (current_commit_time - current_issue_time)
                
            total_interval = total_commit_time
            if total_interval == 0:
                return 0.0
            
            return total_overlapped_interval / total_interval * 100
        
        def _get_bandwidth_profile(self) -> float:
            if not self.entries:
                return 0.0
            
            # Sort entries by commit time
            sorted_entries = sorted(self.entries, key=lambda e: (e.commit_time, -e.issue_time), reverse=True)
            
            # Check total overlapped intervals
            current_issue_time = sorted_entries[0].issue_time
            current_commit_time = sorted_entries[0].commit_time
            current_quantity = 0
            
            for entry in sorted_entries:
                if entry.commit_time <= current_issue_time:
                    break
                else:
                    current_issue_time = min(current_issue_time, entry.issue_time)
                    current_commit_time = max(current_commit_time, entry.commit_time)
                    current_quantity += entry.quantity
            
            total_interval = current_commit_time - current_issue_time
            if total_interval == 0:
                return 0.0
            
            return current_quantity / total_interval
        
        def register_file_saver(self, file_saver: 'ProfilerFileSaver'):
            self._file_saver = file_saver
        
        @property
        def metric_id(self) -> str:
            return f"{self.metric_name} ({self.metric_unit})"
        
        @property
        def is_dirty(self) -> bool:
            return self._is_dirty
        
        
    class GroupedProfilerBase:
        def __init__(self, n_agents: int, metric_name: str, metric_unit: str, n_max_entries: int, profiler_type: ProfilerBase.Type):
            self.n_agents = n_agents
            self.metric_name = metric_name
            self.metric_unit = metric_unit
            self.profiler_type = profiler_type
            self.profilers = [ProfilerBase(metric_name, metric_unit, n_max_entries, profiler_type) for _ in range(n_agents)]
            
        def add_entry(self, agent_id: int, issue_time: int, commit_time: int, quantity: int=0):
            if agent_id < 0 or agent_id >= self.n_agents:
                raise ValueError(f"Invalid agent_id: {agent_id}")
            self.profilers[agent_id].add_entry(issue_time, commit_time, quantity)
            
        def commit(self):
            for profiler in self.profilers:
                profiler.commit()
            
        def get_profile(self) -> float:
            if self.n_agents == 0:
                return 0.0
            
            if self.profiler_type == ProfilerBase.Type.QUANTITY:
                return sum(profiler.get_profile() for profiler in self.profilers)
            if self.profiler_type == ProfilerBase.Type.AVG_QUANTITY:
                return sum(profiler.get_profile() for profiler in self.profilers) / self.n_agents 
            elif self.profiler_type == ProfilerBase.Type.UTILIZATION:
                return sum(profiler.get_profile() for profiler in self.profilers) / self.n_agents
            elif self.profiler_type == ProfilerBase.Type.BANDWIDTH:
                return sum(profiler.get_profile() for profiler in self.profilers) / self.n_agents
            else:
                raise ValueError(f"Unsupported profiler type: {self.profiler_type}") 
            
        @property
        def metric_id(self) -> str:
            return f"Avg {self.metric_name} ({self.metric_unit})"
        
        @property
        def is_dirty(self) -> bool:
            return any(profiler.is_dirty for profiler in self.profilers)


    class ProfilerFileSaver:
        def __init__(self, profiler: ProfilerBase, output_dir: str, chunk_size: 10000):
            self.profiler = profiler
            self.output_dir = output_dir
            self.chunk_size = chunk_size
            
            self.queue = queue.Queue()
            self.buffer = []
            self.file_count = 0
            self.is_running = True
            
            self.metadata = {
                "schema": ["issue_time", "commit_time", "quantity"],
                "chunk_size": self.chunk_size,
                "profiler_info": {
                    "metric_name": profiler.metric_name,
                    "metric_unit": profiler.metric_unit,
                    "type": profiler.profiler_type.name,
                },
                "files": []
            }
            
            os.makedirs(self.output_dir, exist_ok=True)
            self._update_metadata_file()
            
            self.worker = threading.Thread(target=self._process_queue)
            self.worker.daemon = True
            self.worker.start()
            
            profiler.register_file_saver(self)

        def add_entry(self, entry: ProfilerBase.Entry):
            self.queue.put((entry.issue_time, entry.commit_time, entry.quantity))

        def _process_queue(self):
            while self.is_running or not self.queue.empty():
                try:
                    entry = self.queue.get(timeout=0.1)
                    self.buffer.append(entry)
                    
                    if len(self.buffer) >= self.chunk_size:
                        self._flush()
                        
                except queue.Empty:
                    continue

        def _flush(self):
            if not self.buffer:
                return

            df = pd.DataFrame(self.buffer, columns=self.metadata["schema"])
            filename = f"profile_part_{self.file_count:04d}.parquet"
            filepath = os.path.join(self.output_dir, filename)
            
            df.to_parquet(filepath, engine='pyarrow', index=False)

            self.metadata["files"].append({
                "filename": filename,
                "rows": len(self.buffer)
            })
            self._update_metadata_file()

            self.file_count += 1
            self.buffer.clear()

        def _update_metadata_file(self):
            meta_path = os.path.join(self.output_dir, "metadata.json")
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=4)

        def close(self):
            # # add remaining entries
            # for entry in self.profiler.entries:
            #     self.add_entry(entry)
            
            self.is_running = False
            self.worker.join()
            self._flush()
            
            return self
            
        def __del__(self):
            if self.is_running:
                self.close()


    class ProfilerFileSaverHub:
        def __init__(self, output_dir: str, chunk_size: int=10000):
            self.output_dir = output_dir
            self.chunk_size = chunk_size
            
            self._savers: list[ProfilerFileSaver] = []
            
            os.makedirs(self.output_dir, exist_ok=True)
            
            self.metadata = []
            
        def add_profilers(self, *profilers: ProfilerBase | GroupedProfilerBase):
            for profiler in profilers:
                _profiler_ids = []
                
                if isinstance(profiler, ProfilerBase):
                    profiler_id = f"profiler_{len(self._savers)}_{self._get_profiler_dirname(profiler)}"
                    self._savers.append(ProfilerFileSaver(profiler, os.path.join(self.output_dir, profiler_id), self.chunk_size))
                    _profiler_ids.append(profiler_id)
                elif isinstance(profiler, GroupedProfilerBase):
                    for i, p in enumerate(profiler.profilers):
                        profiler_id = f"profiler_{len(self._savers)}_{self._get_profiler_dirname(profiler)}_agent_{i}"
                        self._savers.append(ProfilerFileSaver(p, os.path.join(self.output_dir, profiler_id), self.chunk_size))
                        _profiler_ids.append(profiler_id)
                        
                self.metadata.append({
                    "profiler_type": type(profiler).__name__,
                    "profiler_base": "ProfilerBase" if isinstance(profiler, ProfilerBase) else "GroupedProfilerBase",
                    "profiler_ids": _profiler_ids,
                })
                
            self._update_metadata_file()
                
            return self
        
        def _update_metadata_file(self):
            os.makedirs(self.output_dir, exist_ok=True)
            meta_path = os.path.join(self.output_dir, "metadata.json")
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=4)
        
        def close(self):
            for saver in self._savers:
                saver.close()
            return self
                
        def __del__(self):
            self.close()
                    
        @staticmethod
        def _get_profiler_dirname(profiler: ProfilerBase):
            return profiler.metric_name.replace(" ", "_").lower()
        

    class ProfilerFileLoader:
        class Stat:
            def __init__(self, x: np.ndarray, y: np.ndarray):
                self.x = x
                self.y = y
        
        def __init__(self, target_dir: str):
            # Read files
            self.target_dir = target_dir
            self.meta_path = os.path.join(target_dir, "metadata.json")
            
            if not os.path.exists(self.meta_path):
                raise FileNotFoundError(f"Cannot find metadata file: {self.meta_path}")

            with open(self.meta_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            if not metadata["files"]:
                raise ValueError(f"No data files found in metadata: {self.meta_path}")

            self.file_paths = [os.path.join(target_dir, file_info["filename"]) for file_info in metadata["files"]]
            self.df = pd.read_parquet(self.file_paths, engine='pyarrow')
            
            # Parse profiler information from metadata
            if "profiler_info" not in metadata:
                raise ValueError(f"Profiler information not found in metadata: {self.meta_path}")
            if "metric_name" not in metadata["profiler_info"] or "metric_unit" not in metadata["profiler_info"] or "type" not in metadata["profiler_info"]:
                raise ValueError(f"Incomplete profiler information in metadata: {self.meta_path}")
            
            self._metric_name = metadata["profiler_info"]["metric_name"]
            self._metric_unit = metadata["profiler_info"]["metric_unit"]
            self._profiler_type = ProfilerBase.Type[metadata["profiler_info"]["type"]]
        
        def create_quantity_stat(self):
            total_time = self.df['commit_time'].max()
            x = np.arange(0, total_time + 1)
            y = np.zeros_like(x, dtype=np.float32)
            
            for _, row in self.df.iterrows():
                commit_time = row['commit_time']
                quantity = row['quantity']
                
                y[commit_time:] += quantity
            
            return ProfilerFileLoader.Stat(x, y)
            
        def create_average_stat(self):
            total_time = self.df['commit_time'].max()
            x = np.arange(0, total_time + 1)
            y = np.zeros_like(x, dtype=np.float32)

            for _, row in self.df.iterrows():
                commit_time = row['commit_time']
                quantity = row['quantity']
                
                y[commit_time:] += quantity
                
            y = y / x.clip(min=1)  # avoid division by zero
            
            return ProfilerFileLoader.Stat(x, y)
        
        def create_utilization_stat(self, window_ratio: float=0.01):
            total_time = self.df['commit_time'].max()
            x = np.arange(0, total_time + 1)
            flag = np.zeros_like(x, dtype=np.float32)

            for _, row in self.df.iterrows():
                issue_time = row['issue_time']
                commit_time = row['commit_time']
                
                flag[issue_time:commit_time] = 1
                
            # Smooth the utilization curve with a moving average filter
            window_size = max(1, int(total_time * window_ratio))
            y = np.convolve(flag, np.ones(window_size) / window_size, mode='same')
            
            return ProfilerFileLoader.Stat(x, y)
        
        def create_bandwidth_stat(self, window_ratio: float=0.01):
            total_time = self.df['commit_time'].max()
            x = np.arange(0, total_time + 1)
            flag = np.zeros_like(x, dtype=np.float32)

            for _, row in self.df.iterrows():
                issue_time = row['issue_time']
                commit_time = row['commit_time']
                quantity = row['quantity']
                
                flag[issue_time:commit_time] += quantity
                
            # Smooth the utilization curve with a moving average filter
            window_size = max(1, int(total_time * window_ratio))
            y = np.convolve(flag, np.ones(window_size) / window_size, mode='same')
            
            return ProfilerFileLoader.Stat(x, y)
        
        def create_stat(self, window_ratio: float=0.01) -> 'ProfilerFileLoader.Stat':
            if self._profiler_type == ProfilerBase.Type.QUANTITY:
                return self.create_quantity_stat()
            elif self._profiler_type == ProfilerBase.Type.AVG_QUANTITY:
                return self.create_average_stat()
            elif self._profiler_type == ProfilerBase.Type.UTILIZATION:
                return self.create_utilization_stat(window_ratio)
            elif self._profiler_type == ProfilerBase.Type.BANDWIDTH:
                return self.create_bandwidth_stat(window_ratio)
            else:
                raise ValueError(f"Unsupported profiler type: {self._profiler_type}")


    class ProfilerFileLoaderHub:
        def __init__(self, target_dir: str):
            self.target_dir = target_dir
            
            self.meta_path = os.path.join(target_dir, "metadata.json")
            
            if not os.path.exists(self.meta_path):
                raise FileNotFoundError(f"Cannot find metadata file: {self.meta_path}")
            
            with open(self.meta_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
                
            self.loaders: dict[str, ProfilerFileLoader] = {}
                
            for profiler_file_info in self.metadata:
                for profiler_id in profiler_file_info["profiler_ids"]:
                    loader = ProfilerFileLoader(os.path.join(target_dir, profiler_id))
                    self.loaders[profiler_id] = loader
                    
        def create_stat(self, idx, window_ratio: float=0.01) -> ProfilerFileLoader.Stat:
            if idx < 0 or idx >= len(self.metadata):
                raise ValueError(f"Invalid profiler index: {idx}")
            
            profiler_file_info = self.metadata[idx]
            stats = []
            
            for profiler_id in profiler_file_info["profiler_ids"]:
                loader = self.loaders[profiler_id]
                stat = loader.create_stat(window_ratio)
                stats.append(stat)
            
            # Average stats if there are multiple profilers (e.g., grouped profiler)
            if len(stats) == 1:
                return stats[0]
            else:
                x = stats[0].x
                y = np.mean([stat.y for stat in stats], axis=0)
                return ProfilerFileLoader.Stat(x, y)
            
        @property
        def n_profilers(self) -> int:
            return len(self.metadata)