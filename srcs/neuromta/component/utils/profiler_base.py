import enum


__all__ = ["ProfilerBase", "GroupedProfilerBase"]

try:
    from neuromta_monitor.profiler import GroupedProfilerBase, ProfilerBase
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
        
        def __init__(self, metric_name: str, metric_unit: str, n_max_entries: int, profiler_type: Type):
            self.metric_name = metric_name
            self.metric_unit = metric_unit
            self.n_max_entries = n_max_entries
            self.profiler_type = profiler_type
            
            self.entries: list[ProfilerBase.Entry] = []
            
        def add_entry(self, issue_time: int, commit_time: int, quantity: int=0):
            if issue_time == commit_time:
                commit_time += 1
            
            entry = ProfilerBase.Entry(issue_time, commit_time, quantity)
            if len(self.entries) >= self.n_max_entries:
                self.entries.pop(0)
            self.entries.append(entry)

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
            sorted_entries = sorted(self.entries, key=lambda e: e.commit_time, reverse=True)
            
            # Check total overlapped intervals
            current_issue_time = sorted_entries[0].issue_time
            current_commit_time = sorted_entries[0].commit_time
            total_issue_time = current_issue_time
            total_commit_time = current_commit_time
            total_overlapped_interval = 0
            
            for entry in sorted_entries:
                if entry.commit_time <= current_issue_time:
                    total_overlapped_interval += current_commit_time - current_issue_time
                    current_issue_time = entry.issue_time
                    current_commit_time = entry.commit_time
                else:
                    current_issue_time = min(current_issue_time, entry.issue_time)
                    current_commit_time = max(current_commit_time, entry.commit_time)
                    
                total_issue_time = min(total_issue_time, entry.issue_time)
                total_commit_time = max(total_commit_time, entry.commit_time)
                
            total_interval = total_commit_time - total_issue_time
            if total_interval == 0:
                return 0.0
            
            return total_overlapped_interval / total_interval
        
        def _get_bandwidth_profile(self) -> float:
            if not self.entries:
                return 0.0
            
            # Sort entries by commit time
            sorted_entries = sorted(self.entries, key=lambda e: e.commit_time, reverse=True)
            
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
        
        @property
        def metric_id(self) -> str:
            return f"{self.metric_name} ({self.metric_unit})"
        
        
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