import functools
import math
import numpy as np
import time
from typing import Callable

from neuromta.framework import *


__all__ = [
    "BaseAccelerator",
    "HostJob",
    "HostJobHandle",
    "check_jit_job_prototype",
    "jit_host_job_prototype",
]


def check_jit_job_prototype(_func: Callable) -> bool:
    return hasattr(_func, "_is_jit_job_prototype") and _func._is_jit_job_prototype

def jit_host_job_prototype(_func: Callable):
    @functools.wraps(_func)
    def __jit_host_job_prototype_wrapper(device: 'BaseAccelerator', core_mesh: np.ndarray, *args, **kwargs):
        job = _func(device, core_mesh, *args, **kwargs)
        if not isinstance(job, HostJob):
            raise TypeError(f"Expected HostJob, got {type(job)}")
        job.device = device
        job.core_mesh = core_mesh
        return job
    __jit_host_job_prototype_wrapper._is_jit_job_prototype = True
    return __jit_host_job_prototype_wrapper

def _HOST_JOB_ANNOTATE_START(core: Core, job: 'HostJob'):
    job.annotate_actual_start(core.timestamp)

def _HOST_JOB_ANNOTATE_complete(core: Core, job: 'HostJob'):
    job.annotate_actual_complete(core.timestamp)
    if job.is_completed:
        job.commit()

@jit_prototype
def _HOST_JOB_INITIALIZE(core: Core, job: 'HostJob'):
    core.custom_command_with_ambiguous_func(_HOST_JOB_ANNOTATE_START, core, job)

@jit_prototype
def _HOST_JOB_FINALIZE(core: Core, job: 'HostJob'):
    core.var_atomic_increase(job.sync_var, 1)
    core.custom_command_with_ambiguous_func(_HOST_JOB_ANNOTATE_complete, core, job)


class HostJob:
    def __init__(
        self,
        programs: dict[str, list[Program | KernelPrototype]]=None,
    ):
        self.programs = programs if programs is not None else {}
        self._sync_var = None
        self._sync_var_totals = 0   # auto-initialized when device and core_mesh are set

        self._device: 'BaseAccelerator' = None
        self._core_mesh: np.ndarray = None

        self._schedule_release: int = 0
        self._schedule_deadline: int = 0
        self._actual_start: int | None = None
        self._actual_complete: int = 0
        self._commit_hooks: list[Callable] = []
        self._is_committed = False
        
        self._child_job_handles: list[HostJobHandle] = []
        
    def add_child_job_handles(self, *child_job_handles: 'HostJobHandle'):
        for child_job_handle in child_job_handles:
            if not isinstance(child_job_handle, HostJobHandle):
                raise TypeError(f"Expected HostJobHandle, got {type(child_job_handle)}")
            self._child_job_handles.append(child_job_handle)
        return self

    def annotate_schedule(self, release_timestamp: int, deadline_timestamp: int=-1):
        self._schedule_release = release_timestamp
        self._schedule_deadline = deadline_timestamp
        return self

    def annotate_actual_start(self, start_timestamp: int):
        self._actual_start = start_timestamp if self._actual_start is None else min(self._actual_start, start_timestamp)
        return self

    def annotate_actual_complete(self, complete_timestamp: int):
        self._actual_complete = max(complete_timestamp, self._actual_start if self._actual_start is not None else complete_timestamp)
        return self

    def add_program(self, slot_id: str, program: Program | KernelPrototype):
        self.programs.setdefault(slot_id, []).append(program)

    def add_commit_hook(self, hook: Callable):
        self._commit_hooks.append(hook)
        return self

    def merge(self, other: 'HostJob'):
        for slot_id, programs in other.programs.items():
            self.programs.setdefault(slot_id, []).extend(programs)
        self._commit_hooks.extend(other._commit_hooks)
        return self

    def commit(self):
        if self._is_committed:
            return self
        
        for hook in self._commit_hooks:
            hook(self)
        self._is_committed = True
        
        for child_job_handle in self._child_job_handles:
            child_job = child_job_handle.spawn(self._device)
            if child_job is None:
                continue
            child_job.dispatch()
        
        return self

    def dispatch(self):
        self._sync_var = VariableHandle.tmp(initial_value=0)

        for core_id in self.core_mesh.flatten().tolist():
            core = self.device.initialized_cores[core_id]
            for slot_id in self.programs.keys():
                _HOST_JOB_INITIALIZE(core, self).dispatch(slot_id=slot_id)

        for slot_id, programs in self.programs.items():
            for program in programs:
                program.dispatch(slot_id=slot_id)

        for core_id in self.core_mesh.flatten().tolist():
            core = self.device.initialized_cores[core_id]
            for slot_id in self.programs.keys():
                _HOST_JOB_FINALIZE(core, self).dispatch(slot_id=slot_id)

    @property
    def is_valid(self) -> bool:
        return self._device is not None and self._core_mesh is not None

    @property
    def device(self) -> 'BaseAccelerator':
        return self._device

    @device.setter
    def device(self, device: 'BaseAccelerator'):
        self._device = device
        if self.is_valid:
            self._sync_var = VariableHandle.tmp(initial_value=0)
            self._sync_var_totals = len(self._core_mesh.flatten().tolist()) * len(self.programs.keys())

    @property
    def core_mesh(self) -> np.ndarray:
        return self._core_mesh

    @core_mesh.setter
    def core_mesh(self, core_mesh: np.ndarray):
        self._core_mesh = core_mesh
        if self.is_valid:
            self._sync_var = VariableHandle.tmp(initial_value=0)
            self._sync_var_totals = len(self._core_mesh.flatten().tolist()) * len(self.programs.keys())

    @property
    def is_completed(self) -> bool:
        if not self.is_valid:
            raise RuntimeError("Device and core_mesh must be set before checking complete.")
        return self._sync_var.value >= self._sync_var_totals

    @property
    def sync_var(self) -> VariableHandle:
        return self._sync_var

    @property
    def actual_start(self) -> int | None:
        return self._actual_start

    @property
    def is_committed(self) -> bool:
        return self._is_committed
    
    def __repr__(self):
        return f"HostJob(schedule=({self._schedule_release}, {self._schedule_deadline}), actual=({self._actual_start}, {self._actual_complete}), is_committed={self._is_committed})"

class HostJobHandle:
    def __init__(
        self,
        domain_id: str,
        job_primitives: list[tuple[Callable, tuple, dict]],
        release_timestamp: int=0,
        deadline_timestamp: int=-1,
    ):
        self.domain_id = domain_id
        self.release_timestamp = release_timestamp
        self.deadline_timestamp = deadline_timestamp
        self.job_primitives = job_primitives
        
        self._parent_job_handles: list[HostJobHandle] = []
        self._child_job_handles: list[HostJobHandle] = []
        
        self._is_spawned = False
        self._spawned_job: HostJob = None
        
    def add_dependency(self, parent: 'HostJobHandle'):
        if parent is self:
            raise ValueError("A job handle cannot depend on itself.")
        if parent in self._parent_job_handles:
            return self  # Dependency already exists
        self._parent_job_handles.append(parent)
        parent._child_job_handles.append(self)
        return self

    def spawn(self, device: 'BaseAccelerator') -> HostJob:
        if self._is_spawned:
            raise RuntimeError("This HostJobHandle has already been spawned.")
        if self.domain_id not in device.host_domains.keys():
            raise ValueError(f"Domain ID {self.domain_id} not found in device host domains.")
        if not all(parent.is_spawned for parent in self._parent_job_handles):
            raise RuntimeError("Cannot spawn this job handle because not all parent job handles have been spawned.")
        if not all(parent.spawned_job.is_completed for parent in self._parent_job_handles):
            return None

        core_mesh = device.host_domains[self.domain_id]
        job = HostJob()

        for job_method, job_args, job_kwargs in self.job_primitives:
            if not check_jit_job_prototype(job_method):
                raise TypeError(f"Job method {job_method.__name__} is not a valid JIT host job prototype.")
            job_instance = job_method(device, core_mesh, *job_args, **job_kwargs)
            if not isinstance(job_instance, HostJob):
                raise TypeError(f"Expected HostJob, got {type(job_instance)}")
            job.merge(job_instance)

        if not isinstance(job, HostJob):
            raise TypeError(f"Expected HostJob, got {type(job)}")

        job.device = device
        job.core_mesh = core_mesh
        job.annotate_schedule(self.release_timestamp, self.deadline_timestamp)
        job.add_child_job_handles(*self._child_job_handles)
        
        self._is_spawned = True
        self._spawned_job = job
        
        return job

    def add(self, job_method: Callable, *job_args, **job_kwargs):
        if not check_jit_job_prototype(job_method):
            raise TypeError(f"Job method {job_method.__name__} is not a valid JIT host job prototype.")
        self.job_primitives.append((job_method, job_args, job_kwargs))
        return self

    def reschedule(self, release_timestamp: int, deadline_timestamp: int=-1):
        self.release_timestamp = release_timestamp
        self.deadline_timestamp = deadline_timestamp
        return self
    
    @property
    def is_spawned(self) -> bool:
        return self._is_spawned
    
    @property
    def spawned_job(self) -> HostJob:
        return self._spawned_job
    
    @property
    def has_parent_jobs(self) -> bool:
        return len(self._parent_job_handles) > 0
    
    @property
    def has_child_jobs(self) -> bool:
        return len(self._child_job_handles) > 0
    
    def __eq__(self, other):
        if not isinstance(other, HostJobHandle):
            return NotImplemented
        return id(self) == id(other)

    def __repr__(self):
        return f"HostJobHandle(domain_id={self.domain_id}, schedule=({self.release_timestamp}, {self.deadline_timestamp}))"


class BaseAccelerator(Device):
    def __init__(self):
        super().__init__()

        self._host_domains: dict[str, np.ndarray] = {}
        self._host_schedules: list[HostJobHandle] = []

    def host_reset_domain(self, domain_id: str, core_mesh: np.ndarray):
        self._host_domains[domain_id] = core_mesh
        return self

    def host_new_job_handle(self, domain_id: str, job_method: Callable, *job_args, **job_kwargs) -> HostJobHandle:
        if not check_jit_job_prototype(job_method):
            raise TypeError(f"Job method {job_method.__name__} is not a valid JIT host job prototype.")

        job_handle = HostJobHandle(
            domain_id=domain_id,
            job_primitives=[(job_method, job_args, job_kwargs)],
        )
        self._host_schedules.append(job_handle)
        return job_handle

    def host_materialize_job(
        self,
        core_mesh: np.ndarray,
        job_method: Callable,
        *job_args,
        release_timestamp: int=0,
        deadline_timestamp: int=-1,
        **job_kwargs,
    ) -> HostJob:
        if not self.is_initialized:
            raise RuntimeError("The accelerator must be initialized before materializing a HostJob.")
        if not check_jit_job_prototype(job_method):
            raise TypeError(f"Job method {job_method.__name__} is not a valid JIT host job prototype.")

        resolved_core_mesh = np.asarray(core_mesh, dtype=int)
        if resolved_core_mesh.ndim != 2 or resolved_core_mesh.size == 0:
            raise ValueError(f"Invalid core mesh shape: {resolved_core_mesh.shape}")

        core_ids = resolved_core_mesh.flatten().tolist()
        if any(core_id < 0 for core_id in core_ids):
            raise ValueError("A HostJob core mesh cannot contain an invalid core ID.")
        if len(set(core_ids)) != len(core_ids):
            raise ValueError("A HostJob core mesh cannot contain duplicate core IDs.")

        missing_core_ids = [core_id for core_id in core_ids if core_id not in self.initialized_cores]
        if missing_core_ids:
            raise ValueError(f"Unknown HostJob core IDs: {missing_core_ids}")

        job = job_method(self, resolved_core_mesh.copy(), *job_args, **job_kwargs)
        job.annotate_schedule(release_timestamp, deadline_timestamp)
        return job

    def host_run_next_event(self, max_timestamp: int=None) -> int:
        if not self.is_initialized:
            raise RuntimeError("The accelerator must be initialized before running Host events.")
        if max_timestamp is not None:
            if max_timestamp < self.timestamp:
                raise ValueError(f"Cannot run backwards from timestamp {self.timestamp} to {max_timestamp}.")
            if max_timestamp == self.timestamp:
                return self.timestamp
        if max_timestamp is None:
            self.run_single_step(event_driven_mode=True)
            return self.timestamp
        self._host_run_single_event(max_timestamp)
        return self.timestamp

    def host_run_until(self, stop_condition: Callable[[], bool] | None=None, max_timestamp: int=None) -> int:
        if not self.is_initialized:
            raise RuntimeError("The accelerator must be initialized before running Host events.")
        if stop_condition is not None and not callable(stop_condition):
            raise TypeError("stop_condition must be callable.")
        if max_timestamp is not None and max_timestamp < self.timestamp:
            raise ValueError(f"Cannot run backwards from timestamp {self.timestamp} to {max_timestamp}.")
        while stop_condition is None or not stop_condition():
            if max_timestamp is not None and self.timestamp >= max_timestamp:
                break
            if self.is_idle and max_timestamp is None:
                break
            self.host_run_next_event(max_timestamp=max_timestamp)
        return self.timestamp

    def _host_run_single_event(self, max_timestamp: int):
        self._profile_add_count("step_count")
        rpc_update_core_ids = set(self._rpc_pending_core_ids)
        active_update_core_ids = set(self._active_core_ids)
        profile_start = time.perf_counter() if self._sim_profile_enabled else None
        for core_id in rpc_update_core_ids:
            core = self.initialized_cores[core_id]
            self._sync_core_timestamp(core)
            core.rpc_update_routine()
        if profile_start is not None:
            self._profile_add_time("rpc_update_time", profile_start)
            self._profile_add_count("rpc_pending_core_update_count", len(rpc_update_core_ids))

        cycle_step = max_timestamp - self.timestamp
        remaining_cycles = None
        has_pending_work = False
        for core in self.initialized_cores.values():
            if core.is_idle:
                continue
            has_pending_work = True
            remaining = core.get_remaining_cycles()
            if remaining is not None:
                remaining_cycles = remaining if remaining_cycles is None else min(remaining_cycles, remaining)
        if has_pending_work:
            cycle_step = min(cycle_step, max(1, math.ceil(remaining_cycles)) if remaining_cycles is not None else 1)
        cycle_step = max(1, int(cycle_step))

        profile_start = time.perf_counter() if self._sim_profile_enabled else None
        self._sync_core_timestamp(self.companion_core)
        self.companion_core.update_cycle_time_companion_modules(cycle_time=cycle_step)
        if profile_start is not None:
            self._profile_add_time("companion_update_time", profile_start)

        update_core_ids = active_update_core_ids | set(self._active_core_ids)
        profile_start = time.perf_counter() if self._sim_profile_enabled else None
        for core_id in update_core_ids:
            core = self.initialized_cores[core_id]
            self._sync_core_timestamp(core)
            if not core.is_idle:
                core.update_cycle_time(cycle_time=cycle_step)
        if profile_start is not None:
            self._profile_add_time("core_update_time", profile_start)
            self._profile_add_count("active_core_update_count", len(update_core_ids))

        self._timestamp += cycle_step
        profile_start = time.perf_counter() if self._sim_profile_enabled else None
        for core_id in rpc_update_core_ids | update_core_ids | {self.companion_core.core_id}:
            self._refresh_core_scheduler_state(core_id)
        if profile_start is not None:
            self._profile_add_time("state_refresh_time", profile_start)
            self._profile_add_count("blocked_core_skip_count", len(self._blocked_core_ids))
            self._profile_add_count("idle_core_skip_count", len(self._idle_core_ids))

    def host_run_schedule(self, event_driven_mode: bool=True, verbose: bool=False) -> list[HostJob]:
        # STEP 1: Sort job handles by release timestamp
        self._host_schedules.sort(key=lambda jh: jh.release_timestamp)
        host_schedule_cursor = 0

        # STEP 2: Dispatch jobs to the accelerator based on their release timestamps
        self.reset_simulation()
        
        while host_schedule_cursor < len(self._host_schedules):
            job_handle = self._host_schedules[host_schedule_cursor]

            if job_handle.release_timestamp <= self.timestamp:
                if verbose:
                    print(f"dispatching job handle: {job_handle}")
                if not job_handle.has_parent_jobs:
                    job = job_handle.spawn(self)
                    if job is None:
                        raise RuntimeError("Failed to spawn job handle.")
                    job.dispatch()
                host_schedule_cursor += 1
            else:
                next_release_time = job_handle.release_timestamp
                self.run_kernels(
                    max_timestamp=next_release_time,
                    event_driven_mode=event_driven_mode
                )
                
        # STEP 3: Run the simulation until all jobs are completed
        if not self.is_idle:
            self.run_kernels(
                event_driven_mode=event_driven_mode
            )
        
        # STEP 4: Return the list of completed jobs   
        jobs = [jh.spawned_job for jh in self._host_schedules if jh.is_spawned]
        self._host_schedules.clear()
        
        return jobs

    @property
    def host_domains(self) -> dict[str, np.ndarray]:
        return self._host_domains

    @property
    def host_schedules(self) -> dict[str, list[HostJobHandle]]:
        return self._host_schedules
