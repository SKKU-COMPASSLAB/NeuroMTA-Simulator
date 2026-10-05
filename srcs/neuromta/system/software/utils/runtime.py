import datetime
import enum
import math
import functools
from collections import Counter
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import Callable, Any

import numpy as np

from neuromta.framework import *
from neuromta.framework.logger import logger

from neuromta.component.core.ccg_tile import BCAST_MODE_PARALLEL, CCGTile

from neuromta.system.hardware.base_accelerator import HostJob, jit_host_job_prototype
from neuromta.system.hardware.mesh_accelerator import MeshAccelerator, MeshAcceleratorRuntimeContext

from neuromta.system.software.utils.compiler import (
    MeshDeviceCompiledWorkload,
    MeshDeviceCompiledAction,
    MeshDeviceActionType,
    MeshDeviceCompiledTensorStats,
    MeshDeviceCompiledKernelStats,
)
from neuromta.system.software.utils.descriptor import (
    MeshDeviceDescriptor,
    MeshKernelType,
    MeshMemoryDescriptor,
    MeshMemoryType,
    MeshTensorDescriptor,
    MeshTensorType,
    MeshMemoryBankDescriptor,
)
from neuromta.system.software.utils.scheduler import (
    MeshFRFCFSScheduler,
    MeshDeviceScheduler,
    MeshSchedulingDomain,
    MeshWorkloadSchedulingHint,
)
from neuromta.system.software.utils.runtime_utils import *
from neuromta.system.software.utils import runtime_utils as _runtime_utils


__all__ = [
    "MeshDeviceRuntimeWorkloadState",
    "MeshKernelState",
    "MeshDeviceRuntime",
    "MeshDeviceRuntimeKernelMaterialzer",
    "MeshKernel",
    "MeshSDPAKernel",
    "MeshLinearKernel",
    "MeshConv2dKernel",
    "MeshElementwiseKernel",
    "MeshReductionKernel",
    "MeshMemCopyKernel",
]


class MeshKernelState(enum.Enum):
    WAITING = "WAITING"
    READY = "READY"
    PLACED = "PLACED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"


@dataclass
class _MeshKernelPlacementSnapshot:
    core_mesh: np.ndarray
    memory_bank_ids: tuple[int, ...]


@dataclass
class _MeshRuntimeKernelInvocation:
    execution_id: str
    compiled_kernel: MeshDeviceCompiledKernelStats
    placement: _MeshKernelPlacementSnapshot
    dispatch_cycle: int
    start_cycle: int | None = None
    completion_cycle: int = 0
    state: MeshKernelState = MeshKernelState.RUNNING


class MeshDeviceRuntimeWorkloadState:
    SUSPENDED       = 0     # STATE: The workload is suspended and should be warmed up before execution
    ACTION_WAIT     = 2     # STATE: The workload is scheduled (resolved all the dependencies) and is waiting for the next action to be executed
    ACTION_RUNNING  = 3     # STATE: The workload is currently executing an action on the device (cursor is pointing to the current action)
    COMPLETED       = 4     # STATE: The workload has completed all its actions and is finished executing

    def __init__(
        self,
        workload_id: str,
        compiled_workload: MeshDeviceCompiledWorkload,
        arrival_cycle: int = 0,
        dependent_workload_ids: tuple[str, ...] = (),
        scheduling_hint: MeshWorkloadSchedulingHint = None,
        domain_id: str | None = None,
    ):
        self._workload_id = workload_id
        self._compiled_workload = compiled_workload

        self._arrival_cycle = arrival_cycle
        self._dependent_workload_ids = dependent_workload_ids
        self._scheduling_hint = scheduling_hint
        self._domain_id = domain_id

        self._state: int = MeshDeviceRuntimeWorkloadState.SUSPENDED
        self._cursor = 0
        self._kernel_log: list[_MeshRuntimeKernelInvocation] = []
        self._completion_cycle = 0

    def finalize_warmup(self):
        self._state = MeshDeviceRuntimeWorkloadState.ACTION_WAIT
        self._cursor = 0
        return self

    def issue_current_actions(self) -> list[MeshDeviceCompiledAction]:
        if self._state != MeshDeviceRuntimeWorkloadState.ACTION_WAIT:
            raise RuntimeError(f"Cannot issue actions for workload '{self._workload_id}' in state '{self._state}'.")

        actions: list[MeshDeviceCompiledAction] = []

        while self._cursor < len(self._compiled_workload.main_actions):
            action = self._compiled_workload.main_actions[self._cursor]
            actions.append(action)
            self._cursor += 1

            if action.action_type == MeshDeviceActionType.RUN_KERNEL:
                break  # Stop issuing actions after a RUN_KERNEL action

        self._state = MeshDeviceRuntimeWorkloadState.ACTION_RUNNING

        return actions

    def commit_current_actions(self):
        if self._state != MeshDeviceRuntimeWorkloadState.ACTION_RUNNING:
            raise RuntimeError(f"Cannot commit actions for workload '{self._workload_id}' in state '{self._state}'.")

        if self._cursor >= len(self._compiled_workload.main_actions):
            self._state = MeshDeviceRuntimeWorkloadState.COMPLETED
        else:
            self._state = MeshDeviceRuntimeWorkloadState.ACTION_WAIT

        return self

    def clear(self):
        self._state = MeshDeviceRuntimeWorkloadState.SUSPENDED
        self._cursor = 0
        return self

    @property
    def workload_id(self) -> str:                               return self._workload_id
    @property
    def compiled_workload(self) -> MeshDeviceCompiledWorkload:  return self._compiled_workload
    @property
    def arrival_cycle(self) -> int:                             return self._arrival_cycle
    @property
    def dependent_workload_ids(self) -> tuple[str, ...]:        return self._dependent_workload_ids
    @property
    def scheduling_hint(self) -> MeshWorkloadSchedulingHint:    return self._scheduling_hint
    @property
    def domain_id(self) -> str | None:                          return self._domain_id
    @property
    def state(self) -> str:                 return self._state
    @property
    def cursor(self) -> int:                return self._cursor
    @property
    def kernel_log(self) -> tuple[_MeshRuntimeKernelInvocation, ...]: return tuple(self._kernel_log)
    @property
    def start_cycle(self) -> int | None: return self._kernel_log[0].start_cycle if self._kernel_log else None
    @property
    def completion_cycle(self) -> int: return self._completion_cycle
    @property
    def is_suspended(self) -> bool:         return self._state == self.SUSPENDED
    @property
    def is_action_wait(self) -> bool:       return self._state == self.ACTION_WAIT
    @property
    def is_action_running(self) -> bool:    return self._state == self.ACTION_RUNNING
    @property
    def is_completed(self) -> bool:         return self._state == self.COMPLETED


class MeshDeviceRuntime(ABC):
    """
    Introduction
    ------------
    TBD
    """

    def __init__(
        self,
        device_desc: MeshDeviceDescriptor | MeshAccelerator,
        kernel_materializer: 'MeshDeviceRuntimeKernelMaterialzer' = None,
        scheduler: MeshDeviceScheduler = None,
        enable_debug_log: bool = False,
    ):
        if isinstance(device_desc, MeshAccelerator):
            self._device = device_desc
            self._device_desc = MeshDeviceDescriptor(device_desc)
            self._device_rt_context = MeshAcceleratorRuntimeContext(self._device)
        elif isinstance(device_desc, MeshDeviceDescriptor):
            self._device_desc = device_desc
            self._device = device_desc.device
            self._device_rt_context = MeshAcceleratorRuntimeContext(self._device)
        else:
            raise TypeError(f"Expected MeshDeviceDescriptor or MeshAccelerator, got {type(device_desc).__name__}")

        if kernel_materializer is None:
            self.kernel_materializer = MeshDeviceRuntimeKernelMaterialzer(scheduler=scheduler)
        elif isinstance(kernel_materializer, MeshDeviceRuntimeKernelMaterialzer):
            if scheduler is not None:
                raise ValueError("scheduler cannot be provided with kernel_materializer.")
            self.kernel_materializer = kernel_materializer
        else:
            raise TypeError(f"Expected MeshDeviceRuntimeKernelMaterialzer, got {type(kernel_materializer).__name__}.")

        self.kernel_materializer.register_context(self._device_rt_context)
        self._device_domain = MeshSchedulingDomain("device", self._device.get_ccg_tile_mesh(), tuple(self._device_rt_context.dma_tile_ids))

        self._enable_debug_log = enable_debug_log

        self._workload_states: dict[str, MeshDeviceRuntimeWorkloadState] = {}
        self._persistent_state_placements: dict[str, MeshDeviceCompiledTensorStats] = {}
        self._resident_weight_placements: dict[tuple, MeshDeviceCompiledTensorStats] = {}
        self._resident_weight_users: dict[tuple, set[MeshDeviceCompiledWorkload]] = {}
        self._resident_weight_bindings: dict[MeshDeviceCompiledWorkload, set[tuple]] = {}
        self.kernel_materializer._persistent_state_placements = self._persistent_state_placements
        self.kernel_materializer._resident_weight_placements = self._resident_weight_placements
        self.kernel_materializer._resident_weight_users = self._resident_weight_users
        self.kernel_materializer._resident_weight_bindings = self._resident_weight_bindings
        self._decision_log: list[dict] = []
        self._running_kernel_ids: set[str] = set()
        self._dispatched_workload_ids: set[str] = set()
        self._reset_policy_state()
        self._action_post_hooks: dict[MeshDeviceActionType, list[Callable[[MeshDeviceCompiledAction], None]]] = {
            MeshDeviceActionType.PLACE_TENSOR: [],
            MeshDeviceActionType.RUN_KERNEL: [],
        }

    ###########################################################################
    # Public Runtime Interface
    ###########################################################################

    def submit(
        self,
        compiled_workload: MeshDeviceCompiledWorkload,
        arrival_cycle: int = 0,
        workload_id: str = None,
        dependent_workload_ids: tuple[str, ...] = (),
        scheduling_hint: MeshWorkloadSchedulingHint = None,
        warmup: bool = False,
        domain_id: str | None = None,
    ) -> str:
        if workload_id is None:
            _cnt = 0
            while f"workload_{_cnt}" in self._workload_states:
                _cnt += 1
            workload_id = f"workload_{_cnt}"
        if workload_id in self._workload_states:
            raise ValueError(f"Workload ID '{workload_id}' is already registered.")

        self._workload_states[workload_id] = MeshDeviceRuntimeWorkloadState(
            workload_id=workload_id,
            compiled_workload=compiled_workload,
            arrival_cycle=arrival_cycle,
            dependent_workload_ids=dependent_workload_ids,
            scheduling_hint=scheduling_hint,
            domain_id=domain_id,
        )

        if warmup:
            self.warmup(workload_id)

        return workload_id

    def warmup(self, *workload_ids: str):
        for workload_id in workload_ids:
            workload_state = self._workload_states.get(workload_id)
            if workload_state is None:
                raise ValueError(f"Workload ID '{workload_id}' not found in runtime workload states.")
            domain = self._get_scheduling_domain(workload_state)
            for action in workload_state.compiled_workload.warmup_actions:
                if action.action_type != MeshDeviceActionType.PLACE_TENSOR:
                    raise RuntimeError("Warmup actions must contain only tensor placements.")
                if not self.kernel_materializer.place_tensor(workload_state, action.tensor_id, dma_ids=list(domain.dma_ids), ccg_ids=domain.ccg_tile_mesh.flatten().tolist()):
                    raise RuntimeError(f"Failed to place tensor '{action.tensor_id}' during warmup for workload '{workload_id}'.")
                self._get_action_post_hook()(self.kernel_materializer.Token.done(action, workload_state))
            workload_state.finalize_warmup()
        return self

    def reserve_state(self, states) -> dict[str, MeshDeviceCompiledTensorStats]:
        descriptors = (states,) if isinstance(states, MeshTensorDescriptor) else tuple(states)
        selected = {}
        for descriptor in descriptors:
            if not isinstance(descriptor, MeshTensorDescriptor) or not descriptor.is_persistent:
                raise TypeError("states must contain persistent MeshTensorDescriptor objects.")
            state_id = descriptor.storage_desc.persistent_state_id
            placement = self._persistent_state_placements.get(state_id)
            if placement is None:
                placement = MeshDeviceCompiledTensorStats(descriptor.storage_desc)
                if not self.kernel_materializer.place_tensor_stats(placement):
                    raise RuntimeError(f"Failed to reserve persistent state '{state_id}'.")
                self._persistent_state_placements[state_id] = placement
            selected[state_id] = placement
        return selected

    def deallocate_state(self, states=None) -> int:
        if states is None:
            state_ids = tuple(self._persistent_state_placements)
        else:
            items = (states,) if isinstance(states, (str, MeshTensorDescriptor)) else tuple(states)
            state_ids = tuple(item if isinstance(item, str) else item.storage_desc.persistent_state_id for item in items)
        active = [state.workload_id for state in self._workload_states.values() if not state.is_completed and any(stats.tensor_desc.storage_desc.persistent_state_id in state_ids for stats in state.compiled_workload.tensor_stats_map.values())]
        if active:
            raise RuntimeError(f"Cannot deallocate persistent state used by incomplete workloads: {active}")
        released = 0
        for state_id in state_ids:
            placement = self._persistent_state_placements.pop(state_id, None)
            if placement is not None:
                self.kernel_materializer.release_tensor_stats(placement)
                released += 1
        return released

    def deallocate_weights(self, compiled_workload: MeshDeviceCompiledWorkload=None) -> int:
        if compiled_workload is not None and not isinstance(compiled_workload, MeshDeviceCompiledWorkload):
            raise TypeError(f"Expected MeshDeviceCompiledWorkload, got {type(compiled_workload).__name__}.")
        targets = (compiled_workload,) if compiled_workload is not None else tuple(dict.fromkeys(state.compiled_workload for state in self._workload_states.values()))
        active = [state.workload_id for state in self._workload_states.values() if state.compiled_workload in targets and not state.is_completed]
        if active:
            raise RuntimeError(f"Cannot deallocate weights used by incomplete workloads: {active}")
        released = 0
        for workload in targets:
            for stats in workload.tensor_stats_map.values():
                if stats.tensor_desc.tensor_type == MeshTensorType.WEIGHT:
                    stats.unplace()
            for key in self._resident_weight_bindings.pop(workload, set()):
                users = self._resident_weight_users.get(key)
                if users is None:
                    continue
                users.discard(workload)
                if users:
                    continue
                self._resident_weight_users.pop(key)
                placement = self._resident_weight_placements.pop(key)
                self.kernel_materializer.release_tensor_stats(placement)
                released += 1
        return released

    @property
    def persistent_state_placements(self) -> dict[str, MeshDeviceCompiledTensorStats]:
        return dict(self._persistent_state_placements)

    @property
    def workloads(self) -> tuple[MeshDeviceRuntimeWorkloadState, ...]:
        return tuple(self._workload_states.values())

    def run(self) -> list[HostJob]:
        jobs: list[HostJob] = []
        completion_events: list[MeshDeviceRuntimeKernelMaterialzer.Token] = []

        while not all(workload_state.is_completed for workload_state in self._workload_states.values()):
            for workload_id, workload_state in self._workload_states.items():
                if workload_state.is_suspended and workload_state.arrival_cycle <= self._device.timestamp and all(self._workload_states[dep_id].is_completed for dep_id in workload_state.dependent_workload_ids):
                    try:
                        self.warmup(workload_id)
                    except RuntimeError as error:
                        raise RuntimeError(f"Failed to warmup workload '{workload_id}' at timestamp '{self._device.timestamp}': {error}") from error

            ready_states = tuple(workload_state for workload_state in self._workload_states.values() if workload_state.is_action_wait)
            for workload_state in self._select_dispatchable_workloads(ready_states):
                domain = self._get_scheduling_domain(workload_state)
                for action in workload_state.issue_current_actions():
                    self.kernel_materializer.submit_action(action, workload_state, domain)

            tensor_tokens, kernel_tokens = self.kernel_materializer.schedule_actions(tuple(sorted(self._running_kernel_ids)))

            for token in tensor_tokens:
                self._get_action_post_hook()(token)

            for token in kernel_tokens:
                if not token.is_run_sim:
                    continue
                kernel_stats = token.workload_state.compiled_workload.get_kernel_stats(token.action.kernel_id)
                memory_bank_ids = tuple(dict.fromkeys(bank.owner_id for tensor in kernel_stats.kernel_desc.input_tensors for bank in token.workload_state.compiled_workload.get_tensor_stats(tensor).tile_placement.values()))
                execution_id = f"{token.workload_state.workload_id}/{token.workload_state.cursor}/{token.action.kernel_id}"
                token.kernel_invocation = _MeshRuntimeKernelInvocation(execution_id, kernel_stats, _MeshKernelPlacementSnapshot(kernel_stats.ccg_tile_mesh.copy(), memory_bank_ids), self._device.timestamp)
                token.workload_state._kernel_log.append(token.kernel_invocation)
                token.host_job.add_commit_hook(lambda _, token=token: completion_events.append(token))
                token.host_job.dispatch()
                self._running_kernel_ids.add(execution_id)
                if token.workload_state.workload_id not in self._dispatched_workload_ids:
                    self._dispatched_workload_ids.add(token.workload_state.workload_id)
                    self._on_workload_dispatched(token.workload_state)
                jobs.append(token.host_job)

            if kernel_tokens:
                metadata = self.kernel_materializer.scheduler.decision_metadata
                self._decision_log.append({"cycle": self._device.timestamp, "benefit": metadata.get("benefit", 0.0), "kernels": tuple(token.kernel_invocation.execution_id for token in kernel_tokens), "core_meshes": tuple(token.kernel_invocation.placement.core_mesh.copy() for token in kernel_tokens), "memory_banks": tuple(token.kernel_invocation.placement.memory_bank_ids for token in kernel_tokens), "scheduler": self.kernel_materializer.scheduler.name})

            arrival_cycles = [workload_state.arrival_cycle for workload_state in self._workload_states.values() if workload_state.is_suspended and workload_state.arrival_cycle > self._device.timestamp]
            next_workload_arrival = min(arrival_cycles) if arrival_cycles else None
            self._device.host_run_until(stop_condition=lambda: bool(completion_events), max_timestamp=next_workload_arrival)

            if completion_events:
                completed_events, completion_events = completion_events, []
                for token in completed_events:
                    workload_state = token.workload_state
                    token.kernel_invocation.start_cycle = token.host_job.actual_start
                    token.kernel_invocation.completion_cycle = self._device.timestamp
                    token.kernel_invocation.state = MeshKernelState.COMPLETED
                    self._running_kernel_ids.discard(token.kernel_invocation.execution_id)
                    self._get_action_post_hook()(token)
                    self.kernel_materializer.release_kernel(workload_state, token.action.kernel_id)
                    for tensor_id in token.action.release_tensor_ids:
                        self.kernel_materializer.release_tensor(workload_state, tensor_id)
                    workload_state.commit_current_actions()
                    self.kernel_materializer.scheduler.on_complete(workload_state.workload_id, token.action.kernel_id, self._device.timestamp)
                    if workload_state.is_completed:
                        workload_state._completion_cycle = self._device.timestamp
                        self.kernel_materializer.scheduler.on_workload_complete(workload_state.workload_id, self._device.timestamp)
                        self._on_workload_completed(workload_state)
            elif next_workload_arrival is None and not self._running_kernel_ids:
                raise RuntimeError("Runtime reached a deadlock with incomplete workloads.")

        return jobs

    def _reset_policy_state(self) -> None:
        return None

    def _select_dispatchable_workloads(self, ready_states: tuple[MeshDeviceRuntimeWorkloadState, ...]) -> tuple[MeshDeviceRuntimeWorkloadState, ...]:
        return ready_states

    def _get_scheduling_domain(self, workload_state: MeshDeviceRuntimeWorkloadState) -> MeshSchedulingDomain:
        if workload_state.domain_id not in (None, self._device_domain.domain_id):
            raise ValueError(f"Unknown scheduling domain '{workload_state.domain_id}'.")
        return self._device_domain

    def _on_workload_dispatched(self, workload_state: MeshDeviceRuntimeWorkloadState) -> None:
        return None

    def _on_workload_completed(self, workload_state: MeshDeviceRuntimeWorkloadState) -> None:
        return None

    def _get_action_post_hook(self) -> 'Callable[[MeshDeviceRuntimeKernelMaterialzer.Token], None]':
        def hook(token: MeshDeviceRuntimeKernelMaterialzer.Token):
            action = token.action
            _hs = self._action_post_hooks.get(action.action_type, [])

            for _h in _hs:
                _h(token)

            if self._enable_debug_log:
                _action_repr = f"{action.action_type.name}({action.kernel_id if action.action_type == MeshDeviceActionType.RUN_KERNEL else action.tensor_id if action.action_type == MeshDeviceActionType.PLACE_TENSOR else 'N/A'})"
                _action_details = ""
                if action.action_type == MeshDeviceActionType.PLACE_TENSOR:
                    _tensor_stats = token.workload_state.compiled_workload.get_tensor_stats(action.tensor_id)
                    _action_details = f"tensor_id: {action.tensor_id:<12s} | {_tensor_stats.__repr__()}"
                elif action.action_type == MeshDeviceActionType.RUN_KERNEL:
                    _kernel_stats = token.workload_state.compiled_workload.get_kernel_stats(action.kernel_id)
                    _action_details = f"kernel_id: {action.kernel_id:<12s} | {_kernel_stats.__repr__()}"
                logger.debug(f"timestamp: {self._device.timestamp:<10d} | action: {_action_repr:<24s} | {_action_details}")

        return hook

    @property
    def decision_log(self) -> tuple[dict, ...]:
        return tuple(dict(record) for record in self._decision_log)

    def get_workload(self, workload_id: str) -> MeshDeviceRuntimeWorkloadState:
        if workload_id not in self._workload_states:
            raise ValueError(f"Workload ID '{workload_id}' not found in runtime workload states.")
        return self._workload_states[workload_id]


class MeshDeviceRuntimeKernelMaterialzer:
    """
    Introduction
    ------------
    TBD
    """
    class Token:
        DONE = 0
        RUN_SIM = 1
        RETRY = 2

        def __init__(
            self,
            token_type: int,
            action: MeshDeviceCompiledAction,
            workload_state: MeshDeviceRuntimeWorkloadState,
            host_job: HostJob = None,
            retry_message: str = None,
            kernel_invocation: _MeshRuntimeKernelInvocation = None,
        ):
            self.token_type = token_type
            self.action = action
            self.workload_state = workload_state
            self.host_job = host_job
            self.retry_message = retry_message
            self.kernel_invocation = kernel_invocation

        @classmethod
        def done(cls, action: MeshDeviceCompiledAction, workload_state: MeshDeviceRuntimeWorkloadState):  return cls(cls.DONE, action, workload_state)
        @classmethod
        def run_sim(cls, action: MeshDeviceCompiledAction, workload_state: MeshDeviceRuntimeWorkloadState, host_job: HostJob):  return cls(cls.RUN_SIM, action, workload_state, host_job=host_job)
        @classmethod
        def retry(cls, action: MeshDeviceCompiledAction, workload_state: MeshDeviceRuntimeWorkloadState):  return cls(cls.RETRY, action, workload_state)
        @property
        def is_done(self) -> bool:  return self.token_type == self.DONE
        @property
        def is_run_sim(self) -> bool:  return self.token_type == self.RUN_SIM
        @property
        def is_retry(self) -> bool:  return self.token_type == self.RETRY

    def __init__(self, scheduler: MeshDeviceScheduler = None):
        self.scheduler = MeshFRFCFSScheduler() if scheduler is None else scheduler

        if not isinstance(self.scheduler, MeshDeviceScheduler):
            raise TypeError(f"Expected MeshDeviceScheduler, got {type(scheduler).__name__}.")

        self.mesh_kernel_class_map: dict[MeshKernelType, type[MeshKernel]] = {
            MeshKernelType.SDPA:        MeshSDPAKernel,
            MeshKernelType.LINEAR:      MeshLinearKernel,
            MeshKernelType.CONV2D:      MeshConv2dKernel,
            MeshKernelType.ELEMENTWISE: MeshElementwiseKernel,
            MeshKernelType.REDUCTION:   MeshReductionKernel,
            MeshKernelType.MEMCOPY:     MeshMemCopyKernel,
        }

        self._registered_context: MeshAcceleratorRuntimeContext | None = None
        self._submitted_actions: list[tuple[MeshDeviceCompiledAction, MeshDeviceRuntimeWorkloadState, MeshSchedulingDomain]] = []
        self._persistent_state_placements: dict[str, MeshDeviceCompiledTensorStats] = {}
        self._resident_weight_placements: dict[tuple, MeshDeviceCompiledTensorStats] = {}
        self._resident_weight_users: dict[tuple, set[MeshDeviceCompiledWorkload]] = {}
        self._resident_weight_bindings: dict[MeshDeviceCompiledWorkload, set[tuple]] = {}

    def replace_mesh_kernel_class(self, kernel_type: MeshKernelType, kernel_class: 'type[MeshKernel]') -> None:
        if not issubclass(kernel_class, MeshKernel):
            raise TypeError(f"Expected a subclass of MeshKernel, got {kernel_class.__name__}.")
        self.mesh_kernel_class_map[kernel_type] = kernel_class

    def register_context(self, context: MeshAcceleratorRuntimeContext) -> None:
        if self._registered_context is not None:
            raise RuntimeError("A context is already registered.")
        self._registered_context = context

    def submit_action(self, action: MeshDeviceCompiledAction, workload_state: MeshDeviceRuntimeWorkloadState, domain: MeshSchedulingDomain | None = None):
        domain = self.scheduler._full_device_domain(self._registered_context) if domain is None else domain
        self._submitted_actions.append((action, workload_state, domain))

    def schedule_actions(self, running_kernel_ids: tuple[str, ...] = ()) -> tuple[list[Token], list[Token]]:
        _suspended_kernel_actions: list[tuple[MeshDeviceCompiledAction, MeshDeviceRuntimeWorkloadState]] = []
        _suspended_tensor_actions: list[tuple[MeshDeviceCompiledAction, MeshDeviceRuntimeWorkloadState]] = []
        _tensor_placement_tokens: list[MeshDeviceRuntimeKernelMaterialzer.Token] = []
        _kernel_schedule_tokens: list[MeshDeviceRuntimeKernelMaterialzer.Token] = []

        # STEP 1: Run all tensor placement actions and collect kernel actions for scheduling
        for action, workload_state, domain in self._submitted_actions:
            if action.action_type == MeshDeviceActionType.PLACE_TENSOR:
                if self.place_tensor(workload_state, tensor_id=action.tensor_id, dma_ids=list(domain.dma_ids), ccg_ids=domain.ccg_tile_mesh.flatten().tolist()):
                    _tensor_placement_tokens.append(self.Token.done(action, workload_state))
                else:
                    _suspended_tensor_actions.append((action, workload_state, domain))
            else:
                _suspended_kernel_actions.append((action, workload_state, domain))

        # STEP 2: Schedule kernel actions using the scheduler and materialize them into HostJobs
        _scheduled_kernel_actions, _suspended_kernel_actions = self.scheduler.schedule_actions(self._registered_context, _suspended_kernel_actions, self._persistent_state_placements, running_kernel_ids)

        _scheduled_kernel_actions: list[tuple[MeshDeviceCompiledAction, MeshDeviceRuntimeWorkloadState]]
        _suspended_kernel_actions: list[tuple[MeshDeviceCompiledAction, MeshDeviceRuntimeWorkloadState]]

        # STEP 3: Materialize scheduled kernel actions into HostJobs and create RUN_SIM tokens
        for action, workload_state in _scheduled_kernel_actions:
            job = self.run_kernel(workload_state, action.kernel_id)
            _kernel_schedule_tokens.append(self.Token.run_sim(action, workload_state, host_job=job))

        # STEP 4: Return the list of tokens for the scheduled and suspended actions
        self._submitted_actions = _suspended_tensor_actions + _suspended_kernel_actions

        return _tensor_placement_tokens, _kernel_schedule_tokens

    def clear_submission(self):
        self._submitted_actions.clear()

    def is_submission_empty(self) -> bool:
        return len(self._submitted_actions) == 0

    @staticmethod
    def _weight_residency_key(tensor_desc: MeshTensorDescriptor, dma_ids: list[int], ccg_ids: list[int]) -> tuple:
        storage_desc = tensor_desc.storage_desc
        parameter_id = getattr(storage_desc, "parameter_id", None)
        identity = ("parameter", parameter_id) if isinstance(parameter_id, str) and parameter_id else ("descriptor", id(storage_desc))
        owner_ids = ccg_ids if storage_desc.preferred_mem == MeshMemoryType.LOCAL_CACHE else dma_ids
        return identity, storage_desc.preferred_mem, tuple(sorted(int(owner_id) for owner_id in owner_ids))

    @staticmethod
    def _weight_descriptor_signature(tensor_desc: MeshTensorDescriptor) -> tuple:
        return tensor_desc.shape, tensor_desc.reserved_shape, tensor_desc.tile_shape, tensor_desc.dtype, tensor_desc.tensor_type, tensor_desc.preferred_mem

    def _place_weight_tensor(self, workload_state: MeshDeviceRuntimeWorkloadState, tensor_stats: MeshDeviceCompiledTensorStats, dma_ids: list[int], ccg_ids: list[int]) -> bool:
        descriptor = tensor_stats.tensor_desc
        storage_desc = descriptor.storage_desc
        key = self._weight_residency_key(storage_desc, dma_ids, ccg_ids)
        placement = self._resident_weight_placements.get(key)
        if placement is not None and self._weight_descriptor_signature(placement.tensor_desc) != self._weight_descriptor_signature(storage_desc):
            raise ValueError(f"Weight parameter {key[0]} is already registered with an incompatible descriptor.")
        if placement is None:
            placement = MeshDeviceCompiledTensorStats(storage_desc)
            if not self.place_tensor_stats(placement, dma_ids=dma_ids, ccg_ids=ccg_ids):
                return False
            self._resident_weight_placements[key] = placement
        workload = workload_state.compiled_workload
        self._resident_weight_users.setdefault(key, set()).add(workload)
        self._resident_weight_bindings.setdefault(workload, set()).add(key)
        tensor_stats.place({coord: placement.tile_placement[descriptor.map_tile_coord_to_storage(coord)] for coord in descriptor.get_tile_coords()})
        return True

    def place_tensor(self, workload_state: MeshDeviceRuntimeWorkloadState, tensor_id: str, dma_ids: list[int]=None, ccg_ids: list[int]=None) -> bool:
        # STEP 1: Retrieve tensor stats from the workload
        tensor_stats = workload_state.compiled_workload.tensor_stats_map.get(tensor_id, None)

        if dma_ids is None:
            dma_ids = self._registered_context.dma_tile_ids
        if ccg_ids is None:
            ccg_ids = self._registered_context.ccg_tile_ids

        if tensor_stats is None:
            raise ValueError(f"Tensor ID '{tensor_id}' not found in workload tensor stats map.")

        descriptor = tensor_stats.tensor_desc
        if descriptor.is_persistent:
            placement = self._persistent_state_placements.get(descriptor.storage_desc.persistent_state_id)
            if placement is None:
                placement = MeshDeviceCompiledTensorStats(descriptor.storage_desc)
                if not self.place_tensor_stats(placement, dma_ids=dma_ids, ccg_ids=ccg_ids):
                    return False
                self._persistent_state_placements[descriptor.storage_desc.persistent_state_id] = placement
            tensor_stats.place({coord: placement.tile_placement[descriptor.map_tile_coord_to_storage(coord)] for coord in descriptor.get_tile_coords()})
            return True
        if descriptor.tensor_type == MeshTensorType.WEIGHT:
            return self._place_weight_tensor(workload_state, tensor_stats, dma_ids, ccg_ids)
        if descriptor.is_view:
            storage_stats = next((stats for stats in workload_state.compiled_workload.tensor_stats_map.values() if stats.tensor_desc is descriptor.storage_desc), None)
            if storage_stats is None:
                raise RuntimeError("Tensor view storage is not registered in the compiled workload.")
            if not storage_stats.is_placed and not self.place_tensor_stats(storage_stats, dma_ids=dma_ids, ccg_ids=ccg_ids):
                return False
            tensor_stats.place({coord: storage_stats.tile_placement[descriptor.map_tile_coord_to_storage(coord)] for coord in descriptor.get_tile_coords()})
            return True
        return self.place_tensor_stats(tensor_stats, dma_ids=dma_ids, ccg_ids=ccg_ids)

    def place_tensor_stats(self, tensor_stats: MeshDeviceCompiledTensorStats, dma_ids: list[int]=None, ccg_ids: list[int]=None) -> bool:
        if tensor_stats.is_placed:
            return True
        if dma_ids is None:
            dma_ids = self._registered_context.dma_tile_ids
        if ccg_ids is None:
            ccg_ids = self._registered_context.ccg_tile_ids
        tile_size = tensor_stats.tensor_desc.get_tile_size()
        n_tiles = tensor_stats.tensor_desc.get_reserved_n_tiles()
        tile_coords = list(np.ndindex(*tensor_stats.tensor_desc.reserved_tile_grid_shape))

        # STEP 2: Attempt to place tensor in local cache if preferred
        if tensor_stats.tensor_desc.preferred_mem == MeshMemoryType.LOCAL_CACHE:
            allocated_banks = self._registered_context.allocate_local_cache(ccg_ids=ccg_ids, size=tile_size, n_banks=n_tiles)

            if allocated_banks is not None:
                bank_desc_arr = [MeshMemoryBankDescriptor.LOCAL_CACHE(tile_id=ccg_id, addr=addr, size=size) for ccg_id, addr, size in allocated_banks]
                tensor_stats.place(tile_placement={coord: bank_desc for coord, bank_desc in zip(tile_coords, bank_desc_arr)})

                return True  # Successfully placed in local cache, no need to fallback to device memory

        # STEP 3: If local cache placement failed or was not preferred, attempt to place tensor in device memory
        allocated_banks = self._registered_context.allocate_device_memory(dma_ids=dma_ids, size=tile_size, n_banks=n_tiles,)

        if allocated_banks is not None:
            bank_desc_arr = [MeshMemoryBankDescriptor.DEVICE_MEMORY(addr=addr, size=size) for dma_id, addr, size in allocated_banks]
            tensor_stats.place(tile_placement={coord: bank_desc for coord, bank_desc in zip(tile_coords, bank_desc_arr)})
        else:
            return False  # Failed to place tensor in both local cache and device memory

        return True

    def release_tensor_stats(self, tensor_stats: MeshDeviceCompiledTensorStats) -> None:
        for mem_bank_desc in tensor_stats.tile_placement.values():
            if mem_bank_desc.mem_type == MeshMemoryType.DEVICE_MEMORY:
                self._registered_context.deallocate_device_memory(addr=mem_bank_desc.addr, size=mem_bank_desc.size)
            elif mem_bank_desc.mem_type == MeshMemoryType.LOCAL_CACHE:
                self._registered_context.deallocate_local_cache(ccg_id=mem_bank_desc.owner_id, addr=mem_bank_desc.addr, size=mem_bank_desc.size)
        tensor_stats.unplace()

    def release_tensor(self, workload_state: MeshDeviceRuntimeWorkloadState, tensor_id: str) -> None:
        tensor_stats = workload_state.compiled_workload.tensor_stats_map.get(tensor_id, None)

        if tensor_stats is None:
            raise ValueError(f"Tensor ID '{tensor_id}' not found in workload tensor stats map.")

        storage_desc = tensor_stats.tensor_desc.storage_desc
        aliases = [stats for stats in workload_state.compiled_workload.tensor_stats_map.values() if stats.tensor_desc.storage_desc is storage_desc]
        owner = next((stats for stats in aliases if stats.tensor_desc is storage_desc), tensor_stats)
        if not storage_desc.is_persistent and owner.is_placed:
            self.release_tensor_stats(owner)
        for alias in aliases:
            if alias is not owner:
                alias.unplace()

    def run_kernel(self, workload_state: MeshDeviceRuntimeWorkloadState, kernel_id: str) -> HostJob:
        # STEP 1: Retrieve kernel stats from the workload
        workload = workload_state.compiled_workload
        kernel_stats = workload.kernel_stats_map.get(kernel_id, None)

        if kernel_stats is None:
            raise ValueError(f"Kernel ID '{kernel_id}' not found in workload kernel stats map.")

        if not kernel_stats.is_placed:
            raise RuntimeError(f"Kernel '{kernel_id}' was not placed before materialization.")

        if any(self._registered_context.ccg_kernel_vacancy.get(ccg_id, True) for ccg_id in kernel_stats.ccg_tile_mesh.flatten().tolist()):
            raise RuntimeError(f"Kernel '{kernel_id}' placement was not reserved by the scheduler.")

        for tensor_desc in kernel_stats.kernel_desc.input_tensors + kernel_stats.kernel_desc.output_tensors:
            tensor_id = workload.get_tensor_id(tensor_desc)
            if not workload.get_tensor_stats(tensor_id).is_placed:
                raise RuntimeError(f"Tensor '{tensor_id}' was not placed by the scheduler for kernel '{kernel_id}'.")

        job = self.materialize_kernel(self._registered_context, workload, kernel_stats)
        job.annotate_schedule(release_timestamp=self._registered_context.device.timestamp)

        return job  # Return the HostJob for simulation execution

    def release_kernel(self, workload_state: MeshDeviceRuntimeWorkloadState, kernel_id: str) -> None:
        workload = workload_state.compiled_workload
        kernel_stats = workload.kernel_stats_map.get(kernel_id, None)

        if kernel_stats is None:
            raise ValueError(f"Kernel ID '{kernel_id}' not found in workload kernel stats map.")

        self._registered_context.deallocate_ccg_kernel(ccg_tile_mesh=kernel_stats.ccg_tile_mesh)
        kernel_stats.unplace()

    def materialize_kernel(
        self,
        device_rt_context: MeshAcceleratorRuntimeContext,
        workload: MeshDeviceCompiledWorkload,
        kernel_stats: MeshDeviceCompiledKernelStats,
    ) -> HostJob:
        return _materialize_mesh_kernel_job(
            device_rt_context.device,
            kernel_stats.ccg_tile_mesh,
            device_rt_context,
            workload,
            kernel_stats,
            self.mesh_kernel_class_map,
        )

@jit_host_job_prototype
def _materialize_mesh_kernel_job(
    device: MeshAccelerator,
    core_mesh: np.ndarray,
    device_rt_context: MeshAcceleratorRuntimeContext,
    workload: MeshDeviceCompiledWorkload,
    kernel_stats: MeshDeviceCompiledKernelStats,
    mesh_kernel_class_map: 'dict[MeshKernelType, type[MeshKernel]]'
) -> HostJob:
    kernel_type = kernel_stats.kernel_desc.kernel_type
    kernel_class = mesh_kernel_class_map.get(kernel_type, None)

    if kernel_class is None:
        raise ValueError(f"Unsupported kernel type: {kernel_type}")

    kernel = kernel_class(device_rt_context=device_rt_context, workload=workload, kernel_stats=kernel_stats)

    job = HostJob()
    # try:
    for slot_id, program in kernel.get_programs().items():
        job.add_program(slot_id=slot_id, program=program)
    # except Exception:
    #     kernel.release_remote_cache()
    #     raise
    # if kernel.remote_cache_allocations:
    #     job.add_commit_hook(lambda _, kernel=kernel: kernel.release_remote_cache())
    return job


def _mesh_tile_elements(tensor: MeshTensorDescriptor, coord: tuple[int, ...]) -> int:
    return math.prod(max(0, min(size, (index + 1) * tile_size) - index * tile_size) for index, tile_size, size in zip(coord, tensor.tile_shape, tensor.shape))


def _mesh_contributions(kernel: 'MeshKernel', task: LocatedTask):
    descriptor = kernel.kernel_stats.kernel_desc
    inputs = dict(task.inputs)
    if kernel.kernel_type == MeshKernelType.LINEAR:
        ifm_axis = -2 if descriptor.get_required_kwargs("transpose_ifm") else -1
        wgt_axis = -2 if descriptor.get_required_kwargs("transpose_wgt") else -1
        units = []
        for reduction in sorted({location.ref.coord[ifm_axis] for location in inputs[0]}):
            ifm_tiles = tuple(location for location in inputs[0] if location.ref.coord[ifm_axis] == reduction)
            wgt_tiles = tuple(location for location in inputs[1] if location.ref.coord[wgt_axis] == reduction)
            if not wgt_tiles:
                raise ValueError("Linear reduction tiles have no matching weights.")
            units.append(((0, ifm_tiles), (1, wgt_tiles)))
        if len(descriptor.input_tensors) > 2:
            units[-1] += ((2, inputs[2]),)
        return units
    if kernel.kernel_type == MeshKernelType.CONV2D and descriptor.get_required_kwargs("operation") == "conv2d":
        ifm, wgt = descriptor.input_tensors[:2]
        ofm = descriptor.output_tensors[0]
        coord = task.output.ref.coord
        stride_height, stride_width = descriptor.get_required_kwargs("stride")
        padding_top, _, padding_left, _ = descriptor.get_required_kwargs("padding")
        dilation_height, dilation_width = descriptor.get_required_kwargs("dilation")
        input_channels_per_group = ifm.shape[3] // descriptor.get_required_kwargs("groups")
        output_channels_per_group = ofm.shape[3] // descriptor.get_required_kwargs("groups")
        row_range = range(coord[1] * ofm.tile_shape[1], min(ofm.shape[1], (coord[1] + 1) * ofm.tile_shape[1]))
        column_range = range(coord[2] * ofm.tile_shape[2], min(ofm.shape[2], (coord[2] + 1) * ofm.tile_shape[2]))
        output_channels = range(coord[3] * ofm.tile_shape[3], min(ofm.shape[3], (coord[3] + 1) * ofm.tile_shape[3]))
        units = []
        for weight in inputs[1]:
            kh, kw, weight_output, weight_input = weight.ref.coord
            groups = {channel // output_channels_per_group for channel in output_channels if channel // wgt.tile_shape[2] == weight_output}
            if not groups:
                continue
            spatial = {(coord[0], input_row // ifm.tile_shape[1], input_column // ifm.tile_shape[2]) for row in row_range for column in column_range for input_row, input_column in ((row * stride_height + kh * dilation_height - padding_top, column * stride_width + kw * dilation_width - padding_left),) if 0 <= input_row < ifm.shape[1] and 0 <= input_column < ifm.shape[2]}
            channels = {channel // ifm.tile_shape[3] for group in groups for channel in range(group * input_channels_per_group + weight_input * wgt.tile_shape[3], min((group + 1) * input_channels_per_group, group * input_channels_per_group + (weight_input + 1) * wgt.tile_shape[3]))}
            ifm_tiles = tuple(location for location in inputs[0] if location.ref.coord[:3] in spatial and location.ref.coord[3] in channels)
            if ifm_tiles:
                units.append(((0, ifm_tiles), (1, (weight,))))
        if not units:
            raise ValueError("Conv2d output has no valid input-weight contributions.")
        if len(descriptor.input_tensors) > 2:
            units[-1] += ((2, inputs[2]),)
        return units
    if kernel.kernel_type == MeshKernelType.SDPA:
        units = []
        for key_tile in sorted({location.ref.coord[2] for location in inputs[1]}):
            key_tiles = tuple(location for location in inputs[1] if location.ref.coord[2] == key_tile)
            value_tiles = tuple(location for location in inputs[2] if location.ref.coord[2] == key_tile)
            if not value_tiles:
                raise ValueError("SDPA key tiles have no matching value tiles.")
            unit = ((0, inputs[0]), (1, key_tiles), (2, value_tiles))
            if 3 in inputs:
                mask_tile = 0 if descriptor.input_tensors[3].tile_grid_shape[-1] == 1 else key_tile
                mask_tiles = tuple(location for location in inputs[3] if location.ref.coord[-1] == mask_tile)
                unit += ((3, mask_tiles),)
            units.append(unit)
        return units
    raise ValueError(f"Kernel-specific contributions are unavailable for {kernel.kernel_type}.")


def _mesh_stage_ops(kernel: 'MeshKernel', stage: StagePlan) -> int:
    if stage.kind != "COMPUTE":
        return 0
    descriptor = kernel.kernel_stats.kernel_desc
    ofm = descriptor.output_tensors[0]
    total = 0
    for task, source in zip(stage.group.tasks, stage.group.source_tasks):
        output_elements = _mesh_tile_elements(ofm, task.output.ref.coord)
        inputs = dict(task.inputs)
        if kernel.kernel_type == MeshKernelType.MEMCOPY:
            continue
        if kernel.kernel_type == MeshKernelType.ELEMENTWISE:
            total += output_elements * float(descriptor.get_required_kwargs("ops_per_element"))
        elif kernel.kernel_type == MeshKernelType.REDUCTION:
            ifm = descriptor.input_tensors[0]
            input_elements = sum(_mesh_tile_elements(ifm, location.ref.coord) for location in inputs[0])
            total += input_elements * float(descriptor.get_required_kwargs("ops_per_input_element"))
            if stage.group.last_chunk:
                total += output_elements * float(descriptor.get_required_kwargs("extra_ops_per_output_element"))
        elif kernel.kernel_type == MeshKernelType.LINEAR:
            ifm = descriptor.input_tensors[0]
            reduction_axis = -2 if descriptor.get_required_kwargs("transpose_ifm") else -1
            reduction_indices = {location.ref.coord[reduction_axis] for location in inputs[0]}
            reduction_size = sum(max(0, min(ifm.shape[reduction_axis], (index + 1) * ifm.tile_shape[reduction_axis]) - index * ifm.tile_shape[reduction_axis]) for index in reduction_indices)
            total += 2 * output_elements * reduction_size
            if stage.group.last_chunk and len(descriptor.input_tensors) > 2:
                total += output_elements
            if stage.group.last_chunk:
                total += output_elements * float(descriptor.get_required_kwargs("extra_ops_per_output_element"))
        elif kernel.kernel_type == MeshKernelType.CONV2D:
            operation = descriptor.get_required_kwargs("operation")
            ifm = descriptor.input_tensors[0]
            kernel_height, kernel_width = descriptor.get_required_kwargs("kernel_size")
            stride_height, stride_width = descriptor.get_required_kwargs("stride")
            padding_top, _, padding_left, _ = descriptor.get_required_kwargs("padding")
            dilation_height, dilation_width = descriptor.get_required_kwargs("dilation")
            coord = task.output.ref.coord
            row_start, column_start = coord[1] * ofm.tile_shape[1], coord[2] * ofm.tile_shape[2]
            row_end, column_end = min(ofm.shape[1], row_start + ofm.tile_shape[1]), min(ofm.shape[2], column_start + ofm.tile_shape[2])
            output_batches = max(0, min(ofm.shape[0], (coord[0] + 1) * ofm.tile_shape[0]) - coord[0] * ofm.tile_shape[0])
            if operation == "conv2d":
                wgt = descriptor.input_tensors[1]
                groups = descriptor.get_required_kwargs("groups")
                input_channels_per_group = ifm.shape[3] // groups
                output_channels_per_group = ofm.shape[3] // groups
                input_tiles = {(location.ref.coord[1], location.ref.coord[2], location.ref.coord[3]) for location in inputs[0]}
                weight_tiles = [location.ref.coord for location in inputs[1]]
                for row in range(row_start, row_end):
                    for column in range(column_start, column_end):
                        for kh in range(kernel_height):
                            input_row = row * stride_height + kh * dilation_height - padding_top
                            if input_row < 0 or input_row >= ifm.shape[1]:
                                continue
                            for kw in range(kernel_width):
                                input_column = column * stride_width + kw * dilation_width - padding_left
                                if input_column < 0 or input_column >= ifm.shape[2]:
                                    continue
                                for weight_row, weight_column, weight_output, weight_input in weight_tiles:
                                    if weight_row != kh or weight_column != kw:
                                        continue
                                    for group in range(groups):
                                        output_start = max(coord[3] * ofm.tile_shape[3], group * output_channels_per_group, weight_output * wgt.tile_shape[2])
                                        output_end = min(ofm.shape[3], (coord[3] + 1) * ofm.tile_shape[3], (group + 1) * output_channels_per_group, (weight_output + 1) * wgt.tile_shape[2])
                                        if output_start >= output_end:
                                            continue
                                        input_start = group * input_channels_per_group + weight_input * wgt.tile_shape[3]
                                        input_end = min((group + 1) * input_channels_per_group, input_start + wgt.tile_shape[3])
                                        for channel_tile in range(input_start // ifm.tile_shape[3], (input_end - 1) // ifm.tile_shape[3] + 1):
                                            if (input_row // ifm.tile_shape[1], input_column // ifm.tile_shape[2], channel_tile) in input_tiles:
                                                channel_count = max(0, min(input_end, (channel_tile + 1) * ifm.tile_shape[3]) - max(input_start, channel_tile * ifm.tile_shape[3]))
                                                total += 2 * output_batches * (output_end - output_start) * channel_count
                if stage.group.last_chunk and len(descriptor.input_tensors) > 2:
                    total += output_elements
            else:
                source_tiles = {location.ref.coord for location in source.inputs[0][1]}
                fragment_tiles = {location.ref.coord for location in inputs[0]}
                for batch in range(coord[0] * ofm.tile_shape[0], min(ofm.shape[0], (coord[0] + 1) * ofm.tile_shape[0])):
                    for row in range(row_start, row_end):
                        for column in range(column_start, column_end):
                            for channel in range(coord[3] * ofm.tile_shape[3], min(ofm.shape[3], (coord[3] + 1) * ofm.tile_shape[3])):
                                touched = []
                                for kh in range(kernel_height):
                                    input_row = row * stride_height + kh * dilation_height - padding_top
                                    for kw in range(kernel_width):
                                        input_column = column * stride_width + kw * dilation_width - padding_left
                                        if 0 <= input_row < ifm.shape[1] and 0 <= input_column < ifm.shape[2]:
                                            touched.append((batch // ifm.tile_shape[0], input_row // ifm.tile_shape[1], input_column // ifm.tile_shape[2], channel // ifm.tile_shape[3]))
                                contributions = sum(tile in fragment_tiles for tile in touched)
                                if operation == "avg_pool2d":
                                    total += contributions + int(stage.group.last_chunk and bool(touched))
                                elif touched:
                                    first_tile = next(tile for tile in touched if tile in source_tiles)
                                    total += contributions - int(first_tile in fragment_tiles)
            if stage.group.last_chunk:
                total += output_elements * float(descriptor.get_required_kwargs("extra_ops_per_output_element"))
        elif kernel.kernel_type == MeshKernelType.SDPA:
            q, k, v = descriptor.input_tensors[:3]
            coord = task.output.ref.coord
            query_start = coord[2] * ofm.tile_shape[2]
            query_end = min(q.shape[2], query_start + ofm.tile_shape[2])
            output_width = min(ofm.shape[3], (coord[3] + 1) * ofm.tile_shape[3]) - coord[3] * ofm.tile_shape[3]
            causal = descriptor.get_required_kwargs("is_causal")
            scores = 0
            for key_tile in {location.ref.coord[2] for location in inputs[1]}:
                key_start = key_tile * k.tile_shape[2]
                key_end = min(k.shape[2], key_start + k.tile_shape[2])
                scores += sum(max(0, min(key_end, k.shape[2] - q.shape[2] + query + 1) - key_start) if causal else key_end - key_start for query in range(query_start, query_end))
            total += scores * (2 * q.shape[3] + 2 * output_width + int(descriptor.get_required_kwargs("softmax_ops_per_score")))
        else:
            raise ValueError(f"Unsupported mesh kernel type: {kernel.kernel_type}")
    return math.ceil(total)


def _mesh_plan_ld_reuse(stages: dict[int, list[StagePlan]], scratch_plans: dict[int, ScratchPlan], residency_plan: ResidencyPlan) -> dict[int, list[StagePlan]]:
    if residency_plan.full_resident:
        return stages
    optimized = {}
    for core_id, core_stages in stages.items():
        scratch = scratch_plans[core_id]
        contents = {0: {}, 1: {}}
        future = {0: {}, 1: {}}
        for index, stage in enumerate(core_stages):
            if stage.kind != "COMPUTE":
                continue
            slot = stage.group.index % 2
            for location in [transfer.tile for transfer in stage.loads] + [location for location in stage.local_reads if location.owner_id != core_id]:
                key = (location.storage_key, location.mem_type, location.owner_id, location.addr, location.size)
                future[slot].setdefault(key, []).append(index)
        if not any(len(indices) > 1 for uses in future.values() for indices in uses.values()):
            optimized[core_id] = core_stages
            continue
        optimized_stages = []
        for index, stage in enumerate(core_stages):
            if stage.kind != "COMPUTE":
                optimized_stages.append(stage)
                continue
            slot = stage.group.index % 2
            partial_local = scratch.output_local and not (stage.group.first_chunk and stage.group.last_chunk)
            regions = _runtime_utils._split_regions(scratch.ld_regions)[slot] if partial_local else scratch.ld_slots[slot]
            regions = _runtime_utils._without_regions(regions, stage.scratch_regions)
            original = {(transfer.tile.storage_key, transfer.tile.mem_type, transfer.tile.owner_id, transfer.tile.addr, transfer.tile.size): transfer for transfer in stage.loads}
            mandatory = {key: transfer.tile for key, transfer in original.items()}
            remote = {}
            for location in stage.local_reads:
                if location.owner_id == core_id:
                    continue
                key = (location.storage_key, location.mem_type, location.owner_id, location.addr, location.size)
                if key in contents[slot] or any(value > index for value in future[slot].get(key, ())):
                    remote[key] = location

            def plan(required):
                keep = {key: item for key, item in contents[slot].items() if any(region.addr <= item[1].addr and item[1].addr + item[1].size <= region.addr + region.size for region in regions)}
                while True:
                    missing = [key for key in required if key not in keep]
                    free = _runtime_utils._without_regions(regions, tuple(item[1] for item in keep.values()))
                    packed = _runtime_utils._pack_regions([required[key].size for key in missing], free)
                    if packed is not None:
                        assignments = {key: item[1] for key, item in keep.items() if key in required}
                        assignments.update(zip(missing, packed))
                        keep.update((key, (required[key], region)) for key, region in zip(missing, packed))
                        return assignments, keep
                    evictable = [key for key in keep if key not in required]
                    if not evictable:
                        packed = _runtime_utils._pack_regions([location.size for location in required.values()], regions)
                        if packed is None:
                            return None
                        assignments = dict(zip(required, packed))
                        return assignments, {key: (required[key], region) for key, region in assignments.items()}
                    victim = max(evictable, key=lambda key: next((value for value in future[slot].get(key, ()) if value > index), math.inf))
                    del keep[victim]

            chosen = dict(mandatory)
            result = plan(chosen)
            if result is None:
                raise ValueError(f"Required input tiles do not fit the fixed LD slot on core {core_id}.")
            assignments, retained = result
            free = [[region.addr, region.size] for region in _runtime_utils._without_regions(regions, tuple(item[1] for item in retained.values()))]
            for key, location in sorted(remote.items(), key=lambda item: (-len(future[slot].get(item[0], ())), item[1].size)):
                if key in retained:
                    chosen[key] = location
                    assignments[key] = retained[key][1]
                    continue
                target = next((region for region in free if region[1] >= location.size), None)
                if target is None:
                    continue
                assigned = BufferRegion(core_id, target[0], location.size)
                target[0] += location.size
                target[1] -= location.size
                chosen[key] = location
                assignments[key] = assigned
                retained[key] = (location, assigned)
            loads = tuple(PlannedTransfer(location, assignments[key], original[key].consumers if key in original else ()) for key, location in chosen.items())
            local_reads = tuple(location for location in stage.local_reads if (location.storage_key, location.mem_type, location.owner_id, location.addr, location.size) not in chosen)
            load_writes = [assignments[key] for key in chosen if key not in contents[slot] or contents[slot][key][1] != assignments[key]]
            compute_writes = list(stage.scratch_regions)
            compute_writes.extend(BufferRegion(core_id, output.addr, output.size) for output in stage.outputs if output.owner_id == core_id and output.mem_type == MeshMemoryType.LOCAL_CACHE)
            contents[slot] = {key: item for key, item in retained.items() if any(value > index for value in future[slot].get(key, ()))}
            for other_slot in contents:
                contents[other_slot] = {key: item for key, item in contents[other_slot].items() if not any(item[1].addr < region.addr + region.size and region.addr < item[1].addr + item[1].size for region in compute_writes) and not any(item[1].addr < region.addr + region.size and region.addr < item[1].addr + item[1].size and (other_slot != slot or key not in chosen or item[1] != assignments[key]) for region in load_writes)}
            optimized_stages.append(replace(stage, ld_slot=slot if loads else None, loads=loads, local_reads=local_reads))
        optimized[core_id] = optimized_stages
    return optimized


class MeshKernel(ABC):
    kernel_type: MeshKernelType | None = None

    def __init__(self, device_rt_context: MeshAcceleratorRuntimeContext, workload: MeshDeviceCompiledWorkload, kernel_stats: MeshDeviceCompiledKernelStats):
        if not isinstance(device_rt_context, MeshAcceleratorRuntimeContext):
            raise TypeError(f"Expected MeshAcceleratorRuntimeContext, got {type(device_rt_context).__name__}")
        if not isinstance(workload, MeshDeviceCompiledWorkload):
            raise TypeError(f"Expected MeshDeviceCompiledWorkload, got {type(workload).__name__}")
        if not isinstance(kernel_stats, MeshDeviceCompiledKernelStats):
            raise TypeError(f"Expected MeshDeviceCompiledKernelStats, got {type(kernel_stats).__name__}")
        if not kernel_stats.is_placed:
            raise ValueError("Kernel stats must be placed before materialization.")

        self.device_rt_context = device_rt_context
        self.device = device_rt_context.device
        self.workload = workload
        self.kernel_stats = kernel_stats
        self.core_mesh = np.asarray(kernel_stats.ccg_tile_mesh, dtype=int).copy()
        self.input_placements = [workload.get_tensor_stats(tensor_desc) for tensor_desc in kernel_stats.kernel_desc.input_tensors]
        self.output_placements = [workload.get_tensor_stats(tensor_desc) for tensor_desc in kernel_stats.kernel_desc.output_tensors]
        self.base_input_count = kernel_stats.kernel_desc.base_input_count

        if self.core_mesh.ndim != 2 or self.core_mesh.size == 0 or np.any(self.core_mesh < 0):
            raise ValueError(f"Invalid core mesh: shape={self.core_mesh.shape}, values={self.core_mesh.tolist()}")
        if self.kernel_type is not None and kernel_stats.kernel_desc.kernel_type != self.kernel_type:
            raise ValueError(f"{type(self).__name__} cannot execute {kernel_stats.kernel_desc.kernel_type}.")
        if any(not placement.is_placed for placement in self.input_placements + self.output_placements):
            raise ValueError("All kernel tensors must be placed before materialization.")
        self.core_ids = tuple(int(core_id) for core_id in self.core_mesh.flatten().tolist())
        if any(core_id not in device_rt_context.ld_buffer_ptrs or core_id not in device_rt_context.st_buffer_ptrs for core_id in self.core_ids):
            raise ValueError("Every placed core must have reserved LD and ST buffers.")

        self.ld_buffer_size = min(device_rt_context.ld_buffer_ptrs[core_id][1] for core_id in self.core_ids)
        self.st_buffer_size = min(device_rt_context.st_buffer_ptrs[core_id][1] for core_id in self.core_ids)
        self.ld_slot_size = self.ld_buffer_size // 2
        self.st_slot_size = self.st_buffer_size // 2
        if self.ld_slot_size <= 0 or self.st_slot_size <= 0:
            raise ValueError("LD and ST buffers must each contain two non-empty slots.")
        self.mapping: _Mapping = None

    def create_stage_plans(self, mapping: _Mapping) -> dict[int, list[StagePlan]]:
        if not isinstance(mapping, _Mapping):
            raise TypeError(f"Expected _Mapping, got {type(mapping).__name__}.")
        if not np.array_equal(mapping.core_mesh, self.core_mesh):
            raise ValueError("Mapping core mesh does not match the kernel core mesh.")
        tasks = collect_core_tasks(mapping, getattr(self, "scratch_bytes_per_output", None))
        located_tasks = resolve_tile_locations(tasks, self.input_placements, self.output_placements)
        scratch_plans = plan_scratch_regions(self.core_mesh, self.device_rt_context.ld_buffer_ptrs, self.device_rt_context.st_buffer_ptrs, located_tasks)
        residency_plan = plan_full_residency(located_tasks, scratch_plans)
        split_task = (lambda task: _mesh_contributions(self, task)) if self.kernel_type in (MeshKernelType.LINEAR, MeshKernelType.CONV2D, MeshKernelType.SDPA) else None
        groups = {core_id: group_output_tasks(core_tasks, scratch_plans[core_id], residency_plan, split_task, merge_partial=self.kernel_type == MeshKernelType.LINEAR) for core_id, core_tasks in located_tasks.items()}
        transfers = {core_id: [plan_group_transfers(group, residency_plan, scratch_plans[core_id]) for group in core_groups] for core_id, core_groups in groups.items()}
        stage_plans = schedule_core_stages(groups, transfers, scratch_plans, residency_plan)
        stage_plans = _mesh_plan_ld_reuse(stage_plans, scratch_plans, residency_plan)
        validate_materialization_plan(stage_plans, scratch_plans, residency_plan)
        self.scratch_plans = scratch_plans
        self.residency_plan = residency_plan
        self.stage_plans = stage_plans
        return stage_plans

    @staticmethod
    @jit_prototype
    def ld_core_kernel(core: CCGTile, kernel_obj: 'MeshKernel', stage: StagePlan, stage_index: int, previous_use: int, ld_stage_var: VariableHandle, ex_stage_var: VariableHandle):
        if previous_use >= 0:
            core.var_conditional_wait(ex_stage_var, ex_stage_var.greater_equal(previous_use + 1))
        batches = {}
        remote_reads = {}
        for transfer in stage.loads:
            if transfer.tile.mem_type == MeshMemoryType.DEVICE_MEMORY:
                batches.setdefault(transfer.tile.size, []).append(transfer.tile.addr)
            elif transfer.tile.owner_id != core.core_id:
                remote_reads[transfer.tile.owner_id] = remote_reads.get(transfer.tile.owner_id, 0) + transfer.tile.size
        for size, addresses in batches.items():
            core.dma_read_memory_batch(addresses, size, sync=True)
        for owner_id, size in remote_reads.items():
            core.icnt_recv_data(owner_id, size, sync=True)
        core.var_atomic_increase(ld_stage_var)

    @staticmethod
    @jit_program_prototype
    def ld_program(device: MeshAccelerator, kernel_obj: 'MeshKernel', stages: dict[int, list[StagePlan]], _ld_core_kernel: Callable, _stage_vars: dict[str, dict[int, VariableHandle]]):
        for core_id in kernel_obj.core_ids:
            core = device.get_ccg_tile(core_id)
            previous_uses = {}
            slot_contents = {0: {}, 1: {}}
            for stage_index, stage in enumerate(stages[core_id]):
                previous_use = previous_uses.get(stage.ld_slot, -1) if stage.ld_slot is not None else -1
                if stage.ld_slot is not None:
                    contents = slot_contents[stage.ld_slot]
                    pending = tuple(transfer for transfer in stage.loads if contents.get((transfer.buffer.addr, transfer.buffer.size)) != (transfer.tile.storage_key, transfer.tile.mem_type, transfer.tile.owner_id, transfer.tile.addr, transfer.tile.size))
                    for transfer in pending:
                        for slot in slot_contents:
                            slot_contents[slot] = {address: key for address, key in slot_contents[slot].items() if not (address[0] < transfer.buffer.addr + transfer.buffer.size and transfer.buffer.addr < address[0] + address[1])}
                        slot_contents[stage.ld_slot][(transfer.buffer.addr, transfer.buffer.size)] = (transfer.tile.storage_key, transfer.tile.mem_type, transfer.tile.owner_id, transfer.tile.addr, transfer.tile.size)
                    load_stage = stage if len(pending) == len(stage.loads) else replace(stage, loads=pending)
                else:
                    load_stage = stage
                _ld_core_kernel(core, kernel_obj, load_stage, stage_index, previous_use, _stage_vars["LD"][core_id], _stage_vars["EX"][core_id])
                writes = list(stage.scratch_regions)
                writes.extend(BufferRegion(core_id, output.addr, output.size) for output in stage.outputs if output.owner_id == core_id and output.mem_type == MeshMemoryType.LOCAL_CACHE)
                for slot in slot_contents:
                    slot_contents[slot] = {address: key for address, key in slot_contents[slot].items() if not any(address[0] < region.addr + region.size and region.addr < address[0] + address[1] for region in writes)}
                if stage.ld_slot is not None:
                    previous_uses[stage.ld_slot] = stage_index

    @staticmethod
    @jit_prototype
    def ex_core_kernel(core: CCGTile, kernel_obj: 'MeshKernel', stage: StagePlan, stage_index: int, previous_store: int, ld_stage_var: VariableHandle, ex_stage_var: VariableHandle, st_stage_var: VariableHandle):
        core.var_conditional_wait(ld_stage_var, ld_stage_var.greater_equal(stage_index + 1))
        if previous_store >= 0:
            core.var_conditional_wait(st_stage_var, st_stage_var.greater_equal(previous_store + 1))
        if stage.kind == "COMPUTE":
            resident = kernel_obj.residency_plan.buffer_locations.get(stage.core_id, {}) if kernel_obj.residency_plan.full_resident else {}
            remote_reads = {}
            for location in stage.local_reads:
                if location.storage_key not in resident and location.owner_id != core.core_id:
                    remote_reads[location.owner_id] = remote_reads.get(location.owner_id, 0) + location.size
            for owner_id, size in remote_reads.items():
                core.icnt_recv_data(owner_id, size, sync=True)
            operations = _mesh_stage_ops(kernel_obj, stage)
            if operations:
                core.compute(operations)
            remote_writes = {}
            for task, output in zip(stage.group.tasks, stage.outputs):
                if task.output.mem_type == MeshMemoryType.LOCAL_CACHE and output.addr == task.output.addr and output.owner_id == task.output.owner_id and output.owner_id != core.core_id:
                    remote_writes[output.owner_id] = remote_writes.get(output.owner_id, 0) + output.size
            for owner_id, size in remote_writes.items():
                core.icnt_send_data([owner_id], size, sync=True)
        core.var_atomic_increase(ex_stage_var)

    @staticmethod
    @jit_program_prototype
    def ex_program(device: MeshAccelerator, kernel_obj: 'MeshKernel', stages: dict[int, list[StagePlan]], _ex_core_kernel: Callable, _stage_vars: dict[str, dict[int, VariableHandle]]):
        for core_id in kernel_obj.core_ids:
            core = device.get_ccg_tile(core_id)
            previous_stores = {}
            for stage_index, stage in enumerate(stages[core_id]):
                previous_store = previous_stores.get(stage.st_slot, -1) if stage.st_slot is not None else -1
                _ex_core_kernel(core, kernel_obj, stage, stage_index, previous_store, _stage_vars["LD"][core_id], _stage_vars["EX"][core_id], _stage_vars["ST"][core_id])
                if stage.st_slot is not None and stage.group.last_chunk:
                    previous_stores[stage.st_slot] = stage_index

    @staticmethod
    @jit_prototype
    def st_core_kernel(core: CCGTile, kernel_obj: 'MeshKernel', stage: StagePlan, stage_index: int, ex_stage_var: VariableHandle, st_stage_var: VariableHandle):
        core.var_conditional_wait(ex_stage_var, ex_stage_var.greater_equal(stage_index + 1))
        batches = {}
        remote_writes = {}
        for transfer in stage.stores:
            if transfer.tile.mem_type == MeshMemoryType.DEVICE_MEMORY:
                batches.setdefault(transfer.tile.size, []).append(transfer.tile.addr)
            elif transfer.tile.owner_id != core.core_id:
                remote_writes[transfer.tile.owner_id] = remote_writes.get(transfer.tile.owner_id, 0) + transfer.tile.size
        for owner_id, size in remote_writes.items():
            core.icnt_send_data([owner_id], size, sync=True)
        for size, addresses in batches.items():
            core.dma_write_memory_batch(addresses, size, sync=True)
        core.var_atomic_increase(st_stage_var)

    @staticmethod
    @jit_program_prototype
    def st_program(device: MeshAccelerator, kernel_obj: 'MeshKernel', stages: dict[int, list[StagePlan]], _st_core_kernel: Callable, _stage_vars: dict[str, dict[int, VariableHandle]]):
        for core_id in kernel_obj.core_ids:
            core = device.get_ccg_tile(core_id)
            for stage_index, stage in enumerate(stages[core_id]):
                _st_core_kernel(core, kernel_obj, stage, stage_index, _stage_vars["EX"][core_id], _stage_vars["ST"][core_id])

    def get_programs(self) -> dict[str, Program]:
        stages = self.create_stage_plans(self.mapping)
        stage_vars = {slot: {core_id: VariableHandle.tmp(initial_value=0) for core_id in self.core_ids} for slot in ("LD", "EX", "ST")}
        return {"LD": self.ld_program(self.device, self, stages, self.ld_core_kernel, stage_vars), "EX": self.ex_program(self.device, self, stages, self.ex_core_kernel, stage_vars), "ST": self.st_program(self.device, self, stages, self.st_core_kernel, stage_vars)}


class MeshLinearKernel(MeshKernel):
    kernel_type = MeshKernelType.LINEAR

    def __init__(self, device_rt_context: MeshAcceleratorRuntimeContext, workload: MeshDeviceCompiledWorkload, kernel_stats: MeshDeviceCompiledKernelStats):
        super().__init__(device_rt_context, workload, kernel_stats)

        output_grid, input_grids, ops = create_linear_mapping_requistes(kernel_stats.kernel_desc)
        self.mapping = create_mapping(kernel_stats.ccg_tile_mesh, output_grid, input_grids, ops)
        ifm_bank = next(iter(self.input_placements[0].tile_placement.values()))
        wgt_bank = next(iter(self.input_placements[1].tile_placement.values()))
        shared_axis = -2 if ifm_bank.mem_type == MeshMemoryType.DEVICE_MEMORY and wgt_bank.mem_type != MeshMemoryType.DEVICE_MEMORY else -1
        output_shape = kernel_stats.kernel_desc.output_tensors[0].tile_grid_shape
        output_coords = sorted(np.ndindex(*output_shape), key=lambda coord: (coord[:-2], coord[shared_axis], coord[-1 if shared_axis == -2 else -2]))
        base, remainder = divmod(len(output_coords), len(self.core_ids))
        cursor = 0
        for index, core_coords in enumerate(np.ndindex(*self.core_mesh.shape)):
            assigned = base + int(index < remainder)
            self.mapping.output_mapping[core_coords] = -1
            for slot, coord in enumerate(output_coords[cursor:cursor + assigned]):
                self.mapping.output_mapping[core_coords][slot] = coord
            cursor += assigned


class MeshConv2dKernel(MeshKernel):
    kernel_type = MeshKernelType.CONV2D

    def __init__(self, device_rt_context: MeshAcceleratorRuntimeContext, workload: MeshDeviceCompiledWorkload, kernel_stats: MeshDeviceCompiledKernelStats):
        super().__init__(device_rt_context, workload, kernel_stats)

        output_grid, input_grids, ops = create_conv2d_mapping_requistes(kernel_stats.kernel_desc)
        self.mapping = create_mapping(kernel_stats.ccg_tile_mesh, output_grid, input_grids, ops)


class MeshElementwiseKernel(MeshKernel):
    kernel_type = MeshKernelType.ELEMENTWISE

    def __init__(self, device_rt_context: MeshAcceleratorRuntimeContext, workload: MeshDeviceCompiledWorkload, kernel_stats: MeshDeviceCompiledKernelStats):
        super().__init__(device_rt_context, workload, kernel_stats)

        output_grid, input_grids, ops = create_elementwise_mapping_requistes(kernel_stats.kernel_desc)
        self.mapping = create_mapping(kernel_stats.ccg_tile_mesh, output_grid, input_grids, ops)


class MeshReductionKernel(MeshKernel):
    kernel_type = MeshKernelType.REDUCTION

    def __init__(self, device_rt_context: MeshAcceleratorRuntimeContext, workload: MeshDeviceCompiledWorkload, kernel_stats: MeshDeviceCompiledKernelStats):
        super().__init__(device_rt_context, workload, kernel_stats)

        output_grid, input_grids, ops = create_reduction_mapping_requistes(kernel_stats.kernel_desc)
        self.mapping = create_mapping(kernel_stats.ccg_tile_mesh, output_grid, input_grids, ops)


class MeshSDPAKernel(MeshKernel):
    kernel_type = MeshKernelType.SDPA

    def __init__(self, device_rt_context: MeshAcceleratorRuntimeContext, workload: MeshDeviceCompiledWorkload, kernel_stats: MeshDeviceCompiledKernelStats):
        super().__init__(device_rt_context, workload, kernel_stats)
        ofm = kernel_stats.kernel_desc.output_tensors[0]
        self.scratch_bytes_per_output = {coord: 8 * min(ofm.tile_shape[2], ofm.shape[2] - coord[2] * ofm.tile_shape[2]) for coord in np.ndindex(*ofm.tile_grid_shape)}
        output_grid, input_grids, ops = create_sdpa_mapping_requistes(kernel_stats.kernel_desc)
        self.mapping = create_mapping(kernel_stats.ccg_tile_mesh, output_grid, input_grids, ops)
        output_shape = ofm.tile_grid_shape
        pair_count = (output_shape[3] + 1) // 2
        if output_shape[3] >= 8 and output_shape[2] * pair_count <= kernel_stats.kernel_desc.get_required_kwargs("max_cores_per_head"):
            output_pairs = [tuple((batch, head, query, dim) for dim in range(start, min(start + 2, output_shape[3]))) for batch, head, query in np.ndindex(*output_shape[:3]) for start in range(0, output_shape[3], 2)]
            assignments = [[] for _ in self.core_ids]
            for index, pair in enumerate(output_pairs):
                assignments[index % len(assignments)].extend(pair)
            max_tiles = max(map(len, assignments))
            self.mapping.output_mapping = np.full((*self.core_mesh.shape, max_tiles, len(output_shape)), -1, dtype=np.int64)
            for index, core_coords in enumerate(np.ndindex(*self.core_mesh.shape)):
                for slot, coord in enumerate(assignments[index]):
                    self.mapping.output_mapping[core_coords][slot] = coord


class MeshMemCopyKernel(MeshKernel):
    kernel_type = MeshKernelType.MEMCOPY

    def __init__(self, device_rt_context: MeshAcceleratorRuntimeContext, workload: MeshDeviceCompiledWorkload, kernel_stats: MeshDeviceCompiledKernelStats):
        super().__init__(device_rt_context, workload, kernel_stats)
        descriptor = kernel_stats.kernel_desc
        metadata_count = descriptor.get_required_kwargs("metadata_input_count")
        has_source = len(descriptor.input_tensors) > metadata_count
        source = descriptor.input_tensors[0] if has_source else None
        destination = descriptor.output_tensors[0] if descriptor.output_tensors else None
        if len(descriptor.output_tensors) > 1 or len(descriptor.input_tensors) != metadata_count + int(has_source):
            raise ValueError("MemCopy requires at most one source and one destination tensor.")
        source_coords = list(np.ndindex(*source.tile_grid_shape)) if source is not None else []
        destination_coords = list(np.ndindex(*destination.tile_grid_shape)) if destination is not None else []
        gather = descriptor.get_required_kwargs("gather")
        count = len(destination_coords) if gather else max(len(source_coords), len(destination_coords))
        if count == 0:
            raise ValueError("MemCopy requires a source or destination tile.")
        source_bytes = descriptor.get_required_kwargs("src_traffic_bytes")
        destination_bytes = descriptor.get_required_kwargs("dst_traffic_bytes")
        source_sizes = [min(source.get_tile_size(), max(0, source_bytes - index * source.get_tile_size())) for index in range(count)] if source is not None else []
        destination_sizes = [min(destination.get_tile_size(), max(0, destination_bytes - index * destination.get_tile_size())) for index in range(count)] if destination is not None else []
        if source_sizes and sum(source_sizes) < source_bytes:
            source_sizes[-1] += source_bytes - sum(source_sizes)
        if destination_sizes and sum(destination_sizes) < destination_bytes:
            destination_sizes[-1] += destination_bytes - sum(destination_sizes)
        self.copy_stages = {core_id: {"source_dma": [], "source_cache": [], "destination_dma": [], "destination_cache": []} for core_id in self.core_ids}
        for index in range(count):
            source_coord = source_coords[(index * 1315423911) % len(source_coords)] if gather else source_coords[min(len(source_coords) - 1, index * len(source_coords) // count)] if source_coords else None
            destination_coord = destination_coords[min(len(destination_coords) - 1, index * len(destination_coords) // count)] if destination_coords else None
            source_bank = self.input_placements[0].tile_placement[source_coord] if source_coord is not None else None
            destination_bank = self.output_placements[0].tile_placement[destination_coord] if destination_coord is not None else None
            local_bank = next((bank for bank in (destination_bank, source_bank) if bank is not None and bank.mem_type == MeshMemoryType.LOCAL_CACHE and bank.owner_id in self.copy_stages), None)
            core_id = int(local_bank.owner_id) if local_bank is not None else self.core_ids[index % len(self.core_ids)]
            stage = self.copy_stages[core_id]
            if source_bank is not None and source_sizes[index]:
                if source_bank.mem_type == MeshMemoryType.DEVICE_MEMORY:
                    stage["source_dma"].append((source_bank.addr, source_sizes[index]))
                elif source_bank.owner_id != core_id:
                    stage["source_cache"].append((source_bank.owner_id, source_sizes[index]))
            if destination_bank is not None and destination_sizes[index]:
                if destination_bank.mem_type == MeshMemoryType.DEVICE_MEMORY:
                    stage["destination_dma"].append((destination_bank.addr, destination_sizes[index]))
                elif destination_bank.owner_id != core_id:
                    stage["destination_cache"].append((destination_bank.owner_id, destination_sizes[index]))
        metadata_traffic = descriptor.get_required_kwargs("metadata_traffic_bytes")
        for metadata_index, placement in enumerate(self.input_placements[int(has_source):]):
            core_id = self.core_ids[metadata_index % len(self.core_ids)]
            remaining = metadata_traffic[metadata_index]
            for coord in np.ndindex(*placement.tensor_desc.tile_grid_shape):
                size = min(placement.tensor_desc.get_tile_size(), remaining)
                remaining -= size
                if size <= 0:
                    break
                bank = placement.tile_placement[coord]
                if bank.mem_type == MeshMemoryType.DEVICE_MEMORY:
                    self.copy_stages[core_id]["source_dma"].append((bank.addr, size))
                elif bank.owner_id != core_id:
                    self.copy_stages[core_id]["source_cache"].append((bank.owner_id, size))

    @staticmethod
    @jit_prototype
    def _local_read(core: CCGTile, owner_id: int, size: int):
        core.icnt_recv_data(owner_id, size, sync=True)

    @staticmethod
    @jit_prototype
    def _local_send(core: CCGTile, owner_id: int, size: int):
        core.icnt_send_data([owner_id], size, sync=True)

    @staticmethod
    @jit_program_prototype
    def program(device: MeshAccelerator, core_ids: tuple[int, ...], stages: dict[int, dict]):
        for core_id in core_ids:
            core = device.get_ccg_tile(core_id)
            stage = stages[core_id]
            batches = {}
            for address, size in stage["source_dma"]:
                batches.setdefault(size, []).append(address)
            for size, addresses in batches.items():
                core.dma_read_memory_batch(addresses, size, sync=True)
            remote_reads = {}
            for owner_id, size in stage["source_cache"]:
                remote_reads[owner_id] = remote_reads.get(owner_id, 0) + size
            for owner_id, size in remote_reads.items():
                MeshMemCopyKernel._local_read(core, owner_id, size)
            remote_writes = {}
            for owner_id, size in stage["destination_cache"]:
                remote_writes[owner_id] = remote_writes.get(owner_id, 0) + size
            for owner_id, size in remote_writes.items():
                MeshMemCopyKernel._local_send(core, owner_id, size)
            batches = {}
            for address, size in stage["destination_dma"]:
                batches.setdefault(size, []).append(address)
            for size, addresses in batches.items():
                core.dma_write_memory_batch(addresses, size, sync=True)

    def get_programs(self) -> dict[str, Program]:
        return {"EX": self.program(self.device, self.core_ids, self.copy_stages)}
