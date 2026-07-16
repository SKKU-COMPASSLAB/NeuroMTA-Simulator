import os
import sys
import shutil
import time
from typing import Sequence, Callable  #, Any

from neuromta.framework.core import Core, Kernel, Command, RPCMessage, ThreadGroup
from neuromta.framework.companion import CompanionCore
from neuromta.framework.simulation_mode import SimulationMode, normalize_simulation_mode, set_global_simulation_mode, get_global_simulation_mode
from neuromta.framework.logger import logger
from neuromta.framework.debug_utils import *


__all__ = [
    "Device",
]


class Device:
    def __init__(self):
        self._cores: dict[str, Core] = None
        self._verbose_hook_ids: dict[str, str] = {}

        self._rpc_req_send_inbox: dict[str, list[RPCMessage]] = {}
        self._rpc_rsp_send_inbox: dict[str, list[RPCMessage]] = {}

        self._companion_core = CompanionCore()
        self._simulation_mode: SimulationMode = get_global_simulation_mode()

        self._timestamp = 0

        self._active_core_ids: set[int | str] = set()
        self._blocked_core_ids: set[int | str] = set()
        self._idle_core_ids: set[int | str] = set()
        self._rpc_pending_core_ids: set[int | str] = set()

        self._sim_profile_enabled = False
        self._sim_profile: dict[str, int | float] = {
            "step_count": 0,
            "rpc_update_time": 0.0,
            "core_update_time": 0.0,
            "companion_update_time": 0.0,
            "state_refresh_time": 0.0,
            "deadlock_check_time": 0.0,
            "active_core_update_count": 0,
            "rpc_pending_core_update_count": 0,
            "blocked_core_skip_count": 0,
            "idle_core_skip_count": 0,
            "deadlock_fast_path_count": 0,
            "deadlock_fallback_count": 0,
        }

    def get_core_from_id(self, core_id: int) -> Core:
        return self._cores[core_id]

    @property
    def companion_core(self) -> CompanionCore:
        return self._companion_core

    def change_sim_model_options(self, use_cycle_model: bool = None, use_functional_model: bool = None):
        if use_functional_model is not None:
            self._simulation_mode = SimulationMode.CORRECTNESS if use_functional_model else SimulationMode.PERFORMANCE
            set_global_simulation_mode(self._simulation_mode)
        if self._cores is None:
            return self
        for core in self._cores.values():
            core.change_sim_model_options(use_cycle_model, use_functional_model)
        return self

    def set_simulation_mode(self, mode: SimulationMode | str | bool):
        self._simulation_mode = normalize_simulation_mode(mode)
        set_global_simulation_mode(self._simulation_mode)
        if self._cores is not None:
            for core in self._cores.values():
                core.set_simulation_mode(self._simulation_mode)
        return self

    def _register_core(self, name: str, core: Core | Sequence[Core]):
        if isinstance(core, Sequence):
            for idx, item in enumerate(core):
                if isinstance(item, (Core, Sequence)):
                    self._register_core(f"{name}[{idx}]", item)
        elif isinstance(core, Core):
            if core.core_id in self._cores.keys():
                raise Exception(f"Core with ID '{core.core_id}' already exists. Please use a unique core ID.")
            self._cores[core.core_id] = core

    def initialize(self):
        self._cores = {}

        for name, core in self.__dict__.items():
            if name == "_cores":
                continue    # skip internal members

            if isinstance(core, (Core, Sequence)):
                self._register_core(name, core)

        self._rpc_req_send_inbox = {core.core_id: [] for core in self._cores.values()}
        self._rpc_rsp_send_inbox = {core.core_id: [] for core in self._cores.values()}

        for core in self._cores.values():
            core.set_scheduler_callback(self._on_core_scheduler_event)
            core.set_simulation_mode(self._simulation_mode)
            core.initialize_kernel_dispatch_queue()
            core.initialize_mp_queue_inbox(rpc_req_send_inbox=self._rpc_req_send_inbox, rpc_rsp_send_inbox=self._rpc_rsp_send_inbox)

        self._rebuild_scheduler_state()

        return self

    def reset_simulation(self):
        if not self.is_initialized:
            raise Exception("[ERROR] Device is not initialized. Please call initialize() before using this method.")

        self._timestamp = 0
        for core in self._cores.values():
            core.reset_simulation()
        self._rebuild_scheduler_state()

    def set_simulation_profiler(self, enabled: bool=True):
        self._sim_profile_enabled = enabled
        if enabled:
            self.reset_simulation_profile()
        return self

    def reset_simulation_profile(self):
        for key in self._sim_profile.keys():
            self._sim_profile[key] = 0.0 if key.endswith("_time") else 0
        return self

    def get_simulation_profile(self) -> dict[str, int | float]:
        return dict(self._sim_profile)

    def _profile_add_time(self, key: str, start_time: float):
        if self._sim_profile_enabled:
            self._sim_profile[key] += time.perf_counter() - start_time

    def _profile_add_count(self, key: str, value: int=1):
        if self._sim_profile_enabled:
            self._sim_profile[key] += value

    def _sync_core_timestamp(self, core: Core):
        if core._timestamp < self._timestamp:
            core._timestamp = self._timestamp

    def _clear_core_scheduler_state(self, core_id: int | str):
        self._active_core_ids.discard(core_id)
        self._blocked_core_ids.discard(core_id)
        self._idle_core_ids.discard(core_id)
        self._rpc_pending_core_ids.discard(core_id)

    def _refresh_core_scheduler_state(self, core_id: int | str):
        if not self.is_initialized or core_id not in self._cores:
            return

        core = self._cores[core_id]
        self._clear_core_scheduler_state(core_id)

        has_pending_rpc = core.has_pending_rpc
        if has_pending_rpc:
            self._rpc_pending_core_ids.add(core_id)

        if core.is_idle:
            if has_pending_rpc:
                self._active_core_ids.add(core_id)
            else:
                self._idle_core_ids.add(core_id)
            return

        if has_pending_rpc or not core.check_all_blocked():
            self._active_core_ids.add(core_id)
        else:
            self._blocked_core_ids.add(core_id)

    def _rebuild_scheduler_state(self):
        self._active_core_ids.clear()
        self._blocked_core_ids.clear()
        self._idle_core_ids.clear()
        self._rpc_pending_core_ids.clear()

        if not self.is_initialized:
            return

        for core_id in self._cores.keys():
            self._refresh_core_scheduler_state(core_id)

    def _on_core_scheduler_event(self, core: Core, event: str, payload=None):
        if not self.is_initialized:
            return

        self._sync_core_timestamp(core)

        if event in ("rpc_request_sent", "rpc_response_sent") and payload in self._cores:
            target_core = self._cores[payload]
            self._sync_core_timestamp(target_core)
            self._rpc_pending_core_ids.add(payload)
            self._active_core_ids.add(payload)
            self._idle_core_ids.discard(payload)
            self._blocked_core_ids.discard(payload)

        if event == "kernel_unblocked":
            self._active_core_ids.add(core.core_id)
            self._blocked_core_ids.discard(core.core_id)
            self._idle_core_ids.discard(core.core_id)
            return

        if event in ("main_dispatch", "rpc_dispatch"):
            self._active_core_ids.add(core.core_id)
            self._blocked_core_ids.discard(core.core_id)
            self._idle_core_ids.discard(core.core_id)
            return

        if event == "kernel_blocked":
            self._refresh_core_scheduler_state(core.core_id)
            return

        if event in (
            "kernel_finish",
            "reset",
            "state_may_change",
            "rpc_request_queue_drained",
            "rpc_response_queue_drained",
        ):
            self._refresh_core_scheduler_state(core.core_id)
            return

    def _is_sync_group_idle(self, sync_group: Sequence[int | str]) -> bool:
        return all(core_id in self._idle_core_ids for core_id in sync_group)

    def _has_sync_group_active_core(self, core_ids: Sequence[int | str]) -> bool:
        target_core_ids = set(core_ids)
        if target_core_ids & self._active_core_ids:
            return True
        if target_core_ids & self._rpc_pending_core_ids:
            return True
        return False

    def _check_deadlock_from_scheduler_state(self, core_ids: Sequence[int | str]) -> bool | None:
        target_core_ids = set(core_ids)

        if target_core_ids & self._active_core_ids:
            return False
        if target_core_ids & self._rpc_pending_core_ids:
            return False

        has_blocked_core = False
        for core_id in target_core_ids:
            if core_id not in self._cores:
                return None

            if core_id == self.companion_core.core_id and not self.companion_core.check_idle():
                return False

            if core_id in self._idle_core_ids:
                continue
            if core_id in self._blocked_core_ids:
                has_blocked_core = True
                continue

            return None

        return has_blocked_core

    def set_command_debug_verbosity(self, verbose: bool=True):
        if verbose:
            for core in self._cores.values():
                if core.core_id in self._verbose_hook_ids:
                    continue    # already registered

                hook_id = core.register_command_debug_hook(self._verbose_command_debug_hook)
                self._verbose_hook_ids[core.core_id] = hook_id
        else:
            for core in self._cores.values():
                if core.core_id not in self._verbose_hook_ids:
                    continue    # not registered

                hook_id = self._verbose_hook_ids[core.core_id]
                core.unregister_command_debug_hook(hook_id)
                del self._verbose_hook_ids[core.core_id]

    def _verbose_command_debug_hook(self, core: Core, kernel: Kernel, cmd: Command, issue_time: int, commit_time: int):
        callstack = kernel.callstack if kernel else "N/A"
        if len(callstack) > 100:
            callstack = callstack[:47] + " ... " + callstack[-47:]
        logger.debug(f"{issue_time:<6d} - {commit_time:<6d} | {core.core_id.__str__():<10s} | {callstack:<100s} | command: {cmd.cmd_id}")

    def run_single_step(self, cycle_resolution: int = 1, event_driven_mode: bool = False):
        if not self.is_initialized:
            raise Exception("[ERROR] Device is not initialized. Please call initialize() before using this method.")

        remaining_cycles = None
        self._profile_add_count("step_count")

        rpc_update_core_ids = set(self._rpc_pending_core_ids)
        active_update_core_ids = set(self._active_core_ids)

        t = time.perf_counter() if self._sim_profile_enabled else None
        for core_id in rpc_update_core_ids:
            core = self.initialized_cores[core_id]
            self._sync_core_timestamp(core)
            core.rpc_update_routine()
        if t is not None:
            self._profile_add_time("rpc_update_time", t)
            self._profile_add_count("rpc_pending_core_update_count", len(rpc_update_core_ids))

        if event_driven_mode:
            for core_id, core in self.initialized_cores.items():
                if core.is_idle:
                    continue
                c = core.get_remaining_cycles()
                
                if remaining_cycles is None:
                    remaining_cycles = c
                elif c is not None:
                    remaining_cycles = min(remaining_cycles, c)

        # # for core_id, core in self.initialized_cores.items():
        # #     if core.is_idle:
        # #         continue
        # #     c = core.get_remaining_cycles()

        # #     if remaining_cycles is None:
        # #         remaining_cycles = c
        # #     elif c is not None:
        # #         remaining_cycles = min(remaining_cycles, c)

        # # if remaining_cycles == 0 or remaining_cycles is None:
        # #     remaining_cycles = self.companion_core.update_cycle_time_until_cmd_executed()

        # #     if remaining_cycles == 0 or remaining_cycles is None:
        # #         remaining_cycles = cycle_resolution
        # # else:
        # #     self.companion_core.update_cycle_time_companion_modules(cycle_time=remaining_cycles)

        if remaining_cycles is None or remaining_cycles <= 0:
            remaining_cycles = cycle_resolution

        # # remaining_cycles = 1

        t = time.perf_counter() if self._sim_profile_enabled else None
        self._sync_core_timestamp(self.companion_core)
        self.companion_core.update_cycle_time_companion_modules(cycle_time=remaining_cycles)
        if t is not None:
            self._profile_add_time("companion_update_time", t)

        update_core_ids = active_update_core_ids | set(self._active_core_ids)

        t = time.perf_counter() if self._sim_profile_enabled else None
        for core_id in update_core_ids:
            # with print_log_execution_time(desc=f"  - CORE {core_id:<3d}" if core.core_id != self.companion_core.core_id else f"  - COMPANION", disable=core.core_id != 24):
            core = self.initialized_cores[core_id]
            self._sync_core_timestamp(core)
            if not core.is_idle:
                core.update_cycle_time(cycle_time=remaining_cycles)
        if t is not None:
            self._profile_add_time("core_update_time", t)
            self._profile_add_count("active_core_update_count", len(update_core_ids))

        self._timestamp += remaining_cycles

        t = time.perf_counter() if self._sim_profile_enabled else None
        for core_id in rpc_update_core_ids | update_core_ids | {self.companion_core.core_id}:
            self._refresh_core_scheduler_state(core_id)
        if t is not None:
            self._profile_add_time("state_refresh_time", t)
            self._profile_add_count("blocked_core_skip_count", len(self._blocked_core_ids))
            self._profile_add_count("idle_core_skip_count", len(self._idle_core_ids))

    def run_kernels(
        self,
        cycle_resolution:   int  = 1,   # the number of cycles to update when all the cores are waiting and returning (0 | None) as the minimum remaining cycles
        max_steps:          int  = -1,  # the maximum number of steps to run
        max_timestamp:      int  = None,  # the maximum timestamp to run
        event_driven_mode:  bool = False,  # whether to use event-driven mode (True) or fixed cycle resolution mode (False)

        sync_target_core_groups: Sequence[Sequence[int]] = None,  # the target core groups to synchronize after each step; if None, all cores are synchronized
    ):
        if max_timestamp is None:
            max_timestamp = -1

        if not self.is_initialized:
            raise Exception("[ERROR] Device is not initialized. Please call initialize() before using this method.")

        if sync_target_core_groups is None:
            core_ids = list(self._cores.keys())
            sync_target_core_groups = [core_ids]
        else:
            core_ids = []
            for group in sync_target_core_groups:
                core_ids.extend(group)

        step_cnt = 0
        deadlock_cnt = 0

        # while not all(self.initialized_cores[core_id].is_idle for core_id in core_ids):
        while True:
            # with print_log_execution_time(desc=f"STEP {step_cnt:<4d}"):
            self.run_single_step(cycle_resolution=cycle_resolution, event_driven_mode=event_driven_mode)

            # with print_log_execution_time(desc=f"POST {step_cnt:<4d}"):
            # break condition: step count
            step_cnt += 1
            if step_cnt >= max_steps > 0:
                logger.debug(f"Reached maximum steps: {max_steps}. Stopping simulation.")
                self.debug_current_execution_state(*core_ids)
                break

            # break condition: timestamp
            if self.timestamp >= max_timestamp > 0:
                logger.debug(f"Reached maximum timestamp: {max_timestamp}. Stopping simulation.")
                self.debug_current_execution_state(*core_ids)
                break

            # break condition: idle synchronization targets
            is_idle = False
            for sync_group in sync_target_core_groups:
                is_idle = self._is_sync_group_idle(sync_group)
                if not is_idle:
                    break
            if is_idle:
                logger.debug(f"Synchronization target cores are idle. Stopping simulation.")
                break

            # break condition: deadlock
            if self._has_sync_group_active_core(core_ids):
                deadlock_cnt = 0
            elif self.check_deadlock(*core_ids):
                deadlock_cnt += 1
                if deadlock_cnt >= 10:  # threshold for deadlock detection
                    logger.error(f"Deadlock detected among cores {core_ids}. Stopping simulation.")
                    self.debug_current_execution_state(*core_ids)
                    break
            else:
                deadlock_cnt = 0

    def register_command_debug_hook(self, hook: Callable):
        if not self.is_initialized:
            raise Exception("[ERROR] Device is not initialized. Please call initialize() before using this method.")

        for core in self._cores.values():
            core.register_command_debug_hook(hook)

    def register_kernel_debug_hook(self, hook: Callable, slot_id: str, core_ids: list[int]=None):
        if not self.is_initialized:
            raise Exception("[ERROR] Device is not initialized. Please call initialize() before using this method.")

        if isinstance(core_ids, (int, str)):
            core_ids = [core_ids]

        if core_ids is None:
            core_ids = list(self._cores.keys())

        for core_id in core_ids:
            core = self._cores[core_id]
            core.register_kernel_debug_hook(hook, slot_id)

    def check_deadlock(self, *core_ids: int | str) -> bool:
        if not self.is_initialized:
            raise Exception("[ERROR] Device is not initialized. Please call initialize() before using this method.")

        t = time.perf_counter() if self._sim_profile_enabled else None
        scheduler_result = self._check_deadlock_from_scheduler_state(core_ids)
        if scheduler_result is not None:
            self._profile_add_count("deadlock_fast_path_count")
            if t is not None:
                self._profile_add_time("deadlock_check_time", t)
            return scheduler_result

        self._profile_add_count("deadlock_fallback_count")
        is_not_deadlock = False

        for core_id in core_ids:
            core = self._cores[core_id]
            if not core.check_all_blocked():
                is_not_deadlock = True
                break
            if core.core_id == self.companion_core.core_id and not self.companion_core.check_idle():
                is_not_deadlock = True
                break

        result = not is_not_deadlock
        if t is not None:
            self._profile_add_time("deadlock_check_time", t)
        return result

    def debug_current_execution_state(self, *core_ids: int | str):
        for core_id in core_ids:
            core = self.initialized_cores[core_id]
            if len(core._dispatched_main_kernels) == 0:
                continue
            logger.debug(f"Core '{core_id}'")
            for slot_id, kernel in core._dispatched_main_kernels.items():
                if kernel.is_finished(core):
                    continue

                logger.debug(f"  KERNEL {kernel.callstack}")
                step = kernel.current_step(core)

                def print_step_info(step):
                    if isinstance(step, Kernel):
                        if step.is_finished(core):
                            return
                        logger.debug(f"    calls KERNEL {step.callstack}")
                        print_step_info(step.current_step(core))
                    elif isinstance(step, ThreadGroup):
                        for sub_step in step:
                            if not sub_step.is_finished(core):
                                logger.debug(f"    calls THREAD {sub_step.callstack}" + (" (blocked)" if sub_step.is_blocked else ""))
                                print_step_info(sub_step.current_step(core))
                    else:
                        logger.debug(f"     -> COMMAND {step}")

                print_step_info(step)

    @property
    def timestamp(self) -> int:
        # t = [core.timestamp for core in self._cores.values()]
        # return max(t) if t else 0
        return self._timestamp

    @property
    def simulation_mode(self) -> SimulationMode:
        return self._simulation_mode

    @property
    def is_performance_mode(self) -> bool:
        return self._simulation_mode == SimulationMode.PERFORMANCE

    @property
    def is_initialized(self) -> bool:
        return self._cores is not None

    @property
    def is_idle(self) -> bool:
        if not self.is_initialized:
            return True
        return len(self._idle_core_ids) == len(self._cores)

    @property
    def initialized_cores(self) -> dict[str, Core]:
        return self._cores
