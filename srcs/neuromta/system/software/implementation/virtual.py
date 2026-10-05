from neuromta.system.software.utils.compiler import MeshDeviceCompiler
from neuromta.system.software.utils.scheduler import MeshDeviceScheduler, MeshSchedulingDomain
from neuromta.system.software.utils.runtime import MeshDeviceRuntime, MeshDeviceRuntimeWorkloadState


__all__ = ["VirtualCompiler", "VirtualScheduler", "VirtualRuntime"]


class VirtualCompiler(MeshDeviceCompiler):
    pass


class VirtualScheduler(MeshDeviceScheduler):
    pass


class VirtualRuntime(MeshDeviceRuntime):
    def __init__(self, device_desc, instances=None, kernel_materializer=None, scheduler=None, enable_debug_log=False, require_full_coverage=True, exclusive_dma=True):
        super().__init__(device_desc, kernel_materializer=kernel_materializer, scheduler=VirtualScheduler() if scheduler is None and kernel_materializer is None else scheduler, enable_debug_log=enable_debug_log)
        domains = (self._device_domain,) if instances is None else tuple(instances)
        if not domains or any(not isinstance(domain, MeshSchedulingDomain) for domain in domains):
            raise TypeError("instances must contain MeshSchedulingDomain objects.")
        if len({domain.domain_id for domain in domains}) != len(domains):
            raise ValueError("Virtual instance domain IDs must be unique.")
        device_ccg_ids = set(self._device_rt_context.ccg_tile_ids)
        device_dma_ids = set(self._device_rt_context.dma_tile_ids)
        assigned_ccg_ids = set()
        assigned_dma_ids = set()
        device_mesh = self._device.get_ccg_tile_mesh()
        for domain in domains:
            ccg_ids = set(domain.ccg_tile_mesh.flatten().tolist())
            dma_ids = set(domain.dma_ids)
            if not ccg_ids.issubset(device_ccg_ids) or not dma_ids.issubset(device_dma_ids):
                raise ValueError(f"Virtual domain '{domain.domain_id}' contains resources outside the device.")
            positions = [tuple(position) for core_id in ccg_ids for position in zip(*((device_mesh == core_id).nonzero()))]
            rows = sorted({position[0] for position in positions})
            columns = sorted({position[1] for position in positions})
            expected = device_mesh[rows[0]:rows[-1] + 1, columns[0]:columns[-1] + 1]
            if expected.shape != domain.ccg_tile_mesh.shape or not (expected == domain.ccg_tile_mesh).all():
                raise ValueError(f"Virtual domain '{domain.domain_id}' must be a contiguous rectangular submesh.")
            if assigned_ccg_ids & ccg_ids:
                raise ValueError("Virtual domains must not overlap in CCG resources.")
            if exclusive_dma and assigned_dma_ids & dma_ids:
                raise ValueError("Virtual domains must not overlap in DMA resources when exclusive_dma is enabled.")
            assigned_ccg_ids.update(ccg_ids)
            assigned_dma_ids.update(dma_ids)
        if require_full_coverage and assigned_ccg_ids != device_ccg_ids:
            raise ValueError("Virtual domains must cover every device CCG when require_full_coverage is enabled.")
        self._virtual_domains = {domain.domain_id: domain for domain in domains}
        self._domain_workloads = {domain.domain_id: [] for domain in domains}
        self._active_workload_by_domain = {domain.domain_id: None for domain in domains}

    def _reset_policy_state(self) -> None:
        self._virtual_domains = {}
        self._domain_workloads = {}
        self._active_workload_by_domain = {}

    def submit(self, compiled_workload, arrival_cycle=0, workload_id=None, dependent_workload_ids=(), scheduling_hint=None, warmup=False, domain_id=None):
        if domain_id is None:
            domain_id = min(self._virtual_domains, key=lambda target: (len(self._domain_workloads[target]), target))
        if domain_id not in self._virtual_domains:
            raise ValueError(f"Unknown virtual domain '{domain_id}'.")
        submitted_id = super().submit(compiled_workload, arrival_cycle=arrival_cycle, workload_id=workload_id, dependent_workload_ids=dependent_workload_ids, scheduling_hint=scheduling_hint, warmup=warmup, domain_id=domain_id)
        self._domain_workloads[domain_id].append(submitted_id)
        return submitted_id

    def _select_dispatchable_workloads(self, ready_states: tuple[MeshDeviceRuntimeWorkloadState, ...]) -> tuple[MeshDeviceRuntimeWorkloadState, ...]:
        ready_by_id = {state.workload_id: state for state in ready_states}
        selected = []
        for domain_id in self._virtual_domains:
            active_id = self._active_workload_by_domain[domain_id]
            if active_id in ready_by_id:
                selected.append(ready_by_id[active_id])
                continue
            if active_id is not None:
                continue
            next_state = next((ready_by_id[workload_id] for workload_id in self._domain_workloads[domain_id] if workload_id in ready_by_id), None)
            if next_state is not None:
                self._active_workload_by_domain[domain_id] = next_state.workload_id
                selected.append(next_state)
        return tuple(selected)

    def _get_scheduling_domain(self, workload_state: MeshDeviceRuntimeWorkloadState) -> MeshSchedulingDomain:
        try:
            return self._virtual_domains[workload_state.domain_id]
        except KeyError as error:
            raise ValueError(f"Unknown virtual domain '{workload_state.domain_id}'.") from error

    def _on_workload_completed(self, workload_state: MeshDeviceRuntimeWorkloadState) -> None:
        domain_id = workload_state.domain_id
        if self._active_workload_by_domain.get(domain_id) == workload_state.workload_id:
            self._active_workload_by_domain[domain_id] = None
        if workload_state.workload_id in self._domain_workloads.get(domain_id, ()):
            self._domain_workloads[domain_id].remove(workload_state.workload_id)
