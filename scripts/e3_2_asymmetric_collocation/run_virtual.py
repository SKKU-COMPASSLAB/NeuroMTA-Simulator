import os

from neuromta.framework.logger import logger

if __package__:
    from .common import BENCHMARK_IDS, DEFAULT_CCG_TOPS, DEFAULT_DURATION_CYCLES, DEFAULT_WARMUP_CYCLES, create_scheduler, create_virtual_domains, run_experiment
else:
    from common import BENCHMARK_IDS, DEFAULT_CCG_TOPS, DEFAULT_DURATION_CYCLES, DEFAULT_WARMUP_CYCLES, create_scheduler, create_virtual_domains, run_experiment

from neuromta.system.software.implementation.virtual import VirtualCompiler, VirtualRuntime

DOMAIN_IDS = {benchmark_id: benchmark_id for benchmark_id in BENCHMARK_IDS}


def run(duration_cycles: int = DEFAULT_DURATION_CYCLES, warmup_cycles: int = DEFAULT_WARMUP_CYCLES, ccg_tops: float = DEFAULT_CCG_TOPS, enable_debug_log: bool = False) -> None:
    completed_jobs, completion_cycle, simulation_time, profile_dir = run_experiment("run_virtual", VirtualCompiler, lambda device, debug: VirtualRuntime(device, instances=create_virtual_domains(device), scheduler=create_scheduler(), enable_debug_log=debug, require_full_coverage=True, exclusive_dma=True), duration_cycles, warmup_cycles, ccg_tops, enable_debug_log, DOMAIN_IDS)
    logger.info(f"Completed {completed_jobs} kernels at cycle {completion_cycle}; simulation time: {simulation_time:.6f} s")
    logger.info(f"Kernel profiles: {os.path.join(profile_dir, 'kernel_profile')}")
    logger.info(f"Workload profile: {os.path.join(profile_dir, 'workload_profile.csv')}")
    logger.info(f"Scheduler profile: {os.path.join(profile_dir, 'scheduler_profile.csv')}")
    logger.info(f"Metadata: {os.path.join(profile_dir, 'metadata.json')}")


if __name__ == "__main__":
    run()
