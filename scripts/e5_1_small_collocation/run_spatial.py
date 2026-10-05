import os

from neuromta.framework.logger import logger

if __package__:
    from .common import create_scheduler, run_experiment
else:
    from common import create_scheduler, run_experiment

from neuromta.system.software.implementation.spatial import SpatialCompiler, SpatialRuntime


def run(duration_cycles: int = 4_000_000_000, warmup_cycles: int = 1_000_000_000, ccg_tops: float = 0.05, enable_debug_log: bool = False) -> None:
    completed_jobs, completion_cycle, simulation_time, profile_dir = run_experiment("run_spatial", SpatialCompiler, lambda device, debug: SpatialRuntime(device, scheduler=create_scheduler(), enable_debug_log=debug), duration_cycles, warmup_cycles, ccg_tops, enable_debug_log)
    logger.info(f"Completed {completed_jobs} kernels at cycle {completion_cycle}; simulation time: {simulation_time:.6f} s")
    logger.info(f"Kernel profiles: {os.path.join(profile_dir, 'kernel_profile')}")
    logger.info(f"Workload profile: {os.path.join(profile_dir, 'workload_profile.csv')}")
    logger.info(f"Scheduler profile: {os.path.join(profile_dir, 'scheduler_profile.csv')}")
    logger.info(f"Metadata: {os.path.join(profile_dir, 'metadata.json')}")


if __name__ == "__main__":
    run()
