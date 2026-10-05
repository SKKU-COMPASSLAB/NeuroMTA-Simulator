import os

from neuromta.framework.logger import logger

if __package__:
    from .common import DEFAULT_CCG_TOPS, DEFAULT_DURATION_CYCLES, DEFAULT_WARMUP_CYCLES, create_scheduler, run_experiment
else:
    from common import DEFAULT_CCG_TOPS, DEFAULT_DURATION_CYCLES, DEFAULT_WARMUP_CYCLES, create_scheduler, run_experiment

from neuromta.system.software.implementation.preemptive import PreemptiveCompiler, PreemptiveRuntime


def run(duration_cycles: int = DEFAULT_DURATION_CYCLES, warmup_cycles: int = DEFAULT_WARMUP_CYCLES, ccg_tops: float = DEFAULT_CCG_TOPS, enable_debug_log: bool = False) -> None:
    completed_jobs, completion_cycle, simulation_time, profile_dir = run_experiment("run_preemptive", PreemptiveCompiler, lambda device, debug: PreemptiveRuntime(device, scheduler=create_scheduler(), enable_debug_log=debug), duration_cycles, warmup_cycles, ccg_tops, enable_debug_log)
    logger.info(f"Completed {completed_jobs} kernels at cycle {completion_cycle}; simulation time: {simulation_time:.6f} s")
    logger.info(f"Kernel profiles: {os.path.join(profile_dir, 'kernel_profile')}")
    logger.info(f"Workload profile: {os.path.join(profile_dir, 'workload_profile.csv')}")
    logger.info(f"Scheduler profile: {os.path.join(profile_dir, 'scheduler_profile.csv')}")
    logger.info(f"Metadata: {os.path.join(profile_dir, 'metadata.json')}")


if __name__ == "__main__":
    run()
