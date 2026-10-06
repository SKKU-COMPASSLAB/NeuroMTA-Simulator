import os

from neuromta.framework.logger import logger

if __package__:
    from .common import CAMERAS, DEFAULT_CCG_TOPS, create_scheduler, create_virtual_domains, run_experiment
else:
    from common import CAMERAS, DEFAULT_CCG_TOPS, create_scheduler, create_virtual_domains, run_experiment

from neuromta.system.software.implementation.virtual import VirtualCompiler, VirtualRuntime

DOMAIN_IDS = {f"camera.det.{camera}.1": f"vision.{camera}" for camera in CAMERAS}


def run(ccg_tops: float = DEFAULT_CCG_TOPS, enable_debug_log: bool = False) -> None:
    completed_jobs, completion_cycle, simulation_time, profile_dir = run_experiment("run_virtual", VirtualCompiler, lambda device, debug: VirtualRuntime(device, instances=create_virtual_domains(device), scheduler=create_scheduler(), enable_debug_log=debug, require_full_coverage=True, exclusive_dma=True), ccg_tops, enable_debug_log, DOMAIN_IDS)
    logger.info(f"Completed {completed_jobs} kernels at cycle {completion_cycle}; simulation time: {simulation_time:.6f} s")
    logger.info(f"Kernel profiles: {os.path.join(profile_dir, 'kernel_profile')}")
    logger.info(f"Workload profile: {os.path.join(profile_dir, 'workload_profile.csv')}")
    logger.info(f"Scheduler profile: {os.path.join(profile_dir, 'scheduler_profile.csv')}")
    logger.info(f"Metadata: {os.path.join(profile_dir, 'metadata.json')}")


if __name__ == "__main__":
    run()
