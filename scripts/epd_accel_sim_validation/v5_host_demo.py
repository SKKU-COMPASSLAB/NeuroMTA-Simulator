import numpy as np

from neuromta.framework import *
from neuromta.system.hardware.base_accelerator import *


class CustomCore(Core):
    def __init__(self, core_id: int):
        super().__init__(core_id=core_id)
        
    @core_command_method
    def print_message(self, message: str):
        print(f"[Core {self.core_id}] [{self.timestamp}] {message}")


class CustomDevice(BaseAccelerator):
    def __init__(self):
        super().__init__()
        
        self.custom_cores: list[CustomCore] = [
            CustomCore(core_id=i) for i in range(4)
        ]
        

@jit_prototype
def simple_kernel(core: CustomCore, message: str):
    core.print_message(message)
    
@jit_host_job_prototype
def simple_job(device: CustomDevice, core_mesh: np.ndarray, message: str):
    '''Job prototype defined as a multiple kernels'''
    
    job = HostJob()
    for core_id in core_mesh.flatten().tolist():
        core = device.custom_cores[core_id]
        kernel = simple_kernel(core, message)
        job.add_program(slot_id="MAIN", program=kernel)
        
    return job

@jit_host_job_prototype
def simple_job2(device: CustomDevice, core_mesh: np.ndarray, message: str):
    '''Job prototype defined as a single program with multiple kernels
       - with Program() as p: ... generates a single program
       - You can use @jit_program_prototype to define a program prototype instead'''
    
    job = HostJob()
    with Program() as p:
        for core_id in core_mesh.flatten().tolist():
            simple_kernel(device.custom_cores[core_id], message)
        job.add_program(slot_id="MAIN", program=p)
    return job
        

def print_job_info(job: HostJob, indent: int=0):
    print(" " * indent + f"Schedule Release:    {job._schedule_release}")
    print(" " * indent + f"Schedule Deadline:   {job._schedule_deadline}")
    print(" " * indent + f"Actual Start:        {job._actual_start}")
    print(" " * indent + f"Actual Complete:     {job._actual_complete}")
        

def main():
    device = CustomDevice().initialize()
        
    device.host_reset_domain(domain_id="domain1", core_mesh=np.arange(0, 2))
    device.host_reset_domain(domain_id="domain2", core_mesh=np.arange(2, 4))
    
    handle1 = (
        device
        .host_new_job_handle("domain1", simple_job,  "JOB DOMAIN 1: Hello from the host!")
        .add(simple_job, "JOB DOMAIN 1: Message 2 fro the host!")
        .reschedule(10, 20)
    )
    
    handle2 = device.host_new_job_handle("domain1", simple_job,  "JOB DOMAIN 1: Hello from the host!").reschedule(100, 120)
    handle3 = device.host_new_job_handle("domain1", simple_job2, "JOB DOMAIN 1: Hello from the host!").reschedule(200, 220)
    
    handle4 = device.host_new_job_handle("domain2", simple_job,  "JOB DOMAIN 2: Hello from the host!").reschedule(15, 25)
    handle5 = device.host_new_job_handle("domain2", simple_job,  "Dependent Job (handle 2)").add_dependency(handle2).reschedule(release_timestamp=35)
    
    jobs = device.host_run_schedule(event_driven_mode=True)
    
    for job_idx, job in enumerate(jobs):
        print(f"Job {job_idx}:")
        print_job_info(job, indent=2)
        

if __name__ == "__main__":
    main()