import torch

from neuromta.framework import *
from neuromta.hardware import *


class SimpleNPUCore(Core):
    def __init__(self, core_id):
        super().__init__(core_id, SimpleNPUCoreCycleModel())
    
        self.l1_memory = MemoryHandle(
            mem_id="L1", 
            base_addr=0x00, 
            size=parse_mem_cap_str("2MB")
        )
        
        self.mxu_pe_arr = torch.zeros((128, 128), dtype=torch.int32)
    
    @core_command_method
    def mxu_load(self, psum: BufferPointer):
        data = self.l1_memory.get_content(psum, shape=(128, 128), dtype=torch.int32)
        self.mxu_pe_arr[:, :] = data
        
    @core_command_method
    def mxu_matmul(self, ifm: BufferPointer, wgt: BufferPointer):
        ifm_data = self.l1_memory.get_content(ifm, shape=(128, 128), dtype=torch.int32)
        wgt_data = self.l1_memory.get_content(wgt, shape=(128, 128), dtype=torch.int32)
        
        self.mxu_pe_arr[:, :] = torch.matmul(ifm_data, wgt_data) + self.mxu_pe_arr
        
    @core_command_method
    def mxu_store(self, ofm: BufferPointer):
        self.l1_memory.set_content(ofm, self.mxu_pe_arr)
        
class SimpleNPUCoreCycleModel(CoreCycleModel):
    def __init__(self):
        super().__init__()
        
    def mxu_load(self, psum: BufferPointer):
        return 128
    
    def mxu_matmul(self, ifm: BufferPointer, wgt: BufferPointer):
        return 128
    
    def mxu_store(self, ofm: BufferPointer):
        return 128
    
class SimpleMultiCoreDevice(Device):
    def __init__(self, n_cores: int=4):
        super().__init__()
        
        self.npu_cores: list[SimpleNPUCore] = [
            SimpleNPUCore(core_id) 
            for core_id in range(n_cores)
        ]


@jit_prototype
def example_kernel(
    core:   SimpleNPUCore, 
    ifm:    BufferPointer, 
    wgt:    BufferPointer, 
    psum:   BufferPointer, 
    ofm:    BufferPointer
):
    core.mxu_load(psum)
    core.mxu_matmul(ifm, wgt)
    core.mxu_store(ofm)


if __name__ == "__main__":
    n_cores = 4
    
    device = SimpleMultiCoreDevice(n_cores=n_cores)
    device.initialize()

    ifm_tensor = torch.randint(0, 10, (n_cores, 128, 128), dtype=torch.int32)
    wgt_tensor = torch.randint(0, 10, (n_cores, 128, 128), dtype=torch.int32)
    psum_tensor = torch.randint(0, 10, (n_cores, 128, 128), dtype=torch.int32)
    
    for core_idx, core in enumerate(device.npu_cores):
        ifm = core.l1_memory.allocate_buffer_ptr(page_size=128*128*4, n_pages=1)
        wgt = core.l1_memory.allocate_buffer_ptr(page_size=128*128*4, n_pages=1)
        psum = core.l1_memory.allocate_buffer_ptr(page_size=128*128*4, n_pages=1)
        ofm = core.l1_memory.allocate_buffer_ptr(page_size=128*128*4, n_pages=1)

        core.l1_memory.set_content(ifm, ifm_tensor[core_idx])
        core.l1_memory.set_content(wgt, wgt_tensor[core_idx])
        core.l1_memory.set_content(psum, psum_tensor[core_idx])
            
        kernel = example_kernel(core, ifm, wgt, psum, ofm)
        core.dispatch_main_kernel("example", kernel)

    device.run_kernels()
    
    print(f"simulation terminated in {core.timestamp} cycles")
    
    for i in range(n_cores):
        core = device.npu_cores[i]

        reference = torch.matmul(ifm_tensor[i], wgt_tensor[i]) + psum_tensor[i]
        simulated = core.l1_memory.get_content(ofm, shape=(128, 128), dtype=torch.int32)
        print(f"simulation {'PASSED' if torch.equal(reference, simulated) else 'FAILED'} for core {i}")