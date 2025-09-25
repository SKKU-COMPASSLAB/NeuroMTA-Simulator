from neuromta.framework import *
from neuromta.hardware import *
from neuromta.ip.tenstorrent.architecture import *


__all__ = [
    "MCA_RT_KERNEL_LOCAL_CB_ALLOCATE",
    "MCA_RT_KERNEL_LOCAL_CB_DEALLOCATE",
    "MCA_RT_KERNEL_LOCAL_BUF_ALLOCATE",
    "MCA_RT_KERNEL_LOCAL_BUF_DEALLOCATE",
    "MCA_RT_KERNEL_MEM_BUFFER_COPY",
]


class MCA_RT_KERNEL_LOCAL_CB_ALLOCATE(MCA_RuntimeKernel):
    def __init__(self, core: NPUCore, ref: BufferPointer, page_size: int, n_pages: int):
        super().__init__(core=core)

        self.ref = ref
        self.page_size = page_size
        self.n_pages = n_pages
    
    @MCA_RT_KERNEL_THREAD
    def MAIN(self):
        self.core.cb_allocate(self.ref, self.page_size, self.n_pages)
        

class MCA_RT_KERNEL_LOCAL_CB_DEALLOCATE(MCA_RuntimeKernel):
    def __init__(self, core: NPUCore, ref: BufferPointer):
        super().__init__(core=core)
        
        self.ref = ref
        
    @MCA_RT_KERNEL_THREAD
    def MAIN(self):
        self.core.cb_deallocate(self.ref)
        

class MCA_RT_KERNEL_LOCAL_BUF_ALLOCATE(MCA_RuntimeKernel):
    def __init__(self, core: NPUCore, ref: BufferPointer, page_size: int, n_pages: int):
        super().__init__(core=core)
        
        self.ref = ref
        self.page_size = page_size
        self.n_pages = n_pages
        
    @MCA_RT_KERNEL_THREAD
    def MAIN(self):
        self.core.buf_allocate(self.ref, self.page_size, self.n_pages)
        

class MCA_RT_KERNEL_LOCAL_BUF_DEALLOCATE(MCA_RuntimeKernel):
    def __init__(self, core: NPUCore, ref: BufferPointer):
        super().__init__(core=core)
        
        self.ref = ref
        
    @MCA_RT_KERNEL_THREAD
    def MAIN(self):
        self.core.buf_deallocate(self.ref)


class MCA_RT_KERNEL_MEM_BUFFER_COPY(MCA_RuntimeKernel):
    def __init__(self, core: NPUCore, src: BufferPointer, dst: BufferPointer, n_pages: int):
        super().__init__(core=core)

        self.src = src
        self.dst = dst
        self.n_pages = n_pages
        
    @MCA_RT_KERNEL_THREAD
    def MAIN(self):
        self.core.mem_buffer_copy(self.dst, self.src, n_pages=self.n_pages)



