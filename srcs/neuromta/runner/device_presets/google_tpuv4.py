import torch

from neuromta.framework import *
from neuromta.component import *
from neuromta.system.hardware.google_tpu import GoogleTPUConfig, GoogleTPUDevice
    
config = GoogleTPUConfig.V4()
DEVICE = GoogleTPUDevice(**config).initialize()

CORE_GROUP_OFFSET = 0
CORE_GROUP_SHAPE  = 2

RECIPE = dict(
    main_space_size_per_channel=parse_mem_cap_str("2GB"),
    data_space_size_per_core=parse_mem_cap_str("32MB"),
    spad_space_size_per_core=parse_mem_cap_str("4MB"),
    context_buffer_slot_num=16,
    fifo_buffer_slot_num=16,
    temporal_reuse_target=MCA_OperatorGraphCompiler.CompileRecipe.ReuseTarget.ALL,
    spatial_reuse_target=MCA_OperatorGraphCompiler.CompileRecipe.ReuseTarget.SINGLE_MAIN,
    
    dtype=torch.float16,
    acc_dtype=torch.float16,
)
