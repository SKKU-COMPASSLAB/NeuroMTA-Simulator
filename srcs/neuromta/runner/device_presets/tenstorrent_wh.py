import torch

from neuromta.framework import *
from neuromta.component import *
from neuromta.system.hardware.tenstorrent import TenstorrentConfig, TenstorrentDevice
    
config = TenstorrentConfig.WORMHOLE()
DEVICE = TenstorrentDevice(**config).initialize()

CORE_GROUP_OFFSET = (0, 0)
CORE_GROUP_SHAPE = (8, 12)

RECIPE = dict(
    main_space_size_per_channel=parse_mem_cap_str("2GB"),
    data_space_size_per_core=parse_mem_cap_str("1MB"),
    spad_space_size_per_core=parse_mem_cap_str("512KB"),
    broadcast_optimize_queue_depth=8,
    broadcast_optimize_max_ref_cnt=4,
    context_buffer_slot_num=16,
    ld_ex_buffer_slot_num=16,
    ex_st_buffer_slot_num=8,
    concurrent_load_num=1,
    temporal_reuse_type=MCA_OperatorGraphCompiler.CompileRecipe.ReuseType.ALL,
    spatial_reuse_type=MCA_OperatorGraphCompiler.CompileRecipe.ReuseType.SINGLE_MAIN,
    greedy_temporal_reuse=True,
    
    dtype=torch.float16,
    acc_dtype=torch.float16,
)