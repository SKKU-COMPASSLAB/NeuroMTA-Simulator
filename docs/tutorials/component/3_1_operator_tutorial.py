import torch

from neuromta.framework import *
from neuromta.component import *
from neuromta.system.software.common.operator import MCA_OP_LINEAR


main_mem_config = MainMemoryConfig(
    dramsim3_enable=False
)

global_config = GlobalContextConfig(
    n_npu_core=4,
    n_dma_core=2,
    l1_mem_bank_size=parse_mem_cap_str("1MB"),
    l1_mem_dynamic_space_size_per_bank=0,
    main_mem_config=main_mem_config,
)

icnt_config = IcntConfig(
    processor_clock_freq=parse_freq_str("1GHz"),
    shape=(2, 3),
    booksim2_enable=False
)
icnt_config.update_core_map((0, 0), global_config.dma_core_ids[0])
icnt_config.update_core_map((0, 1), global_config.npu_core_ids[0])
icnt_config.update_core_map((0, 2), global_config.npu_core_ids[1])
icnt_config.update_core_map((1, 0), global_config.dma_core_ids[1])
icnt_config.update_core_map((1, 1), global_config.npu_core_ids[2])
icnt_config.update_core_map((1, 2), global_config.npu_core_ids[3])

mxu_config = MXUConfig(
    pe_arr_height=32,
    pe_arr_width=32,
    seq_len=32,
)

vpu_config = VPUConfig(
    # use default config
)

device = MTA_DeviceBase(
    global_config=global_config,
    icnt_config=icnt_config,
    mxu_config=mxu_config,
    vpu_config=vpu_config
)

device.initialize()
device.set_command_debug_verbosity(False)
logger.set_print_options(LogLevel.DEBUG)

core_group = device.get_npu_core_group((0, 0), (2, 2))

main_mem_space = device.create_main_mem_space(parse_mem_cap_str("1GB"))
l1_mem_space = device.create_l1_mem_space(parse_mem_cap_str("512KB"), core_group)

M, N, K = 256, 256, 256

torch_ifm = torch.randint(0, 10, (M, K), dtype=torch.int32)
torch_wgt = torch.randint(0, 10, (N, K), dtype=torch.int32)
torch_bias = torch.randint(0, 10, (N,), dtype=torch.int32)

ifm = MCA_TensorBuffer(l1_mem_space, shape=(M, K), dtype=torch.int32)
wgt = MCA_TensorBuffer(main_mem_space, shape=(N, K), dtype=torch.int32)
bias = MCA_TensorBuffer(main_mem_space, shape=(N,), dtype=torch.int32)
ofm = MCA_TensorBuffer(l1_mem_space, shape=(M, N), dtype=torch.int32)

ifm.allocate().update(torch_ifm)
wgt.allocate().update(torch_wgt)
bias.allocate().update(torch_bias)
ofm.allocate()

op = MCA_OP_LINEAR(ifm, wgt, bias, ofm)

compiler = MCA_OperatorGraphCompiler()
recipe = MCA_OperatorGraphCompiler.CompileRecipe(
    device, core_group, parse_mem_cap_str("512KB")
)

compiler.add_op(op)
program = compiler.compile(recipe)

main_mem_space.remove()
l1_mem_space.remove()

program.dispatch()

device.run_kernels()

print(f"simulation terminated with timestamp: {device.timestamp} cycles")

reference_ofm = torch_ifm @ torch_wgt.T + torch_bias
simulated_ofm = ofm.restore()

if torch.equal(reference_ofm, simulated_ofm):
    print("simulation successful: the output matches the reference.")
else:
    print("simulation failed: the output does not match the reference.")
    print("reference OFM:")
    print(reference_ofm)
    print("simulated OFM:")
    print(simulated_ofm)
