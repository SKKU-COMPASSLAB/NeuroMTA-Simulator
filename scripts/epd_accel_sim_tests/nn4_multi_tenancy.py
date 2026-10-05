import os
import time

import torch

from neuromta.framework.logger import *
from neuromta.system.hardware import *
from neuromta.system.software.implementation.spatial import SpatialCompiler, SpatialRuntime
from neuromta.system.software.nn.vit import *
from neuromta.system.software.nn.llama2 import *
from neuromta.system.software.utils.scheduler import MeshRoundRobinScheduler


def main():
    logger.set_print_options(log_level=LogLevel.DEBUG)

    config = MeshAcceleratorConfig.medium()
    device = MeshAccelerator(**config).initialize()

    with MeshDeviceRuntimeContext(
        device=device,
        default_tile_shape=(32, 32),
        default_dtype=torch.bfloat16,
        runtime=SpatialRuntime(device, scheduler=MeshRoundRobinScheduler(), enable_debug_log=True),
    ) as context:
        with context.new_compiler_context(SpatialCompiler(), arrival_cycle=0, workload_id="ViT.forward") as compiler:
            vit_model = ViT(image_size=64, patch_size=32, num_layers=2, num_heads=4, hidden_dim=128, mlp_dim=256, num_classes=10)
            x = MeshTensorDescriptor(shape=(1, 64, 64, 3), tile_shape=(1, 1, 32, 32), dtype=torch.bfloat16)
            y = vit_model.forward(x)
            if y.shape != (1, 10):
                raise RuntimeError(f"Unexpected ViT output shape: {y.shape}")
            
        with context.new_compiler_context(SpatialCompiler(), arrival_cycle=0, workload_id="Llama2.prefill_decode") as compiler:
            prompt_length = 4
            decode_tokens = 2
            
            llm_model = Llama2(vocab_size=128, hidden_dim=64, intermediate_dim=128, num_layers=1, num_heads=2, max_seq_len=8)
            kv_cache = mesh_kv_cache(batch_size=1, max_seq_len=8, num_layers=llm_model.num_layers, num_kv_heads=llm_model.num_kv_heads, head_dim=llm_model.head_dim)
            prompt = MeshTensorDescriptor(shape=(1, prompt_length), tile_shape=(1, 32), dtype=torch.int64)
            logits = llm_model.forward(prompt, kv_cache)
            
            if logits.shape != (1, prompt_length, llm_model.vocab_size):
                raise RuntimeError(f"Unexpected prefill output shape: {logits.shape}")
            token = mesh_argmax(mesh_narrow(logits, 1, prompt_length - 1, 1), dim=-1, keepdim=False)
            
            for _ in range(decode_tokens):
                logits = llm_model.forward(token, kv_cache)
                if logits.shape != (1, 1, llm_model.vocab_size):
                    raise RuntimeError(f"Unexpected decode output shape: {logits.shape}")
                token = mesh_argmax(logits, dim=-1, keepdim=False)
            
            if kv_cache.current_length != prompt_length + decode_tokens:
                raise RuntimeError(f"Unexpected KV cache length: {kv_cache.current_length}")

        runtime = context.runtime

        simulation_start = time.perf_counter()
        jobs = runtime.run()
        simulation_time = time.perf_counter() - simulation_start

        if not all(workload.state == MeshDeviceRuntimeWorkloadState.COMPLETED for workload in runtime.workloads):
            raise RuntimeError("The ViT workload did not complete.")
        concurrent_decisions = [decision for decision in runtime.decision_log if len(decision["kernels"]) > 1]
        kernel_stats = {}
        for workload in runtime.workloads:
            for kernel in workload.kernel_log:
                kernel_type = kernel.compiled_kernel.kernel_desc.kernel_type.value
                kernel_cycles = kernel.completion_cycle - kernel.dispatch_cycle
                count, cycles = kernel_stats.get(kernel_type, (0, 0))
                kernel_stats[kernel_type] = count + 1, cycles + kernel_cycles
        slowest_kernels = sorted((kernel for workload in runtime.workloads for kernel in workload.kernel_log), key=lambda kernel: kernel.completion_cycle - kernel.dispatch_cycle, reverse=True)[:5]

        logger.info(f"Simulation time: {simulation_time:.6f} s")
        logger.info(f"Simulated cycles: {device.timestamp}")
        logger.info(f"Concurrent decisions: {len(concurrent_decisions)}")
        for kernel_type, (count, cycles) in sorted(kernel_stats.items()):
            logger.info(f"kernel_type={kernel_type} count={count} cycles={cycles}")
        for kernel in slowest_kernels:
            logger.info(f"slowest_kernel={kernel.execution_id} cycles={kernel.completion_cycle - kernel.dispatch_cycle} cores={tuple(kernel.placement.core_mesh.flatten().tolist())} active_dma={kernel.placement.memory_bank_ids}")
        logger.info(f"Completed {len(jobs)} kernels at cycle {device.timestamp}.")
        logger.info(f"Deallocated {runtime.deallocate_weights()} resident weight tensors.")

if __name__ == "__main__":
    main()
