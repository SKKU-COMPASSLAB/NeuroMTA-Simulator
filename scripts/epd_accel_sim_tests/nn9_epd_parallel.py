import os
import time

import torch

from neuromta.framework.logger import *
from neuromta.system.hardware import *
from neuromta.system.software.implementation.spatial import SpatialCompiler, SpatialRuntime
from neuromta.system.software.nn.qwen2_5_omni import *
from neuromta.system.software.utils.scheduler import MeshRoundRobinScheduler


def main():
    logger.set_print_options(log_level=LogLevel.DEBUG)

    config = MeshAcceleratorConfig.medium()
    device = MeshAccelerator(**config).initialize()

    prompt_length = 1024
    decode_tokens = 1
    with MeshDeviceRuntimeContext(
        device=device, 
        default_tile_shape=(32, 32), 
        default_dtype=torch.bfloat16, 
        runtime=SpatialRuntime(device, scheduler=MeshRoundRobinScheduler(), enable_debug_log=True)
    ) as context:
        # synthetic workloads for testing E/P/D parallel scenario
        with context.new_compiler_context(SpatialCompiler(), arrival_cycle=0, workload_id="Qwen2.5-Omni.ENCODE"):
            model = Qwen2_5_Omni()
            video_shape = model.video_shape()
            video = MeshTensorDescriptor(shape=video_shape, tile_shape=(1, 1, 1, 32, 32), dtype=torch.bfloat16)
            visual_tokens = model.vision_encoder(video)
            if visual_tokens.shape != (1, 1, model.hidden_dim):
                raise RuntimeError(f"Unexpected visual-token shape: {visual_tokens.shape}")
        with context.new_compiler_context(SpatialCompiler(), arrival_cycle=0, workload_id="Qwen2.5-Omni.PREFILL"):
            kv_cache = mesh_kv_cache(batch_size=1, max_seq_len=2048, num_layers=model.num_layers, num_kv_heads=model.num_kv_heads, head_dim=model.head_dim)
            prompt = MeshTensorDescriptor(shape=model.prompt_shape(prompt_length), tile_shape=(1, 32), dtype=torch.int64)
            logits = model.language_model(prompt, kv_cache, visual_tokens=visual_tokens)
            if logits.shape != (1, prompt_length + visual_tokens.shape[1], model.vocab_size):
                raise RuntimeError(f"Unexpected multimodal prefill output shape: {logits.shape}")
        with context.new_compiler_context(SpatialCompiler(), arrival_cycle=0, workload_id="Qwen2.5-Omni.DECODE"):
            token = mesh_argmax(mesh_narrow(logits, 1, logits.shape[1] - 1, 1), dim=-1, keepdim=False)
            for _ in range(decode_tokens):
                logits = model.language_model(token, kv_cache)
                if logits.shape != (1, 1, model.vocab_size):
                    raise RuntimeError(f"Unexpected decode output shape: {logits.shape}")
                token = mesh_argmax(logits, dim=-1, keepdim=False)
            if kv_cache.current_length != visual_tokens.shape[1] + prompt_length + decode_tokens:
                raise RuntimeError(f"Unexpected KV cache length: {kv_cache.current_length}")

        runtime = context.runtime

        simulation_start = time.perf_counter()
        jobs = runtime.run()
        simulation_time = time.perf_counter() - simulation_start

        if not all(workload.state == MeshDeviceRuntimeWorkloadState.COMPLETED for workload in runtime.workloads):
            raise RuntimeError("The Qwen2.5-Omni workload did not complete.")
        kernel_stats = {}
        for workload in runtime.workloads:
            for kernel in workload.kernel_log:
                kernel_type = kernel.compiled_kernel.kernel_desc.kernel_type.value
                kernel_cycles = kernel.completion_cycle - kernel.dispatch_cycle
                count, cycles = kernel_stats.get(kernel_type, (0, 0))
                kernel_stats[kernel_type] = count + 1, cycles + kernel_cycles
        logger.info(f"Model config: vision_layers={model.vision_num_layers}, language_layers={model.num_layers}, heads={model.num_heads}, kv_heads={model.num_kv_heads}, hidden={model.hidden_dim}, vocab={model.vocab_size}")
        logger.info(f"Video shape: {video.shape}")
        logger.info(f"Visual-token shape: {visual_tokens.shape}")
        logger.info(f"Prompt shape: {prompt.shape}")
        logger.info(f"Final logits shape: {logits.shape}")
        logger.info(f"Generated tokens: {decode_tokens}")
        logger.info(f"KV cache length: {kv_cache.current_length}/{kv_cache.capacity}")
        logger.info(f"Simulation time: {simulation_time:.6f} s")
        logger.info(f"Simulated cycles: {device.timestamp}")
        for kernel_type, (count, cycles) in sorted(kernel_stats.items()):
            logger.info(f"kernel_type={kernel_type} count={count} cycles={cycles}")
        logger.info(f"Completed {len(jobs)} kernels at cycle {device.timestamp}.")
        logger.info(f"Deallocated {runtime.deallocate_weights()} resident weight tensors.")
        logger.info(f"Deallocated {context.deallocate_state(kv_cache.storage_descriptors)} KV cache tensors.")


if __name__ == "__main__":
    main()
