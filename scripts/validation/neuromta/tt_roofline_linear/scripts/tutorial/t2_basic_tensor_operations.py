import pytest
import torch
import numpy as np
import ttnn
from loguru import logger

def test_main():
    # Open the Device
    device = ttnn.open_device(device_id=0)
    
    # tensor creation examples
    logger.info("\n--- TT-NN Tensor Creation with Tiles (32x32) ---")
    
    t1 = torch.rand((32, 32), dtype=torch.float32)
    tt_t1 = ttnn.from_torch(t1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    
    tt_t2 = ttnn.full(
        shape=(32, 32),
        fill_value=1.0,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    tt_t3 = ttnn.zeros(
        shape=(32, 32),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    tt_t4 = ttnn.ones(
        shape=(32, 32),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    t5 = np.array([[5, 6], [7, 8]], dtype=np.float32).repeat(16, axis=0).repeat(16, axis=1)
    tt_t5 = ttnn.Tensor(t5, device=device, layout=ttnn.TILE_LAYOUT)
    
    # basic tensor arithmetic operations
    logger.info("\n--- TT-NN Tensor Operations on (32x32) Tiles ---")
    add_result = ttnn.add(tt_t1, tt_t4)
    logger.info(f"Addition:\n{add_result}")

    mul_result = ttnn.mul(tt_t1, tt_t5)
    logger.info(f"Element-wise Multiplication:\n{mul_result}")

    matmul_result = ttnn.matmul(tt_t4, tt_t1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    logger.info(f"Matrix Multiplication:\n{matmul_result}")
    
    # simulated broadcasting example
    logger.info("\n--- Simulated Broadcasting (32x32 + Broadcasted Row Vector) ---")
    broadcast_vector = torch.tensor(np.arange(0, 32), dtype=torch.float32).repeat(32, 1)
    logger.info(f"Broadcast Row Vector:\n{broadcast_vector}")

    broadcast_tt = ttnn.from_torch(broadcast_vector, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    broadcast_add_result = ttnn.add(tt_t1, broadcast_tt)
    logger.info(f"Broadcast Add Result (TT-NN):\n{ttnn.to_torch(broadcast_add_result)}")
    
    # close the device
    ttnn.close_device(device)

if __name__ == "__main__":
    test_main()