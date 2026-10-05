import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRCS_ROOT = os.path.join(REPO_ROOT, "srcs")
if SRCS_ROOT not in sys.path:
    sys.path.insert(0, SRCS_ROOT)

from neuromta.component.context.mem_context import MemoryConfig, MemorySimulator


def create_simulator() -> MemorySimulator:
    config = MemoryConfig(
        dramsim3_enable=False,
        mem_addr_offset=0,
        n_instance=1,
        channel_size=1024,
        n_channel_per_instance=2,
        lightweight_read_latency_cycles=10,
        lightweight_write_latency_cycles=12,
        lightweight_channel_bandwidth_bytes_per_cycle=64,
        lightweight_dma_granularity=256,
        instance_dma_map={0: 0},
    )
    return MemorySimulator(config)


def assert_eq(actual, expected, label):
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected}, got {actual}")


def test_mapping():
    simulator = create_simulator()
    ch0 = simulator.get_memory_mapping(0)
    ch1 = simulator.get_memory_mapping(1024)
    assert_eq(ch0["inst_id"], 0, "channel 0 instance")
    assert_eq(ch0["channel_id"], 0, "channel 0 id")
    assert_eq(ch1["inst_id"], 0, "channel 1 instance")
    assert_eq(ch1["channel_id"], 1, "channel 1 id")


def test_same_channel_contention():
    simulator = create_simulator()
    first = simulator.send_request(addr=0, size=256, is_write=False, current_cycle=0)
    second = simulator.send_request(addr=256, size=256, is_write=False, current_cycle=0)
    assert_eq(first["latency_cycles"], 14, "first read latency")
    assert_eq(second["chunks"][0]["bus_start_cycle"], 4, "second read bus start")
    assert_eq(second["latency_cycles"], 18, "second read latency with contention")
    assert_eq(simulator.channel_next_free_cycle[(0, 0)], 8, "channel 0 next free")


def test_different_channel_parallelism():
    simulator = create_simulator()
    ch0 = simulator.send_request(addr=0, size=256, is_write=False, current_cycle=0)
    ch1 = simulator.send_request(addr=1024, size=256, is_write=False, current_cycle=0)
    assert_eq(ch0["latency_cycles"], 14, "channel 0 read latency")
    assert_eq(ch1["chunks"][0]["bus_start_cycle"], 0, "channel 1 bus start")
    assert_eq(ch1["latency_cycles"], 14, "channel 1 read latency")
    assert_eq(simulator.channel_next_free_cycle[(0, 0)], 4, "channel 0 next free")
    assert_eq(simulator.channel_next_free_cycle[(0, 1)], 4, "channel 1 next free")


def test_dma_granularity_chunking():
    simulator = create_simulator()
    result = simulator.send_request(addr=0, size=512, is_write=False, current_cycle=0)
    assert_eq(result["n_chunks"], 2, "512B request chunk count")
    assert_eq(result["chunks"][0]["bus_start_cycle"], 0, "chunk 0 bus start")
    assert_eq(result["chunks"][0]["bus_finish_cycle"], 4, "chunk 0 bus finish")
    assert_eq(result["chunks"][1]["bus_start_cycle"], 4, "chunk 1 bus start")
    assert_eq(result["chunks"][1]["bus_finish_cycle"], 8, "chunk 1 bus finish")
    assert_eq(result["latency_cycles"], 18, "512B request latency")


def test_global_addr_request_args():
    simulator = create_simulator()
    result = simulator.send_request(addr=1024, size=256, is_write=False, current_cycle=0)
    assert_eq(result["chunks"][0]["inst_id"], 0, "global addr instance id")
    assert_eq(result["chunks"][0]["channel_id"], 1, "global addr channel id")
    assert_eq(result["latency_cycles"], 14, "global addr read latency")


def main():
    tests = [
        test_mapping,
        test_same_channel_contention,
        test_different_channel_parallelism,
        test_dma_granularity_chunking,
        test_global_addr_request_args,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")


if __name__ == "__main__":
    main()
