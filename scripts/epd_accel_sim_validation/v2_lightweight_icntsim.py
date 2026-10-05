import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRCS_ROOT = os.path.join(REPO_ROOT, "srcs")
if SRCS_ROOT not in sys.path:
    sys.path.insert(0, SRCS_ROOT)

from neuromta.component.context.icnt_context import IcntConfig, IcntContext, IcntSimulator


SRC_CORE_ID = 10
DST_CORE_ID = 20


def create_config() -> IcntConfig:
    config = IcntConfig(
        processor_clock_freq=1_000_000_000,
        shape=(2, 3),
        flit_size=64,
        max_payload_size=4,
        subnets=2,
        booksim2_enable=False,
        lightweight_router_latency_cycles=1,
        lightweight_link_latency_cycles=1,
        lightweight_flits_per_cycle_per_channel=2,
        lightweight_injection_flits_per_cycle=4,
        lightweight_egress_flits_per_cycle=4,
    )
    config.update_core_map((0, 0), SRC_CORE_ID)
    config.update_core_map((0, 2), DST_CORE_ID)
    return config


def assert_eq(actual, expected, label):
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected}, got {actual}")


def assert_gt(actual, expected, label):
    if actual <= expected:
        raise AssertionError(f"{label}: expected > {expected}, got {actual}")


def test_data_size_request_packetization():
    simulator = IcntSimulator(create_config())
    result = simulator.send_request(
        src_core_id=SRC_CORE_ID,
        dst_core_id=DST_CORE_ID,
        data_size=320,
        is_write=True,
        current_cycle=0,
    )
    assert_eq(result["src_id"], 0, "source node id")
    assert_eq(result["dst_id"], 2, "destination node id")
    assert_eq(result["n_flits"], 5, "total flits")
    assert_eq(result["n_payloads"], 2, "payload count")
    assert_eq(result["payloads"][0]["n_flits"], 4, "payload 0 flits")
    assert_eq(result["payloads"][1]["n_flits"], 1, "payload 1 flits")
    assert_eq([p["subnet"] for p in result["payloads"]], [0, 1], "payload subnet distribution")
    assert_eq(result["latency_cycles"], 18, "request latency is max payload latency")


def test_single_payload_route_and_latency():
    simulator = IcntSimulator(create_config())
    result = simulator.send_request(
        src_core_id=SRC_CORE_ID,
        dst_core_id=DST_CORE_ID,
        data_size=256,
        is_write=False,
        current_cycle=0,
    )
    payload = result["payloads"][0]
    assert_eq(result["n_flits"], 4, "single payload flits")
    assert_eq(result["n_payloads"], 1, "single payload count")
    assert_eq(payload["src_coord"], (0, 0), "source coord")
    assert_eq(payload["dst_coord"], (0, 2), "destination coord")
    assert_eq(payload["hop_count"], 2, "hop count")
    assert_eq(payload["serialization_cycles"], 2, "serialization cycles")
    assert_eq(payload["injection_cycles"], 1, "injection cycles")
    assert_eq(payload["egress_cycles"], 1, "egress cycles")
    assert_eq(payload["latency_cycles"], 18, "single payload latency")
    assert_eq(len(payload["resources"]), 8, "resource count")


def test_same_path_contention():
    simulator = IcntSimulator(create_config())
    first = simulator.send_request(SRC_CORE_ID, DST_CORE_ID, data_size=256, current_cycle=0)
    second = simulator.send_request(SRC_CORE_ID, DST_CORE_ID, data_size=256, current_cycle=0)
    assert_gt(second["latency_cycles"], first["latency_cycles"], "same path contention latency")
    assert_gt(second["payloads"][0]["resources"][0]["queue_delay_cycles"], 0, "injection queue delay")


def test_subnet_resource_isolation_inside_one_request():
    simulator = IcntSimulator(create_config())
    result = simulator.send_request(SRC_CORE_ID, DST_CORE_ID, data_size=320, current_cycle=0)
    payload0 = result["payloads"][0]
    payload1 = result["payloads"][1]
    assert_eq(payload0["subnet"], 0, "payload 0 subnet")
    assert_eq(payload1["subnet"], 1, "payload 1 subnet")
    assert_eq(payload1["resources"][0]["queue_delay_cycles"], 0, "different subnet injection queue delay")


def test_context_packetization_still_matches_booksim_args():
    context = IcntContext(create_config())
    args = context.get_icnt_data_transfer_args(src_core_id=SRC_CORE_ID, dst_core_id=DST_CORE_ID, data_size=320, is_write=True)
    assert_eq(context.is_icnt_simulator_enabled, True, "context simulator availability")
    assert_eq(len(args), 2, "payload count")
    assert_eq(args[0]["src_id"], 0, "payload 0 src id")
    assert_eq(args[0]["dst_id"], 2, "payload 0 dst id")
    assert_eq(args[0]["subnet"], 0, "payload 0 subnet")
    assert_eq(args[0]["n_flits"], 4, "payload 0 flits")
    assert_eq(args[1]["subnet"], 1, "payload 1 subnet")
    assert_eq(args[1]["n_flits"], 1, "payload 1 flits")


def main():
    tests = [
        test_data_size_request_packetization,
        test_single_payload_route_and_latency,
        test_same_path_contention,
        test_subnet_resource_isolation_inside_one_request,
        test_context_packetization_still_matches_booksim_args,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")


if __name__ == "__main__":
    main()
