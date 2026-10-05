import math

from neuromta.framework import *

from neuromta.component.context.icnt_context import IcntContext
from neuromta.component.context.mem_context import MemoryContext
from neuromta.component.context.global_context import GlobalContext


__all__ = [
    "DMATile",
]


class DMATile(Core):
    def __init__(
        self,
        core_id: int,
        global_context: GlobalContext,
        icnt_context: IcntContext,
        mem_context: MemoryContext,
    ):
        super().__init__(core_id, cycle_model=DMATileCycleModel(self))

        self._icnt_context = icnt_context
        self._mem_context = mem_context
        self._global_context = global_context

    @core_command_method
    def _dma_lightweight_request_handle(self, addr: int, size: int, is_write: bool):
        pass

    @core_command_method
    def _icnt_data_transfer_handle(self, src_core_id: int, dst_core_id: int, data_size: int, is_write: bool):
        pass

    @jit_prototype
    def dma_read_memory_batch(self, req_core_id: int, addrs: list[int], size: int):
        with new_parallel_thread("DMA"):
            if self._mem_context.is_simulator_available:
                for addr in addrs:
                    self._dma_lightweight_request_handle(addr=addr, size=size, is_write=False)
            else:
                data_rd_requests = [
                    RPCMessage(
                        src_core_id=self.core_id,
                        dst_core_id=COMPANION_CORE_ID,
                        cmd_id="send_companion_command",
                    ).with_args(
                        self._mem_context.config.dramsim3_module_id,
                        **self._mem_context.get_mem_access_args(
                            address=addr,
                            size=size,
                            is_write=False
                        )
                    )
                    for addr in addrs
                ]

                for msg in data_rd_requests:
                    self.async_rpc_send_req_msg(msg)
                for msg in data_rd_requests:
                    self.async_rpc_wait_rsp_msg(msg)

        with new_parallel_thread("NOC"):
            if self._icnt_context.is_icnt_simulator_enabled:
                self._icnt_data_transfer_handle(
                    src_core_id=self.core_id,
                    dst_core_id=req_core_id,
                    data_size=size * len(addrs),
                    is_write=False
                )
            else:
                noc_args = self._icnt_context.get_icnt_data_transfer_args(
                    src_core_id=self.core_id,
                    dst_core_id=req_core_id,
                    data_size=size * len(addrs),
                    is_write=False
                )

                noc_requests = [
                    RPCMessage(
                        src_core_id=self.core_id,
                        dst_core_id=COMPANION_CORE_ID,
                        cmd_id="send_companion_command",
                    ).with_args(
                        self._icnt_context.config.booksim2_module_id,
                        **args
                    )
                    for args in noc_args
                ]

                for msg in noc_requests:
                    self.async_rpc_send_req_msg(msg)
                for msg in noc_requests:
                    self.async_rpc_wait_rsp_msg(msg)

    @jit_prototype
    def dma_write_memory_batch(self, req_core_id: int, addrs: list[int], size: int):
        with new_parallel_thread("NOC"):
            if self._icnt_context.is_icnt_simulator_enabled:
                self._icnt_data_transfer_handle(
                    src_core_id=self.core_id,
                    dst_core_id=req_core_id,
                    data_size=size * len(addrs),
                    is_write=True
                )
            else:
                noc_args = self._icnt_context.get_icnt_data_transfer_args(
                    src_core_id=self.core_id,
                    dst_core_id=req_core_id,
                    data_size=size * len(addrs),
                    is_write=True
                )

                noc_requests = [
                    RPCMessage(
                        src_core_id=self.core_id,
                        dst_core_id=COMPANION_CORE_ID,
                        cmd_id="send_companion_command",
                    ).with_args(
                        self._icnt_context.config.booksim2_module_id,
                        **args
                    )
                    for args in noc_args
                ]

                for msg in noc_requests:
                    self.async_rpc_send_req_msg(msg)
                for msg in noc_requests:
                    self.async_rpc_wait_rsp_msg(msg)

        with new_parallel_thread("DMA"):
            if self._mem_context.is_simulator_available:
                for addr in addrs:
                    self._dma_lightweight_request_handle(addr=addr, size=size, is_write=True)
            else:
                data_wr_requests = [
                    RPCMessage(
                        src_core_id=self.core_id,
                        dst_core_id=COMPANION_CORE_ID,
                        cmd_id="send_companion_command",
                    ).with_args(
                        self._mem_context.config.dramsim3_module_id,
                        **self._mem_context.get_mem_access_args(
                            address=addr,
                            size=size,
                            is_write=True
                        )
                    )
                    for addr in addrs
                ]

                for msg in data_wr_requests:
                    self.async_rpc_send_req_msg(msg)
                for msg in data_wr_requests:
                    self.async_rpc_wait_rsp_msg(msg)

        self.parallel_merge()

    @jit_prototype
    def dma_read_memory(self, req_core_id: int, addr: int, size: int):
        with new_parallel_thread("DMA"):
            if self._mem_context.is_simulator_available:
                self._dma_lightweight_request_handle(addr=addr, size=size, is_write=False)
            else:
                data_wr_request = RPCMessage(
                    src_core_id=self.core_id,
                    dst_core_id=COMPANION_CORE_ID,
                    cmd_id="send_companion_command",
                ).with_args(
                    self._mem_context.config.dramsim3_module_id,
                    **self._mem_context.get_mem_access_args(
                        address=addr,
                        size=size,
                        is_write=False
                    )
                )

                self.async_rpc_send_req_msg(data_wr_request)
                self.async_rpc_wait_rsp_msg(data_wr_request)

        with new_parallel_thread("NOC"):
            if self._icnt_context.is_icnt_simulator_enabled:
                self._icnt_data_transfer_handle(
                    src_core_id=self.core_id,
                    dst_core_id=req_core_id,
                    data_size=size,
                    is_write=False
                )
            else:
                noc_args = self._icnt_context.get_icnt_data_transfer_args(
                    src_core_id=req_core_id,
                    dst_core_id=self.core_id,
                    data_size=size,
                    is_write=False
                )

                noc_requests = [
                    RPCMessage(
                        src_core_id=self.core_id,
                        dst_core_id=COMPANION_CORE_ID,
                        cmd_id="send_companion_command",
                    ).with_args(
                        self._icnt_context.config.booksim2_module_id,
                        **args
                    )
                    for args in noc_args
                ]

                for msg in noc_requests:
                    self.async_rpc_send_req_msg(msg)
                for msg in noc_requests:
                    self.async_rpc_wait_rsp_msg(msg)

        self.parallel_merge()

    @jit_prototype
    def dma_write_memory(self, req_core_id: int, addr: int, size: int) -> None:
        with new_parallel_thread("NOC"):
            if self._icnt_context.is_icnt_simulator_enabled:
                self._icnt_data_transfer_handle(
                    src_core_id=self.core_id,
                    dst_core_id=req_core_id,
                    data_size=size,
                    is_write=True
                )
            else:
                noc_args = self._icnt_context.get_icnt_data_transfer_args(
                    src_core_id=self.core_id,
                    dst_core_id=req_core_id,
                    data_size=size,
                    is_write=True
                )

                noc_requests = [
                    RPCMessage(
                        src_core_id=self.core_id,
                        dst_core_id=COMPANION_CORE_ID,
                        cmd_id="send_companion_command",
                    ).with_args(
                        self._icnt_context.config.booksim2_module_id,
                        **args
                    )
                    for args in noc_args
                ]

                for msg in noc_requests:
                    self.async_rpc_send_req_msg(msg)
                for msg in noc_requests:
                    self.async_rpc_wait_rsp_msg(msg)

        with new_parallel_thread("DMA"):
            if self._mem_context.is_simulator_available:
                self._dma_lightweight_request_handle(addr=addr, size=size, is_write=True)
            else:
                data_wr_request = RPCMessage(
                    src_core_id=self.core_id,
                    dst_core_id=COMPANION_CORE_ID,
                    cmd_id="send_companion_command",
                ).with_args(
                    self._mem_context.config.dramsim3_module_id,
                    **self._mem_context.get_mem_access_args(
                        address=addr,
                        size=size,
                        is_write=True
                    )
                )

                self.async_rpc_send_req_msg(data_wr_request)
                self.async_rpc_wait_rsp_msg(data_wr_request)

        self.parallel_merge()

class DMATileCycleModel(CoreCycleModel):
    def __init__(self, core: DMATile):
        super().__init__()

        self.core = core

    def _dma_lightweight_request_handle(self, addr: int, size: int, is_write: bool) -> int:
        if not self.core._mem_context.is_simulator_available:
            raise RuntimeError("Memory simulator is not available. Please use the DRAMSim3 companion module is properly installed and configured.")

        result = self.core._mem_context.simulator.send_request(
            addr=addr,
            size=size,
            is_write=is_write,
            current_cycle=self.core.timestamp
        )

        latency_cycles = result["latency_cycles"]
        return latency_cycles

    def _icnt_data_transfer_handle(self, src_core_id: int, dst_core_id: int, data_size: int, is_write: bool):
        if not self.core._icnt_context.is_icnt_simulator_enabled:
            raise RuntimeError("ICNT simulator is not available. Please use the BookSim2 companion module is properly installed and configured.")

        if src_core_id == dst_core_id:
            return 1

        result = self.core._icnt_context.simulator.send_request(
            src_core_id=src_core_id,
            dst_core_id=dst_core_id,
            data_size=data_size,
            is_write=is_write,
            current_cycle=self.core.timestamp
        )

        latency_cycles = result["latency_cycles"]
        return latency_cycles

    # def dma_read_memory_batch(self, req_core_id: int, addrs: list[int], size: int) -> int:
    #     return self._dma_memory_batch(req_core_id, addrs, size, False)

    # def dma_write_memory_batch(self, req_core_id: int, addrs: list[int], size: int) -> int:
    #     return self._dma_memory_batch(req_core_id, addrs, size, True)

    # def _dma_memory_batch(self, req_core_id: int, addrs: list[int], size: int, is_write: bool) -> int:
    #     if not addrs or size <= 0:
    #         return 0
    #     start_cycle = self.core.timestamp
    #     memory_simulator = self.core._mem_context.simulator
    #     memory_config = self.core._mem_context.config
    #     channel_bytes = {}
    #     for addr in addrs:
    #         if self.core._mem_context.get_dma_id_with_address(addr) != self.core.core_id:
    #             raise ValueError(f"Address {addr} does not belong to DMA engine {self.core.core_id}.")
    #         if self.core._mem_context.get_dma_id_with_address(addr + size - 1) != self.core.core_id:
    #             raise ValueError(f"Address range [{addr}, {addr + size}) crosses DMA engine boundaries.")
    #         mapping = memory_simulator.get_memory_mapping(addr)
    #         channel_key = (mapping["inst_id"], mapping["channel_id"])
    #         channel_bytes[channel_key] = channel_bytes.get(channel_key, 0) + size
    #     memory_finish_cycle = start_cycle
    #     memory_completion_cycle = start_cycle
    #     raw_latency = memory_config.lightweight_write_latency_cycles if is_write else memory_config.lightweight_read_latency_cycles
    #     startup_latency = memory_config.lightweight_request_startup_latency_cycles
    #     row_latency = memory_config.lightweight_row_miss_penalty_cycles
    #     for channel_key, transfer_bytes in channel_bytes.items():
    #         bus_start_cycle = max(start_cycle + startup_latency + row_latency, memory_simulator._channel_next_free_cycle[channel_key])
    #         last_is_write = memory_simulator._channel_last_is_write[channel_key]
    #         if last_is_write is not None and last_is_write != is_write:
    #             bus_start_cycle += memory_config.lightweight_write_to_read_turnaround_cycles if last_is_write else memory_config.lightweight_read_to_write_turnaround_cycles
    #         bus_finish_cycle = bus_start_cycle + max(1, math.ceil(transfer_bytes / memory_config.lightweight_channel_bandwidth_bytes_per_cycle))
    #         completion_cycle = bus_start_cycle + memory_config.lightweight_write_accept_latency_cycles if is_write and memory_config.lightweight_write_completion_policy == "accept" else bus_finish_cycle + raw_latency
    #         memory_simulator._channel_next_free_cycle[channel_key] = bus_finish_cycle
    #         memory_simulator._channel_last_is_write[channel_key] = is_write
    #         memory_finish_cycle = max(memory_finish_cycle, bus_finish_cycle)
    #         memory_completion_cycle = max(memory_completion_cycle, completion_cycle)
    #     src_core_id = self.core.core_id if is_write else req_core_id
    #     dst_core_id = req_core_id if is_write else self.core.core_id
    #     icnt_simulator = self.core._icnt_context.simulator
    #     icnt_config = self.core._icnt_context.config
    #     src_id = icnt_simulator.core_id_to_node_id(src_core_id)
    #     dst_id = icnt_simulator.core_id_to_node_id(dst_core_id)
    #     n_flits = math.ceil(len(addrs) * size / icnt_config.flit_size)
    #     subnet = (src_id + dst_id) % icnt_config.subnets
    #     icnt_result = icnt_simulator._send_payload(src_id=src_id, dst_id=dst_id, subnet=subnet, n_flits=n_flits, is_write=is_write, is_response=not is_write, current_cycle=start_cycle, payload_index=0)
    #     finish_cycle = max(memory_finish_cycle, memory_completion_cycle, icnt_result["finish_cycle"])
    #     return finish_cycle - start_cycle
