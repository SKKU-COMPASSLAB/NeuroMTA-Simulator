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
    def _dma_lightweight_batch_request_handle(self, addrs: list[int], size: int, is_write: bool):
        pass

    @core_command_method
    def _icnt_data_transfer_handle(self, src_core_id: int, dst_core_id: int, data_size: int, is_write: bool):
        pass

    @jit_prototype
    def dma_read_memory_batch(self, req_core_id: int, addrs: list[int], size: int):
        with new_parallel_thread("DMA"):
            if self._mem_context.is_simulator_available:
                self._dma_lightweight_batch_request_handle(addrs=addrs, size=size, is_write=False)
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
                self._dma_lightweight_batch_request_handle(addrs=addrs, size=size, is_write=True)
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

    def _dma_lightweight_batch_request_handle(self, addrs: list[int], size: int, is_write: bool) -> int:
        if not self.core._mem_context.is_simulator_available:
            raise RuntimeError("Memory simulator is not available.")
        result = self.core._mem_context.simulator.send_requests(
            addrs=addrs,
            size=size,
            is_write=is_write,
            current_cycle=self.core.timestamp,
        )
        return result["latency_cycles"]

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
