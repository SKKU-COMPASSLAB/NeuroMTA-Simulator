from typing import Any

from neuromta.framework import *

from neuromta.component.context.compute_tile_context import ComputeTileContext
from neuromta.component.context.mem_context import MemoryContext
from neuromta.component.context.global_context import GlobalContext
from neuromta.component.context.icnt_context import IcntContext


__all__ = [
    "CCGTile",
    "CCGTileCycleModel",
    "BCAST_MODE_CHAINING",
    "BCAST_MODE_PARALLEL",
]


BCAST_MODE_CHAINING = "CHAINING"
BCAST_MODE_PARALLEL = "PARALLEL"


class CCGTile(Core):
    def __init__(
        self,
        core_id: int,
        global_context: GlobalContext,
        ctile_context: ComputeTileContext,
        icnt_context: IcntContext,
        dma_context: MemoryContext,
    ):
        super().__init__(core_id, cycle_model=CCGTileCycleModel(self))

        self._global_context = global_context
        self._ctile_context = ctile_context
        self._icnt_context  = icnt_context
        self._dma_context = dma_context

    def get_dma_id_with_address(self, address: int) -> int:
        if not self._dma_context.check_address_range(address):
            raise ValueError(f"Address {address} is out of range for the DMA context.")
        return self._dma_context.get_dma_id_with_address(address)

    @core_command_method
    def _icnt_data_transfer_handle(self, src_core_id: int, dst_core_id: int, data_size: int, is_write: bool):
        pass

    @jit_prototype
    def dma_read_memory(self, addr: int, size: int, sync: bool=True):
        dma_id = self.get_dma_id_with_address(addr)

        if dma_id is None:
            raise ValueError(f"No DMA assigned to the memory channel for address {addr}")

        req_msg = RPCMessage(
            src_core_id=self.core_id,
            dst_core_id=dma_id,
            cmd_id="dma_read_memory",
        ).with_args(
            req_core_id=self.core_id,
            addr=addr,
            size=size,
        )

        self.async_rpc_send_req_msg(req_msg)
        if sync:
            self.async_rpc_wait_rsp_msg(req_msg)

    @jit_prototype
    def dma_read_memory_batch(self, addrs: list[int], size: int, sync: bool=True):
        if len(addrs) == 0:
            return

        dma_ids = {self.get_dma_id_with_address(addr) for addr in addrs}
        dma_addr_map = {dma_id: [addr for addr in addrs if self.get_dma_id_with_address(addr) == dma_id] for dma_id in dma_ids}
        dma_req_msgs = []

        for dma_id, addr_list in dma_addr_map.items():
            req_msg = RPCMessage(
                src_core_id=self.core_id,
                dst_core_id=dma_id,
                cmd_id="dma_read_memory_batch",
            ).with_args(
                req_core_id=self.core_id,
                addrs=addr_list,
                size=size,
            )

            self.async_rpc_send_req_msg(req_msg)
            dma_req_msgs.append(req_msg)

        if sync:
            for req_msg in dma_req_msgs:
                self.async_rpc_wait_rsp_msg(req_msg)

    @jit_prototype
    def dma_write_memory(self, addr: int, size: int, sync: bool=True) -> None:
        dma_id = self.get_dma_id_with_address(addr)

        if dma_id is None:
            raise ValueError(f"No DMA assigned to the memory channel for address {addr}")

        req_msg = RPCMessage(
            src_core_id=self.core_id,
            dst_core_id=dma_id,
            cmd_id="dma_write_memory",
        ).with_args(
            req_core_id=self.core_id,
            addr=addr,
            size=size,
        )

        self.async_rpc_send_req_msg(req_msg)
        if sync:
            self.async_rpc_wait_rsp_msg(req_msg)

    @jit_prototype
    def dma_write_memory_batch(self, addrs: list[int], size: int, sync: bool=True) -> None:
        if len(addrs) == 0:
            return

        dma_ids = {self.get_dma_id_with_address(addr) for addr in addrs}
        dma_addr_map = {dma_id: [addr for addr in addrs if self.get_dma_id_with_address(addr) == dma_id] for dma_id in dma_ids}
        dma_req_msgs = []

        for dma_id, addr_list in dma_addr_map.items():
            req_msg = RPCMessage(
                src_core_id=self.core_id,
                dst_core_id=dma_id,
                cmd_id="dma_write_memory_batch",
            ).with_args(
                req_core_id=self.core_id,
                addrs=addr_list,
                size=size,
            )

            self.async_rpc_send_req_msg(req_msg)
            dma_req_msgs.append(req_msg)

        if sync:
            for req_msg in dma_req_msgs:
                self.async_rpc_wait_rsp_msg(req_msg)

    @core_command_method
    def _icnt_data_transfer_batch_handle(self, transfers: tuple[tuple[int, int, int, bool], ...]):
        pass

    @jit_prototype
    def icnt_send_data_batch(self, target_sizes: dict[int, int], sync: bool=True) -> None:
        if not target_sizes:
            return
        if self._icnt_context.is_icnt_simulator_enabled:
            transfers = tuple((self.core_id, target_id, size, True) for target_id, size in target_sizes.items())
            self._icnt_data_transfer_batch_handle(transfers)
        else:
            for target_id, size in target_sizes.items():
                self.icnt_send_data([target_id], size, sync=sync)

    @jit_prototype
    def icnt_recv_data_batch(self, source_sizes: dict[int, int], sync: bool=True) -> None:
        if not source_sizes:
            return
        if self._icnt_context.is_icnt_simulator_enabled:
            transfers = tuple((source_id, self.core_id, size, True) for source_id, size in source_sizes.items())
            self._icnt_data_transfer_batch_handle(transfers)
        else:
            for source_id, size in source_sizes.items():
                self.icnt_recv_data(source_id, size, sync=sync)

    @jit_prototype
    def icnt_send_data(
        self,
        target_core_ids: list[int],
        size: int,
        mode: str=BCAST_MODE_PARALLEL,
        sync: bool=True,
    ) -> None:
        if isinstance(target_core_ids, int):
            target_core_ids = [target_core_ids]
        if mode not in [BCAST_MODE_PARALLEL, BCAST_MODE_CHAINING]:
            raise ValueError(f"Invalid mode: {mode}. Must be either '{BCAST_MODE_PARALLEL}' or '{BCAST_MODE_CHAINING}'")

        # parallel broadcast to all target cores
        if mode == BCAST_MODE_PARALLEL:
            noc_req_msg_q = {target_core_id: [] for target_core_id in target_core_ids}

            for target_core_id in target_core_ids:
                if self._icnt_context.is_icnt_simulator_enabled:
                    self._icnt_data_transfer_handle(
                        src_core_id=self.core_id,
                        dst_core_id=target_core_id,
                        data_size=size,
                        is_write=True
                    )
                else:
                    icnt_args: list[dict[str, Any]] = self._icnt_context.get_icnt_data_transfer_args(
                        src_core_id=self.core_id,
                        dst_core_id=target_core_id,
                        data_size=size,
                        is_write=True
                    )

                    for i, args in enumerate(icnt_args):
                        noc_req_msg_q[target_core_id].append(RPCMessage(
                            src_core_id=self.core_id,
                            dst_core_id=COMPANION_CORE_ID,
                            cmd_id="send_companion_command",
                        ).with_args(
                            self._icnt_context.config.booksim2_module_id,
                            **args
                        ))

            n_msgs = max(len(msgs) for msgs in noc_req_msg_q.values())
            for step in range(n_msgs):
                for target_core_id, msgs in noc_req_msg_q.items():
                    if step < len(msgs):
                        with new_parallel_thread(f"NOC_{target_core_id}"):
                            msg = msgs[step]
                            self.async_rpc_send_req_msg(msg)
                        self.parallel_merge()

            if sync:
                for _, msgs in noc_req_msg_q.items():
                    for msg in msgs:
                        self.async_rpc_wait_rsp_msg(msg)

        # one-to-one broadcast chaining, where each core sends to the next core in the list
        elif mode == BCAST_MODE_CHAINING:
            if len(target_core_ids) == 0:
                return

            current_target = target_core_ids[0]

            icnt_args: list[dict[str, Any]] = self._icnt_context.get_icnt_data_transfer_args(
                src_core_id=self.core_id,
                dst_core_id=current_target,
                data_size=size,
                is_write=True
            )

            req_lock = VariableHandle.tmp(initial_value=0)
            in_flight_requests = []

            for i, args in enumerate(icnt_args):
                with new_parallel_thread(f"NOC_{i}"):
                    self.var_atomic_wait(req_lock, i)

                    if self._icnt_context.is_icnt_simulator_enabled:
                        self._icnt_data_transfer_handle(
                            src_core_id=self.core_id,
                            dst_core_id=current_target,
                            data_size=args["n_flits"] * self._icnt_context.flit_size,
                            is_write=True
                        )
                    else:
                        noc_req_msg = RPCMessage(
                            src_core_id=self.core_id,
                            dst_core_id=COMPANION_CORE_ID,
                            cmd_id="send_companion_command",
                        ).with_args(
                            self._icnt_context.config.booksim2_module_id,
                            **args
                        )

                        self.async_rpc_send_req_msg(noc_req_msg)
                        self.async_rpc_wait_rsp_msg(noc_req_msg)

                    self.var_atomic_compare_and_swap(req_lock, i, i + 1)

                    if len(target_core_ids) <= 1:
                        return

                    chain_req_msg = RPCMessage(
                        src_core_id=self.core_id,
                        dst_core_id=current_target,
                        cmd_id="icnt_send_data",
                    ).with_args(
                        target_core_ids=target_core_ids[1:],
                        size=args["n_flits"] * self._icnt_context.flit_size,
                        mode=BCAST_MODE_CHAINING
                    )

                    self.async_rpc_send_req_msg(chain_req_msg)
                    in_flight_requests.append(chain_req_msg)

            self.parallel_merge()

            if sync:
                for req_msg in in_flight_requests:
                    self.async_rpc_wait_rsp_msg(req_msg)

    @core_command_method
    def icnt_recv_data(self, src_core_id: int, size: int, sync: bool=True) -> None:
        req_msg = RPCMessage(
            src_core_id=self.core_id,
            dst_core_id=src_core_id,
            cmd_id="icnt_send_data",
        ).with_args(
            target_core_ids=[self.core_id],
            size=size,
            mode=BCAST_MODE_PARALLEL
        )

        self.async_rpc_send_req_msg(req_msg)
        if sync:
            self.async_rpc_wait_rsp_msg(req_msg)

    @core_command_method
    def compute(self, n_ops: int) -> None:
        pass

    @property
    def dma_context(self) -> MemoryContext:
        return self._dma_context

    @property
    def ctile_context(self) -> ComputeTileContext:
        return self._ctile_context

    @property
    def icnt_context(self) -> IcntContext:
        return self._icnt_context


class CCGTileCycleModel(CoreCycleModel):
    def __init__(self, core: CCGTile):
        super().__init__()

        self.core = core

    def compute(self, n_ops: int) -> int:
        return int(self.core.ctile_context.get_compute_cycles(n_ops=n_ops))

    def _icnt_data_transfer_batch_handle(self, transfers: tuple[tuple[int, int, int, bool], ...]) -> int:
        if not self.core._icnt_context.is_icnt_simulator_enabled:
            raise RuntimeError("ICNT simulator is not available.")
        current_cycle = self.core.timestamp
        finish_cycle = current_cycle
        simulator = self.core._icnt_context.simulator
        for source_id, target_id, size, is_write in transfers:
            if source_id == target_id:
                finish_cycle = max(finish_cycle, current_cycle + 1)
                continue
            result = simulator.send_request(
                src_core_id=source_id,
                dst_core_id=target_id,
                data_size=size,
                is_write=is_write,
                current_cycle=current_cycle,
            )
            finish_cycle = max(finish_cycle, result["finish_cycle"])
        return finish_cycle - current_cycle

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
