import math

from neuromta.framework import *
from neuromta.component.companions.booksim import *


BOOKSIM = "BOOKSIM"


class DebugCore(Core):
    def __init__(self, core_id, booksim2_config: BookSim2Config):
        super().__init__(core_id, None)

        self.booksim2_config = booksim2_config

    def noc_create_data_read_transaction(self, src_id: int, dst_id: int, data_size: int):
        n_flits = math.ceil(data_size / self.booksim2_config._flit_size)
        
        data_req_msg = RPCMessage(
            src_core_id=self.core_id,
            dst_core_id=COMPANION_CORE_ID,
            cmd_id="send_companion_command",
        ).with_args(
            BOOKSIM,
            src_id, dst_id, 
            subnet=(src_id + dst_id) % self.booksim2_config._subnets, n_flits=n_flits, 
            is_write=False, is_response=False
        )
        
        self.async_rpc_send_req_msg(data_req_msg)
        self.async_rpc_wait_rsp_msg(data_req_msg)
        
        data_rsq_msg = RPCMessage(
            src_core_id=self.core_id,
            dst_core_id=COMPANION_CORE_ID,
            cmd_id="send_companion_command",
        ).with_args(
            BOOKSIM,
            dst_id, src_id, 
            subnet=(src_id + dst_id) % self.booksim2_config._subnets, n_flits=n_flits, 
            is_write=False, is_response=True
        )
        
        self.async_rpc_send_req_msg(data_rsq_msg)
        self.async_rpc_wait_rsp_msg(data_rsq_msg)
          
    def noc_create_data_write_transaction(self, src_id: int, dst_id: int, data_size: int):
        n_flits = math.ceil(data_size / self.booksim2_config._flit_size)
        
        data_req_msg = RPCMessage(
            src_core_id=self.core_id,
            dst_core_id=COMPANION_CORE_ID,
            cmd_id="send_companion_command",
        ).with_args(
            BOOKSIM,
            src_id, dst_id, 
            # subnet=(src_id + dst_id) % self.icnt_context.config.booksim2_config._subnets, n_flits=n_flits, 
            subnet=0, n_flits=n_flits, 
            is_write=True, is_response=False
        )
        
        self.async_rpc_send_req_msg(data_req_msg)
        self.async_rpc_wait_rsp_msg(data_req_msg)
        
        # TODO: Currently, I just ignore the write response. It's because it may seem unnecessary
        # for the write transaction to have a response. It assumes that the network is reliable enough
        # and the producer node does not need to check whether the write transaction is successfully
        # received by the consumer node. However, if you want to ensure the reliability of the write 
        # transaction, you can uncomment the following code to wait for the write response.
        
        # data_rsq_msg = RPCMessage(
        #     src_core_id=self.core_id,
        #     dst_core_id=COMPANION_CORE_ID,
        #     cmd_id="send_companion_command",
        # ).with_args(
        #     BOOKSIM,
        #     dst_id, src_id, 
        #     subnet=(src_id + dst_id) % self.booksim2_config._subnets, n_flits=n_flits, 
        #     is_write=True, is_response=True
        # )
        
        # self.async_rpc_send_req_msg(data_rsq_msg)
        # self.async_rpc_wait_rsp_msg(data_rsq_msg)
        

class DebugDevice(Device):
    def __init__(self, booksim2_config: BookSim2Config):
        super().__init__()
        
        self._booksim2_module = BookSim2(booksim2_config)
        
        self._companion_core = CompanionCore()
        self._companion_core.register_companion_module(
            BOOKSIM, self._booksim2_module
        )
        
        self._debug_core = DebugCore(
            core_id="DEBUG_CORE", 
            booksim2_config=booksim2_config
        )
        
    @property
    def core(self):
        return self._debug_core