from neuromta.framework import *

from neuromta.hardware.core.main_mem_core import MainMemoryCore


__all__ = [
    "MainMemCoreAnalyzer",
    "MainMemCoreAnalyzerEntry",
]


class MainMemCoreAnalyzerEntry:
    def __init__(self, addr: int, size: int, is_write: bool, st_time: int, ed_time: int=None):
        self.addr = addr
        self.size = size
        self.is_write = is_write
        self.st_time = st_time
        self.ed_time = ed_time
        
    @property
    def bandwidth(self) -> float:
        duration = self.ed_time - self.st_time
        if duration <= 0: return 0.0
        return self.size / duration  # number of bytes per cycle
    
    
class MainMemCoreAnalyzer:
    def __init__(self, main_mem_core: MainMemoryCore=None):
        self._main_mem_core = main_mem_core
        self._entries: list[MainMemCoreAnalyzerEntry] = []
        self._ongoing_transactions: dict[str, MainMemCoreAnalyzerEntry] = {}
        
        if self._main_mem_core is not None:
            self._hook_id = self._main_mem_core.register_command_debug_hook(self._analyzer_debug_entry)
        else:
            self._hook_id = None
        
    def register_core(self, core: Core):
        if not isinstance(core, MainMemoryCore):
            raise TypeError("The provided core is not an instance of MainMemoryCore.")

        if self._main_mem_core is not None and self._hook_id is not None:
            self._main_mem_core.unregister_command_debug_hook(self._hook_id)

        self._main_mem_core = core
        self._entries.clear()
        self._ongoing_transactions.clear()
        self._hook_id = self._main_mem_core.register_command_debug_hook(self._analyzer_debug_entry)
        
    @property
    def entries(self) -> list[MainMemCoreAnalyzerEntry]:
        return self._entries
        
    def _analyzer_debug_entry(self, core: Core, kernel: Kernel, cmd: Command, *args, **kwargs):
        if cmd.cmd_id == "async_rpc_send_req_msg":
            pargs = parse_arguments(cmd.args, cmd.kwargs, ["req_msg"])
            msg: RPCMessage = pargs["req_msg"]
            
            msg_id = msg.msg_id
            
            if msg_id in self._ongoing_transactions.keys():
                return  # already ongoing transaction with the same msg_id
            elif msg.cmd_id != "send_companion_command":
                return  # not a companion command
            
            pargs = parse_arguments(msg.args, msg.kwargs, ["module_id", "addr", "size", "is_write"])

            module_id   = pargs["module_id"]
            addr        = pargs["addr"]
            size        = pargs["size"]
            is_write    = pargs["is_write"]
            
            if module_id != self._main_mem_core.cmap_context.config.dramsim_module_id:
                return  # not an interconnect command

            self._ongoing_transactions[msg_id] = MainMemCoreAnalyzerEntry(addr, size, is_write, self._main_mem_core.timestamp, None)

        elif cmd.cmd_id == "async_rpc_wait_rsp_msg":
            pargs = parse_arguments(cmd.args, cmd.kwargs, ["req_msg"])
            msg: RPCMessage = pargs["req_msg"]
            
            msg_id = msg.msg_id
            
            if msg_id not in self._ongoing_transactions.keys():
                return  # no ongoing transaction with the same msg_id
            elif msg.cmd_id != "send_companion_command":
                return  # not a companion command
            
            entry = self._ongoing_transactions[msg_id]
            entry.ed_time = self._main_mem_core.timestamp

            self._entries.append(entry)

            del self._ongoing_transactions[msg_id]

    def save_traces(self, save_path: str):
        with open(save_path, "wt") as file:
            file.write("addr,size,type,st_time,ed_time,bandwidth\n")
            for entry in self._entries:
                file.write(f"{entry.addr},{entry.size},{'WRITE' if entry.is_write else 'READ'},{entry.st_time},{entry.ed_time},{entry.bandwidth}\n")

        logger.info(f"Main memory core traces saved to \"{save_path}\".")
        
    def load_traces(self, load_path: str):
        self._entries.clear()
        self._ongoing_transactions.clear()
        
        with open(load_path, "rt") as file:
            for line in file.readlines()[1:]:
                addr, size, type_str, st_time, ed_time, _ = line.strip().split(",")
                is_write = (type_str == "WRITE")
                
                entry = MainMemCoreAnalyzerEntry(int(addr), int(size), is_write, int(st_time), int(ed_time))
                self._entries.append(entry)
        
        logger.info(f"Main memory core traces loaded from \"{load_path}\".")
        
    def dump_bandwidth_analysis(self, bin_size: int=1) -> list[float]:
        if len(self._entries) == 0:
            logger.warning("No main memory core trace entry found. Bandwidth analysis skipped.")
            return []
        
        max_time = max(entry.ed_time for entry in self._entries if entry.ed_time is not None)
        n_bins = (max_time // bin_size) + 1
        bins = [0.0 for _ in range(n_bins)]
        
        for entry in self._entries:
            if entry.ed_time is None:
                logger.warning("Found a main memory core trace entry with undefined end time. Skipping this entry in bandwidth analysis.")
                continue
            
            st_bin = entry.st_time // bin_size
            ed_bin = entry.ed_time // bin_size
            
            for b in range(st_bin, ed_bin + 1):
                bins[b] += entry.bandwidth
        
        return bins
        
    def save_bandwidth_analysis(self, save_path: str, bin_size: int=1):
        bins = self.dump_bandwidth_analysis(bin_size)
        
        with open(save_path, "wt") as file:
            file.write("timestamp,bandwidth[Byte/cycle]\n")
            for i, bw in enumerate(bins):
                file.write(f"{i * bin_size},{bw}\n")
        
        logger.info(f"Main memory core bandwidth analysis saved to \"{save_path}\".")