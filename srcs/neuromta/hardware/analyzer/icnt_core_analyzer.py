from neuromta.framework import *

from neuromta.hardware.core.icnt_core import IcntCore


__all__ = [
    "IcntCoreAnalyzer",
    "IcntCoreAnalyzerEntry",
]


class IcntCoreAnalyzerEntry:
    def __init__(self, src_id: int=None, dst_id: int=None, st_time: int=None, ed_time: int=None, n_flits: int=None):
        self.src_id = src_id
        self.dst_id = dst_id
        self.st_time = st_time
        self.ed_time = ed_time
        self.n_flits = n_flits
        
    @property
    def bandwidth(self) -> float:  # number of flits per cycle
        duration = self.ed_time - self.st_time
        if duration <= 0: return 0.0
        return self.n_flits / duration


class IcntCoreAnalyzer:
    def __init__(self, icnt_core: IcntCore=None):
        if icnt_core is not None and not isinstance(icnt_core, IcntCore):
            raise TypeError("The provided core is not an instance of IcntCore.")
        
        self._icnt_core = icnt_core
        self._entries: list[IcntCoreAnalyzerEntry] = []
        self._ongoing_transactions: dict[str, IcntCoreAnalyzerEntry] = {}
        
        if self._icnt_core is not None:
            self._hook_id = self._icnt_core.register_command_debug_hook(self._analyzer_debug_entry)
        else:
            self._hook_id = None
        
    @property
    def entries(self) -> list[IcntCoreAnalyzerEntry]:
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
            
            pargs = parse_arguments(msg.args, msg.kwargs, ["module_id", "src_id", "dst_id", "subnet", "n_flits", "is_write", "is_response"])

            module_id   = pargs["module_id"]
            src_id      = pargs["src_id"]
            dst_id      = pargs["dst_id"]
            n_flits     = pargs["n_flits"]
            is_write    = pargs["is_write"]
            is_response = pargs["is_response"]
            
            if module_id != self._icnt_core.cmap_context.config.booksim_module_id:
                return  # not a BookSim2 command
            
            n_flits = 1 + (n_flits if ((is_write and not is_response) or (not is_write and is_response)) else 0)
            
            self._ongoing_transactions[msg_id] = IcntCoreAnalyzerEntry(src_id, dst_id, self._icnt_core.timestamp, None, n_flits)
        
        elif cmd.cmd_id == "async_rpc_wait_rsp_msg":
            pargs = parse_arguments(cmd.args, cmd.kwargs, ["req_msg"])
            msg: RPCMessage = pargs["req_msg"]
            
            msg_id = msg.msg_id
            
            if msg_id not in self._ongoing_transactions.keys():
                return  # no ongoing transaction with the same msg_id
            elif msg.cmd_id != "send_companion_command":
                return  # not a companion command
            
            entry = self._ongoing_transactions[msg_id]
            entry.ed_time = self._icnt_core.timestamp
            
            self._entries.append(entry)
            
            del self._ongoing_transactions[msg_id]
            
    def save_traces(self, save_path: str):
        with open(save_path, "wt") as file:            
            file.write("src_id,dst_id,st_time,ed_time,n_flits,bandwidth\n")
            for entry in self._entries:
                file.write(f"{entry.src_id},{entry.dst_id},{entry.st_time},{entry.ed_time},{entry.n_flits},{entry.bandwidth}\n")
                
        logger.info(f"Interconnect core traces saved to \"{save_path}\".")
        
    def load_traces(self, load_path: str):
        self._entries.clear()
        self._ongoing_transactions.clear()
        
        with open(load_path, "rt") as file:
            for line in file.readlines()[1:]:
                src_id, dst_id, st_time, ed_time, n_flits, _ = line.strip().split(",")
                entry = IcntCoreAnalyzerEntry(int(src_id), int(dst_id), int(st_time), int(ed_time), int(n_flits))
                self._entries.append(entry)
                
        logger.info(f"Interconnect core traces loaded from \"{load_path}\".")
        
    def dump_bandwidth_analysis(self, bin_size: int=1) -> list[float]:
        if len(self._entries) == 0:
            logger.warning("No interconnect core trace entry found. Bandwidth analysis skipped.")
            return []
        
        max_time = max(entry.ed_time for entry in self._entries if entry.ed_time is not None)
        n_bins = (max_time // bin_size) + 1
        bins = [0.0 for _ in range(n_bins)]
        
        for entry in self._entries:
            if entry.ed_time is None:
                logger.warning("Found an interconnect core trace entry with undefined end time. Skipping this entry in bandwidth analysis.")
                continue
            
            st_bin = entry.st_time // bin_size
            ed_bin = entry.ed_time // bin_size
            
            for b in range(st_bin, ed_bin + 1):
                bins[b] += entry.bandwidth
            
        return bins
        
    def save_bandwidth_analysis(self, save_path: str, bin_size: int=1):
        bins = self.dump_bandwidth_analysis(bin_size)
            
        with open(save_path, "wt") as file:
            file.write("timestamp,bandwidth[flits/cycle]\n")
            for i, bw in enumerate(bins):
                file.write(f"{i * bin_size},{bw}\n")
                
        logger.info(f"Interconnect core bandwidth analysis saved to \"{save_path}\".")
        