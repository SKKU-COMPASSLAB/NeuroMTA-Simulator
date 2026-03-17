import signal
import time
import sys
import os
import threading as th
import multiprocessing as mp
from typing import Any

from neuromta.framework.logger import LogLevel, _LOG_LEVEL_COLORS, _COLOR_RESET, set_global_monitoring_window, unset_global_monitoring_window
from neuromta.framework.core import Core, Kernel, Command

__all__ = [
    "MonitoringWindow",
]


class ProgressBarHandle:
    def __init__(self, desc: str="", ncols: int=80):
        self.desc = desc
        self.ncols = ncols
        
        self._cached_total    = 0
        self._cached_progress = 0
        
    def draw(self):
        percentage = min(((self._cached_progress) / self._cached_total * 100) if self._cached_total and self._cached_total > 0 else 0.0, 100.0)

        header = f"{self.desc} [{self._cached_progress:<3d}/{self._cached_total:<3d}] {percentage:6.2f}% "
        pbar_prefix = "|"
        tail   = "| "
        
        bar_width = self.ncols - len(header) - len(pbar_prefix) - len(tail)
        
        if 0 <= bar_width < 10:
            sys.stdout.write(f"{header}")
        elif bar_width >= 10:
            if self._cached_total > 0:
                filled_len = int(round(bar_width * percentage / 100))
                bar = '█' * filled_len + ' ' * (bar_width - filled_len)
            else:
                bar = _LOG_LEVEL_COLORS[LogLevel.INFO] + "IDLE".center(bar_width, '-') + _COLOR_RESET
            
            sys.stdout.write(f"{header}{pbar_prefix}{bar}{tail}")
        else:
            sys.stdout.write(f"{'NO SPACE':<{self.ncols}s}")
            
    def __getstate__(self):
        return {
            'desc': self.desc,
            'ncols': self.ncols,
            "_cached_total": self._cached_total,
            "_cached_progress": self._cached_progress,
        }
        
    def __setstate__(self, state):
        self.desc = state['desc']
        self.ncols = state['ncols']
        self._cached_total = state["_cached_total"]
        self._cached_progress = state["_cached_progress"]
        
    def update(self, total: int, progress: int):
        self._cached_total = total
        self._cached_progress = progress


class CoreProgressBarHandle(ProgressBarHandle):
    def __init__(self, desc: str="", ncols: int=80):
        super().__init__(desc, ncols)
        
        self._hook_ids: list[str] = []
        self._binded_core = None
        
    def bind_core(self, core: Core):
        self._binded_core = core
        for slot_id in core._dispatched_main_kernels.keys():
            self._hook_ids.append(core.register_kernel_debug_hook(self._update_pbar_kernel_status, slot_id, 0))  # slot_level=0 means the hook will be triggered for top level kernels in the slot
            self._update_pbar_kernel_status(core=core)  # initial update
        
    def unbind_core(self):
        for hook_id in self._hook_ids:
            self._binded_core.unregister_kernel_debug_hook(hook_id)
            
        self._hook_ids = []
        self._binded_core = None
        self._cached_total = None
    
    def _update_pbar_kernel_status(self, core: Core, kernel: Kernel=None):
        if core.is_idle:
            self._cached_total = 0
            self._cached_progress = 0
            return
        
        if kernel is None:
            self._cached_total = core.n_dispatched_main_kernels
            return
        
        self._cached_progress += 1
        
        
class LogEntryHandle:
    def __init__(self, message: str, level: LogLevel):
        self.message = message
        self.level = level
        
    def draw(self, max_len: int = -1):
        message_formatted = f"[{self.level.name}] {self.message}"
        
        if max_len > 0 and len(message_formatted) > max_len:
            message_formatted = message_formatted[:max_len - 3] + "..."

        sys.stdout.write(f"{_LOG_LEVEL_COLORS[self.level]}{message_formatted}{_COLOR_RESET}")

    def __getstate__(self):
        return {
            'message': self.message,
            'level': self.level,
        }
        
    def __setstate__(self, state):
        self.message = state['message']
        self.level = state['level']
        
        
class MWCommand:
    CLOSE = "CLOSE"
    DRAW  = "DRAW"
    
    def __init__(self, action: str, payload: Any=None):
        self.action = action
        self.payload = payload
        
    def __eq__(self, value):
        if isinstance(value, str):
            return self.action == value
        return super().__eq__(value)
        
        
class MWDrawingProcess(mp.Process):
    def __init__(self, cmd_recv_queue: mp.Queue, default_term_height: int):
        super().__init__()

        self.cmd_recv_queue = cmd_recv_queue
        self.default_term_height = default_term_height
        self._enable_prompt = False
        
    def setup_terminal(self):
        signal.signal(signal.SIGINT, self.exception_handle)
        
        sys.stdout.write("\033[2J")     # erase the entire screen
        sys.stdout.write("\033[?25l")   # hide cursor (prevent blinking)

    def teardown_terminal(self, height: int):
        sys.stdout.write(f"\033[{height};1H\n")  # move cursor to the line after the last log line
        sys.stdout.write("\033[?25h")            # show cursor again
        sys.stdout.flush()
        
    def exception_handle(self):
        sys.stdout.write("\033[?25h") # show cursor again
        sys.stdout.flush()
        
    def draw(self, log_messages: list[LogEntryHandle], pbar_handles: list[CoreProgressBarHandle], term_width: int, term_height: int):
        PROMPT_LINES = 1 if self._enable_prompt else 0
        
        sys.stdout.write("\033[2;1H")  # move cursor to (2, 1)
        for _ in range(term_height - PROMPT_LINES):
            sys.stdout.write("\033[K")  # erase the entire line
            sys.stdout.write("\033[B")  # move cursor down by 1 line
        
        sys.stdout.write("\033[2;1H")  # move cursor to (2, 1)
        
        window_height_offset = 0
        window_width_offset = 0

        if len(pbar_handles) > 0:
            window_height_offset += 1
            
            sys.stdout.write("=== PROGRESS BARS " + "=" * (term_width - 18) + "\n")
            window_height_offset += 1
            
            for pbar in pbar_handles:
                if pbar.ncols > term_width:
                    pbar.ncols = term_width
                
                window_width_offset += pbar.ncols
                
                if window_width_offset > term_width:
                    window_height_offset += 1
                    sys.stdout.write("\n")
                    window_width_offset = pbar.ncols
                
                pbar.draw()
                
            sys.stdout.write("\n")
            window_height_offset += 1  # extra line between progress bars and logs
        else:
            sys.stdout.write("\n")
        
        sys.stdout.write("=== LOG MESSAGES " + "=" * (term_width - 17) + "\n")
        window_height_offset += 1
        
        log_window_height = term_height - (window_height_offset) - PROMPT_LINES

        while len(log_messages) > log_window_height:
            log_messages.pop(0)

        for msg_idx, msg in enumerate(log_messages):
            if msg_idx > 0:
                sys.stdout.write("\n")
            msg.draw(max_len=term_width)
            
        sys.stdout.flush()
        
    def run(self):
        log_messages = []
        pbar_handles = []
        term_width = 160
        term_height = self.default_term_height if self.default_term_height > 0 else 100
        
        self.setup_terminal()
        
        while True:
            try:
                command = self.cmd_recv_queue.get()

                if command == MWCommand.CLOSE:
                    break
                elif command == MWCommand.DRAW:
                    log_messages: list[LogEntryHandle] = command.payload['log_messages']
                    pbar_handles: list[ProgressBarHandle] = command.payload['pbar_handles']
                    term_width: int = command.payload.get('term_width', 160)
                    term_height: int = command.payload.get('term_height', self.default_term_height if self.default_term_height > 0 else 100)

                    self.draw(log_messages, pbar_handles, term_width, term_height)
            except Exception:
                pass  # timeout or other exceptions
            
        self.teardown_terminal(term_height)
        

class MonitoringWindow:
    def __init__(self, leave: bool=False, update_interval: float=0.05, default_term_height: int = -1):
        self.leave = leave
        self.update_interval = update_interval
        self.default_term_height = default_term_height
        
        set_global_monitoring_window(self)
        
        self._log_messages: list[LogEntryHandle]    = []
        self._pbar_handles: list[ProgressBarHandle] = []
        
        self.mw_drawing_process_cmd_queue = mp.Queue()
        self.mw_drawing_process = MWDrawingProcess(
            cmd_recv_queue=self.mw_drawing_process_cmd_queue,
            default_term_height=self.default_term_height
        )
        self.mw_drawing_process.start()
        
        self.mw_timer_stop_event = th.Event()
        self.mw_timer_process = th.Thread(target=self._mw_timer_process_func)
        self.mw_timer_process.start()
        
    def send_mw_draw_command(self):
        try:
            term_width, term_height = os.get_terminal_size()
        except OSError:
            term_width = 160
            term_height = self.default_term_height if self.default_term_height > 0 else 100
            
        if term_height > self.default_term_height > 0:
            term_height = self.default_term_height
        
        self.mw_drawing_process_cmd_queue.put(MWCommand(MWCommand.DRAW, {
            'log_messages': self._log_messages,
            'pbar_handles': self._pbar_handles,
            'term_width': term_width,
            'term_height': term_height,
        }))
            
    def _mw_timer_process_func(self):
        while not self.mw_timer_stop_event.is_set():
            self.send_mw_draw_command()
            time.sleep(self.update_interval)
    
    def add_log(self, message: str, level: LogLevel = LogLevel.INFO) -> LogEntryHandle:
        log_entry = LogEntryHandle(message, level)
        self._log_messages.append(log_entry)
        
        try:
            term_width, term_height = os.get_terminal_size()
        except OSError:
            term_width = 160
            term_height = self.default_term_height if self.default_term_height > 0 else 100
            
        if term_height > self.default_term_height > 0:
            term_height = self.default_term_height
            
        while len(self._log_messages) > term_height:
            self._log_messages.pop(0)

    def add_core_pbar(self, desc: str, ncols: int) -> int:
        if ncols <= 0:
            try:
                term_width, _ = os.get_terminal_size()
                ncols = term_width
            except OSError:
                ncols = 160 if self.default_term_height <= 0 else 100
        
        pbar = CoreProgressBarHandle(desc, ncols)
        self._pbar_handles.append(pbar)
        
        return len(self._pbar_handles) - 1  # return index of the pbar
    
    def add_pbar(self, desc: str, ncols: int=-1) -> int:
        if ncols <= 0:
            try:
                term_width, _ = os.get_terminal_size()
                ncols = term_width
            except OSError:
                ncols = 160 if self.default_term_height <= 0 else 100
        
        pbar = ProgressBarHandle(desc, ncols)
        self._pbar_handles.append(pbar)
        
        return len(self._pbar_handles) - 1  # return index of the pbar
    
    def remove_pbar(self, pbar_index: int):
        pbar = self._pbar_handles[pbar_index]
        if isinstance(pbar, CoreProgressBarHandle):
            pbar.unbind_core()
        self._pbar_handles.pop(pbar_index)
        
    def close(self):
        self.send_mw_draw_command()  # final draw before closing
    
        self.mw_drawing_process_cmd_queue.put(MWCommand(MWCommand.CLOSE))
        self.mw_drawing_process.join()

        self.mw_timer_stop_event.set()
        self.mw_timer_process.join()

        if not self.leave:
            sys.stdout.write("\033[2J")   # erase the entire screen
            sys.stdout.write("\033[2;1H")
            
        for pbar in self._pbar_handles:
            if isinstance(pbar, CoreProgressBarHandle):
                pbar.unbind_core()

        unset_global_monitoring_window()

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        
    @property
    def pbar_handles(self) -> list[ProgressBarHandle | CoreProgressBarHandle]:
        return self._pbar_handles