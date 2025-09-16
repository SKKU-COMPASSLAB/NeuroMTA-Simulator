import time
import sys
import os
import threading as th
import multiprocessing as mp
from typing import Any

from neuromta.framework.logger import LogLevel, _LOG_LEVEL_COLORS, _COLOR_RESET, set_global_monitoring_window, unset_global_monitoring_window
from neuromta.framework.core import Core

__all__ = [
    "MonitoringWindow",
]


class ProgressBarHandle:
    def __init__(self, desc: str="", ncols: int=80, percentage: float=0.0):
        self.desc = desc
        self.ncols = ncols
        self.percentage = percentage
        
        self._hook_id = None
        self._binded_core = None
        self._cached_total = None

    def draw(self):
        header = f"{self.desc} {self.percentage:6.2f}% |"
        tail   = "|  "
        
        bar_width = max(0, self.ncols - len(header) - len(tail) - 1)
        filled_len = int(round(bar_width * self.percentage / 100))
        bar = '█' * filled_len + '-' * (bar_width - filled_len)
        
        sys.stdout.write(f"{header}{bar}{tail}")
        
    def __getstate__(self):
        return {
            'desc': self.desc,
            'ncols': self.ncols,
            'percentage': self.percentage,
        }
        
    def __setstate__(self, state):
        self.desc = state['desc']
        self.ncols = state['ncols']
        self.percentage = state['percentage']
        
    def bind_core(self, core):
        self._binded_core = core
        self._hook_id = core.register_command_debug_hook(self._update_pbar_debug_hook)
        
    def unbind_core(self):
        if self._hook_id is not None and self._binded_core is not None:
            self._binded_core.unregister_command_debug_hook(self._hook_id)
            
            self._hook_id = None
            self._binded_core = None
            self._cached_total = None
    
    def _update_pbar_debug_hook(self, core: Core, *args, **kwargs):
        if core.is_idle:
            self.percentage = 100.0
            self._cached_total = None
            return
        
        n_ongoing_kernels = 0
        n_suspended_kernels = 0
        kernel_progress = 0.0
        
        for slot_id in core._dispatched_main_kernels.keys():
            ongoing_kernel = core._dispatched_main_kernels[slot_id]
            n_ongoing_kernels += 1
            kernel_progress += max(((ongoing_kernel._execution_cursor + 1) / len(ongoing_kernel._execution_steps)) if len(ongoing_kernel._execution_steps) > 0 else 1.0, 1.0)
            
            if slot_id in core._suspended_main_kernels:
                suspended_kernels = core._suspended_main_kernels[slot_id]
                n_suspended_kernels += len(suspended_kernels)
                
        if self._cached_total is None:
            self._cached_total = n_ongoing_kernels + n_suspended_kernels
        
        net_ongoing = self._cached_total - (n_ongoing_kernels + n_suspended_kernels) + kernel_progress
        self.percentage = ((net_ongoing) / self._cached_total * 100) if self._cached_total > 0 else 100.0

        if net_ongoing >= self._cached_total:
            self._cached_total = None
        
        
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
        sys.stdout.write("\033[2J")     # erase the entire screen
        sys.stdout.write("\033[?25l")   # hide cursor (prevent blinking)

    def teardown_terminal(self, height: int):
        sys.stdout.write(f"\033[{height};1H\n")     # move cursor to the line after the last log line
        sys.stdout.write("\033[?25h")                   # show cursor again
        sys.stdout.flush()
        
    def draw(self, log_messages: list[LogEntryHandle], pbar_handles: list[ProgressBarHandle], term_width: int, term_height: int):
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

    def add_pbar(self, desc: str, ncols: int) -> ProgressBarHandle:
        if ncols <= 0:
            try:
                term_width, _ = os.get_terminal_size()
                ncols = term_width
            except OSError:
                ncols = 160 if self.default_term_height <= 0 else 100
        
        pbar = ProgressBarHandle(desc, ncols)
        self._pbar_handles.append(pbar)
        
        return pbar
        
    def close(self):
        # global _global_monitoring_window
        
        self.send_mw_draw_command()  # final draw before closing
    
        self.mw_drawing_process_cmd_queue.put(MWCommand(MWCommand.CLOSE))
        self.mw_drawing_process.join()

        self.mw_timer_stop_event.set()
        self.mw_timer_process.join()

        # _global_monitoring_window = None
        
        if not self.leave:
            sys.stdout.write("\033[2J")   # erase the entire screen
            sys.stdout.write("\033[2;1H")
            
        for pbar in self._pbar_handles:
            pbar.unbind_core()
            
        unset_global_monitoring_window()

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()