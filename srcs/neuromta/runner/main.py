import os
import asyncio
import shutil
import sys
import threading
import multiprocessing as mp

import prompt_toolkit
from prompt_toolkit import PromptSession, print_formatted_text
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.auto_suggest import AutoSuggest, Suggestion
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.formatted_text import ANSI
# from prompt_toolkit.application import run_in_terminal

from neuromta.framework import logger
from neuromta.runner.session import SessionCommand, SessionMessage, Session


ROOT = os.path.dirname(os.path.abspath(__file__))


class ANSIStreamWrapper:
    def __init__(self, proxy_stream):
        self.proxy_stream = proxy_stream

    def write(self, text):
        if not text:
            return
        print_formatted_text(ANSI(text), file=self.proxy_stream, end='')

    def flush(self):
        self.proxy_stream.flush()

    def __getattr__(self, attr):
        return getattr(self.proxy_stream, attr)


class MethodAutoSuggest(AutoSuggest):
    def __init__(self, methods: list[str], keywords: list[str]):
        self.methods  = methods
        self.keywords = keywords
    
    @staticmethod
    def find_suggestion(token: str, candidates: list[str]) -> str:
        matches = [c for c in candidates if c.startswith(token)]
        if not matches:
            return None
        common_prefix = os.path.commonprefix(matches)
        return common_prefix

    def get_suggestion(self, buffer, document):
        text = document.text
        
        if not text:
            return None
        
        if ' ' in text:
            cmd, partial_arg = text.split(' ', 1)
            args = partial_arg.split(' ')
            
            if len(args) >= 1:
                last_arg = args[-1]
            else:
                return None
            
            if len(last_arg) == 0:
                return None
                
            if cmd in self.methods:
                suggestion = self.find_suggestion(last_arg, self.keywords)
                if suggestion:
                    return Suggestion(suggestion[len(last_arg):])
                return None
            else:
                return None
        else:
            suggestion = self.find_suggestion(text, self.methods)
            if suggestion:
                return Suggestion(suggestion[len(text):])
        return None


class Runner:
    class Arg:
        def __init__(self, t: str, v: list[str]):
            self.type = t
            self.values = v
        
        @classmethod
        def choice(cls, *v):
            return cls("choice", list(v))

        @classmethod
        def optional(cls, *v):
            return cls("optional", list(v))
        
        @classmethod
        def required(cls, *v):
            return cls("required", list(v))
        
        def primitive(self):
            if self.type == "choice":
                return f"[{'|'.join(self.values)}]"
            elif self.type == "optional":
                return f"({'|'.join(self.values)})"
            elif self.type == "required":
                return f"<{' '.join(self.values)}>"
            else:
                return ""
    
    def __init__(self):
        self.device_presets_dirs = [os.path.join(ROOT, "device_presets")]
        self.model_presets_dirs  = [os.path.join(ROOT, "model_presets")]
        
        self._keywords = ["help", "exit", "model", "device"]
        self._methods = {
            "list": {
                "description": "Lists all available models and devices.",
                "primitives": [self.Arg.choice("model", "device")],
                "method": self._command_list,
            },
            "open_repo": {
                "description": "Opens a directory storing model or device presets.",
                "primitives": [self.Arg.choice("model", "device"), self.Arg.required("path")],
                "method": self._command_open_repo,
            },
            "open_session": {
                "description": "Opens a new session with the specified model and device presets.",
                "primitives": [self.Arg.required("device_name"), self.Arg.required("model_name")],
                "method": self._command_open_session,
            },
            "close_session": {
                "description": "Closes the currently open session.",
                "primitives": [],
                "method": self._command_close_session,
            },
            "set_session_recipe": {
                "description": "Changes the session's device recipe parameter.",
                "primitives": [self.Arg.required("key"), self.Arg.required("value")],
                "method": self._command_set_session_recipe,
            },
            "set_core_group_shape": {
                "description": "Changes the session's core group shape.",
                "primitives": [self.Arg.required("dim1"), self.Arg.optional("dim2")],
                "method": self._command_set_core_group_shape,
            },
            "set_core_group_offset": {
                "description": "Changes the session's core group offset.",
                "primitives": [self.Arg.required("offset1"), self.Arg.optional("offset2")],
                "method": self._command_set_core_group_offset,
            },
            "compile_graph": {
                "description": "Compiles the model on the device and prepares for execution.",
                "primitives": [],
                "method": self._command_compile_graph,
            },
            "run_graph": {
                "description": "Runs the compiled graph.",
                "primitives": [],
                "method": self._command_run_graph,
            },
            "clear": {
                "description": "Clears the console screen.",
                "primitives": [],
            },
            "help": {
                "description": "Shows this help message.",
                "primitives": [],
            },
            "exit": {
                "description": "Exits the NeuroMTA Runner.",
                "primitives": [],
            },
        }
        
        self._cached_model_presets:  dict[str, str] = None
        self._cached_device_presets: dict[str, str] = None
        
        self._keywords.extend(self._get_model_presets().keys())
        self._keywords.extend(self._get_device_presets().keys())

        self.history = InMemoryHistory()
        self.suggester = MethodAutoSuggest(list(self._methods.keys()), self._keywords)
        self.bindings = KeyBindings()
        
        self._sessions: list[Session] = []
        self._session_cmd_q: list[mp.Queue] = []
        self._session_msg_q: mp.Queue = mp.Queue()  # Shared message queue for receiving messages from all sessions
        self._session_compile_summary: dict[int, list[dict[str, str]]] = None  # Store the compile summary from the session after compilation

        @self.bindings.add('tab')
        def _(event):
            b = event.app.current_buffer
            text = b.text
            
            last_token, candidates = self._tokenize_text(text)
            matches = [m for m in candidates if m.startswith(last_token)]
            
            if not matches:
                return

            if len(matches) == 1:
                remainder = matches[0][len(last_token):]
                b.insert_text(remainder)
                
                if self.is_token_path_like(matches[0]) and os.path.isdir(matches[0]):
                    b.insert_text(os.sep)  # add a separator if it's a directory
                else:
                    b.insert_text(" ")  # add a space after completing a command/keyword
            else:
                common_prefix = os.path.commonprefix(matches)
                if len(common_prefix) > len(last_token):
                    remainder = common_prefix[len(last_token):]
                    b.insert_text(remainder)
                else:
                    # def print_candidates():
                    #     sys.stdout.write('  '.join([(os.path.split(m)[-1] if self.is_token_path_like(m) else m) for m in matches]) + '\n')
                    # run_in_terminal(print_candidates)
                    sys.stdout.write('  '.join([(os.path.split(m)[-1] if self.is_token_path_like(m) else m) for m in matches]) + '\n')

    @staticmethod
    def is_token_path_like(token: str) -> bool:
        return os.sep in token or token.startswith('~')
                    
    def _tokenize_text(self, text: str) -> tuple[str, list[str]]:
        tokens = []
        
        for t in text.split():
            tokens.append(t)
            
        if text.endswith(' ') or len(tokens) == 0:
            tokens.append('')
        
        last_token = tokens[-1]
        
        # CASE 1: If the last token is empty, suggest top-level commands or keywords based on the command primitive
        if len(last_token) == 0:
            if len(tokens) <= 1:    # currently typing the command
                return last_token, list(self._methods.keys())
            else:
                arg_idx = len(tokens) - 2
                cmd_token = tokens[0]
                arg_primitive = self._methods.get(cmd_token, {}).get("primitives", [])
                
                if arg_idx < len(arg_primitive):
                    prim: Runner.Arg = arg_primitive[arg_idx]
                    if prim.type == "choice":
                        return last_token, prim.values
                    
                    cds = []
                    if "model_name" in prim.values:
                        cds += list(self._get_model_presets().keys())
                    if "device_name" in prim.values:
                        cds += list(self._get_device_presets().keys())
                        
                    if len(cds) > 0:
                        return last_token, cds
                return last_token, self._keywords
            
        # CASE 2: If the last token is not empty and is the path-like string, suggest filesystem paths
        if self.is_token_path_like(last_token):
            return self._tokenize_path_candidate(last_token)
            
        # CASE 3: If the last token is not empty, suggest based on the current token
        cds = list(self._methods.keys()) + self._keywords
        return last_token, [c for c in cds if c.startswith(last_token)]
    
    def _tokenize_path_candidate(self, path: str) -> tuple[str, list[str]]:
        if path.startswith('~'):
            path = os.path.expanduser(path)
        path = os.path.expandvars(path)
        
        if os.path.isdir(path):
            candidates = [os.path.join(path, f) for f in os.listdir(path)]
            return path, [c for c in candidates if c.startswith(path)]
        
        tokens = path.split(os.sep)
        parent_path = os.sep.join(tokens[:-1]) if len(tokens) > 1 else os.curdir
        
        if os.path.isdir(parent_path):
            candidates = [os.path.join(parent_path, f) for f in os.listdir(parent_path)]
            return path, [c for c in candidates if c.startswith(path)]
        
        return path, []
        
    def _get_model_presets(self):
        if self._cached_model_presets is not None:
            return self._cached_model_presets
        
        self._cached_model_presets = {}        
        
        for dir in self.model_presets_dirs:
            if os.path.exists(dir):
                for file in os.listdir(dir):
                    if file.endswith(".py") and file != "__init__.py":
                        model_name = file[:-3]
                        self._cached_model_presets[model_name] = os.path.join(dir, file)
            else:
                logger.warning(f"Model presets directory not found: {dir}")

        return self._cached_model_presets
    
    def _get_device_presets(self):
        if self._cached_device_presets is not None:
            return self._cached_device_presets

        self._cached_device_presets = {}
        for dir in self.device_presets_dirs:
            if os.path.exists(dir):
                for file in os.listdir(dir):
                    if file.endswith(".py") and file != "__init__.py":
                        device_name = file[:-3]
                        self._cached_device_presets[device_name] = os.path.join(dir, file)
            else:
                logger.warning(f"Device presets directory not found: {dir}")
                
        return self._cached_device_presets

    def _list_models(self):
        logger.info("Models:")
        for model_name in self._get_model_presets().keys():
            logger.info(f" - {model_name}")
    
    def _list_devices(self):
        logger.info("Devices:")
        for device_name in self._get_device_presets().keys():
            logger.info(f" - {device_name}")

    def _command_list(self, *args):
        if len(args) == 0:
            self._list_models()
            self._list_devices()
        elif args[0] == "model":
            self._list_models()
        elif args[0] == "device":
            self._list_devices()
        else:
            logger.error("Invalid argument for 'list' command. Use 'model' or 'device'.")
    
    def _command_open_repo(self, repo_type: str, path: str):
        if repo_type not in ["model", "device"]:
            logger.error("Invalid repository type. Use 'model' or 'device'.")
            return
        
        if not os.path.exists(path):
            logger.error(f"Path does not exist: {path}")
            return
        
        if repo_type == "model":
            self.model_presets_dirs.append(path)
            self._cached_model_presets = None  # Invalidate cache
            logger.info(f"Added '{path}' to model presets directories.")
        else:
            self.device_presets_dirs.append(path)
            self._cached_device_presets = None
            logger.info(f"Added '{path}' to device presets directories.")
    
        device_presets = self._get_device_presets()
        model_presets = self._get_model_presets()
        
        self._keywords = list(set(self._keywords).union(list(device_presets.keys()) + list(model_presets.keys())))
        self.suggester.keywords = self._keywords
    
    def _command_open_session(self, device_preset_name: str, model_preset_name: str, n_workers: str = "1"):
        if len(self._sessions) > 0:
            logger.error("A session is already open. Please close the current session before opening a new one.")
            return
        
        n_workers = int(n_workers)
        device_preset_path = self._get_device_presets().get(device_preset_name, None)
        model_preset_path  = self._get_model_presets().get(model_preset_name, None)
        
        if device_preset_path is None:
            logger.error(f"Device preset not found: {device_preset_name}")
            return
        if model_preset_path is None:
            logger.error(f"Model preset not found: {model_preset_name}")
            return
        
        self._session_cmd_q = [mp.Queue() for _ in range(n_workers)]
        self._sessions = [
            Session(
                session_id=i, cmd_q=self._session_cmd_q[i], msg_q=self._session_msg_q, 
                device_lib_path=device_preset_path, model_lib_path=model_preset_path
            )
            for i in range(n_workers)
        ]
        self._session_compile_summary = None  # Reset compile summary when opening a new session
        
        for session in self._sessions:
            session.start()
            
        is_session_initialized = True
        
        for _ in range(len(self._sessions)):
            msg: SessionMessage = self._session_msg_q.get()
            if msg.msg_type == "error":
                logger.error(f"Session {msg.session_id} initialization failed: {msg.payload}")
                is_session_initialized = False
            elif msg.msg_type == "done":
                logger.info(f"Session {msg.session_id} initialization succeeded.")
            else:
                logger.warning(f"Session {msg.session_id} sent unknown message type: {msg.msg_type}")
                is_session_initialized = False
                
        if not is_session_initialized:
            logger.error("One or more sessions failed to initialize. Terminating sessions...")
            for session in self._sessions:
                if session.is_alive():
                    session.terminate()
        else:
            logger.info("All sessions initialized successfully.")
    
    def _command_close_session(self):
        for session in self._sessions:
            if session.is_alive():
                session.cmd_q.put(SessionCommand(cmd_type="exit"))
                session.join(timeout=5)
                if session.is_alive():
                    logger.warning(f"Session {session.session_id} did not exit gracefully. Terminating forcefully...")
                    session.terminate()
                else:
                    logger.info(f"Session {session.session_id} closed successfully.")
                    
        self._session_cmd_q = []
        self._sessions = []
        self._session_compile_summary = None  # Clear compile summary when sessions are closed
    
    def _command_set_session_recipe(self, key: str, value: str):
        if len(self._sessions) == 0:
            logger.error("No active session to set recipe.")
            return
        
        for session in self._sessions:
            session.cmd_q.put(SessionCommand(cmd_type="change_recipe", args=(key, value)))
    
    def _command_set_core_group_shape(self, *dims: str):
        if len(self._sessions) == 0:
            logger.error("No active session to set core group shape.")
            return
        
        shape = tuple(map(int, dims))
        
        for session in self._sessions:
            session.cmd_q.put(SessionCommand(cmd_type="change_core_group_shape", args=(shape,)))
            
        for _ in range(len(self._sessions)):
            msg: SessionMessage = self._session_msg_q.get()
            if msg.msg_type == "error":
                logger.error(f"Failed to set core group shape for session {msg.session_id}: {msg.payload}")
            elif msg.msg_type == "done":
                logger.info(f"Core group shape updated successfully for session {msg.session_id}.")
            else:
                logger.warning(f"Session {msg.session_id} sent unknown message type: {msg.msg_type}")
    
    def _command_set_core_group_offset(self, *dims: str):
        if len(self._sessions) == 0:
            logger.error("No active session to set core group offset.")
            return
        
        offset = tuple(map(int, dims))
        
        for session in self._sessions:
            session.cmd_q.put(SessionCommand(cmd_type="change_core_group_offset", args=(offset,)))
            
        for _ in range(len(self._sessions)):
            msg: SessionMessage = self._session_msg_q.get()
            if msg.msg_type == "error":
                logger.error(f"Failed to set core group offset for session {msg.session_id}: {msg.payload}")
            elif msg.msg_type == "done":
                logger.info(f"Core group offset updated successfully for session {msg.session_id}.")
            else:
                logger.warning(f"Session {msg.session_id} sent unknown message type: {msg.msg_type}")
    
    def _command_compile_graph(self):
        if len(self._sessions) == 0:
            logger.error("No active session to compile graph.")
            return
        
        for session in self._sessions:
            session.cmd_q.put(SessionCommand(cmd_type="compile_graph"))
        
        compile_summary: dict[int, list[dict[str, str]]] = None
        
        for _ in range(len(self._sessions)):
            msg: SessionMessage = self._session_msg_q.get()
            if msg.msg_type == "error":
                logger.error(f"Failed to compile graph for session {msg.session_id}: {msg.payload}")
            elif msg.msg_type == "done":
                logger.info(f"Graph compiled successfully for session {msg.session_id}.")
                if compile_summary is None:
                    compile_summary = msg.payload
            else:
                logger.warning(f"Session {msg.session_id} sent unknown message type: {msg.msg_type}")

        if isinstance(compile_summary, dict):
            logger.info("Compilation Summary:")
            for group_idx, group in compile_summary.items():
                logger.info(f"  GROUP {group_idx}:")
                for entry_idx, entry in enumerate(group):
                    logger.info(f"    ENTRY {entry_idx}: node={entry['node']}, op_method={entry['op_method']}")

            self._session_compile_summary = compile_summary  # Store the compile summary for later use in run_graph
        
    def _command_run_graph(self):
        if len(self._sessions) == 0:
            logger.error("No active session to run graph.")
            return
        
        if self._session_compile_summary is None:
            logger.error("No compiled graph available. Please compile the graph before running.")
            return
        
        session_to_entry_info_map = {session_id: [] for session_id in range(len(self._sessions))}
        session_iter = 0
        total_cnt = 0
        
        for group_idx, group in self._session_compile_summary.items():
            for entry_idx, entry in enumerate(group):
                logger.info(f"Scheduling execution for GROUP {group_idx} ENTRY {entry_idx} on session {session_iter}")
                session_to_entry_info_map[session_iter].append((group_idx, entry_idx))
                session_iter = (session_iter + 1) % len(self._sessions)
                
        for session_id, entry_info_list in session_to_entry_info_map.items():
            for group_idx, entry_idx in entry_info_list:
                self._sessions[session_id].cmd_q.put(SessionCommand(cmd_type="run_graph", args=(group_idx, entry_idx)))
                total_cnt += 1
                
        for _ in range(total_cnt):
            msg: SessionMessage = self._session_msg_q.get()
            if msg.msg_type == "error":
                logger.error(f"Failed to run graph entry for session {msg.session_id}: {msg.payload}")
            elif msg.msg_type == "done":
                logger.info(f"Graph entry executed successfully for session {msg.session_id}.)")
                group_idx = msg.payload.get("group_idx", None)
                entry_idx = msg.payload.get("entry_idx", None)
                result = msg.payload.get("result", {})
                logger.info(f"  Result for GROUP {group_idx} ENTRY {entry_idx}")
                for key, value in result.items():
                    logger.info(f"    {key}: {value}")
            else:
                logger.warning(f"Session {msg.session_id} sent unknown message type: {msg.msg_type}")
                
        logger.info("All scheduled graph entries have been executed.")
                
    def get_help_text(self):
        help_text = ""
        for cmd, info in self._methods.items():
            help_text += f"{cmd} {' '.join(map(lambda x: x.primitive(), info['primitives']))}\n"
            if 'description' in info:
                help_text += (' ' * 4) + f"{info['description']}\n"
            if 'example' in info:
                help_text += (' ' * 4) + f"Example: {info['example']}\n"
            help_text += "\n"
        return help_text
        
    async def _show_help(self):
        # Split help text into lines and set the number of lines per page
        lines = self.get_help_text().split('\n')
        terminal_size = shutil.get_terminal_size()
        page_size = max(1, terminal_size.lines - 1)
        
        # Keybindings exclusively for the pager session
        pager_bindings = KeyBindings()
        
        @pager_bindings.add('space')
        def _(event):
            # Exit the prompt returning 'space'
            event.app.exit(result='space')
            
        @pager_bindings.add('q')
        @pager_bindings.add('c-c')
        def _(event):
            # Exit the prompt returning 'q'
            event.app.exit(result='q')
            
        # Create a temporary session just for pagination
        pager_session = PromptSession(
            message="-- More -- (Press 'space' for next page, 'q' to quit) ",
            key_bindings=pager_bindings
        )
        
        for i in range(0, len(lines), page_size):
            # Print the current chunk of lines
            chunk = "\n".join(lines[i:i+page_size])
            print(chunk)
            
            # If there are more lines left, prompt the user to continue
            if i + page_size < len(lines):
                result = await pager_session.prompt_async()
                if result == 'q':
                    break

    def _execute_in_thread(self, cmd_name, *args):
        if cmd_name in self._methods:
            method = self._methods.get(cmd_name, {}).get("method", None)
            if method is None:
                logger.error(f"Command '{cmd_name}' is not implemented yet.")
                return
            try:
                method(*args)
            except Exception as e:
                logger.error(f"An error occurred while executing command '{cmd_name}': {str(e)}")
        else:
            logger.error(f"Unknown command: {cmd_name}")

    async def _run_async(self):
        # Create a prompt session with history, auto-suggest, and custom keybindings
        session = PromptSession(
            history=self.history,
            auto_suggest=self.suggester,
            key_bindings=self.bindings
        )
        
        title = r"""
 __  __                                               ______  ______     
/\ \/\ \                                     /'\_/`\ /\__  _\/\  _  \    
\ \ `\\ \      __    __  __   _ __    ___   /\      \\/_/\ \/\ \ \L\ \   
 \ \ . ` \   /'__`\ /\ \/\ \ /\`'__\ / __`\ \ \ \__\ \  \ \ \ \ \  __ \  
  \ \ \`\ \ /\  __/ \ \ \_\ \\ \ \/ /\ \L\ \ \ \ \_/\ \  \ \ \ \ \ \/\ \ 
   \ \_\ \_\\ \____\ \ \____/ \ \_\ \ \____/  \ \_\\ \_\  \ \_\ \ \_\ \_\ 
    \/_/\/_/ \/____/  \/___/   \/_/  \/___/    \/_/ \/_/   \/_/  \/_/\/_/


Welcome to the NeuroMTA v1.0

Copyright (c) 2026 COMPASSLAB(SKKU). All rights reserved.
"""

        print(title)

        with patch_stdout():
            sys.stdout = ANSIStreamWrapper(sys.stdout)
            sys.stderr = ANSIStreamWrapper(sys.stderr)
            
            while True:
                try:
                    user_input = await session.prompt_async(">>> ")
                    user_input = user_input.strip()

                    if not user_input:
                        continue

                    # Split the input string into the command and its arguments
                    tokens = user_input.split()
                    cmd_token = tokens[0]
                    arg_tokens = tokens[1:]

                    if cmd_token == 'exit':
                        if len(self._sessions) > 0:
                            logger.info("Closing active sessions before exiting...")
                            self._command_close_session()
                        logger.info("Exiting NeuroMTA Runner...")
                        break    
                    if cmd_token == 'help':
                        await self._show_help()
                        continue
                    if cmd_token == 'clear':
                        prompt_toolkit.shortcuts.clear()
                        continue

                    if cmd_token in self._methods.keys():
                        thread = threading.Thread(target=self._execute_in_thread, args=(cmd_token, *arg_tokens))
                        thread.start()
                    else:
                        logger.error(f"Unknown command: {cmd_token}")

                except KeyboardInterrupt:
                    continue
                except EOFError:
                    break

    def run(self):
        asyncio.run(self._run_async())


def main():
    runner = Runner()
    runner.run()


if __name__ == "__main__":
    main()