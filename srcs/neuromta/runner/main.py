import os
import asyncio
import shutil
import sys
import threading
import importlib

import prompt_toolkit
from prompt_toolkit import PromptSession, print_formatted_text
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.auto_suggest import AutoSuggest, Suggestion
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.formatted_text import ANSI
# from prompt_toolkit.application import run_in_terminal

from neuromta.framework import logger


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
            "create_device": {
                "description": "Creates a new device using the specified preset.",
                "primitives": [self.Arg.required("device_name")],
                "method": self._command_create_device,
            },
            "create_model": {
                "description": "Creates a new model using the specified preset.",
                "primitives": [self.Arg.required("model_name")],
                "method": self._command_create_model,
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
    
    def _command_create_device(self, *args):
        pass

    def _command_create_model(self, *args):
        pass

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


if __name__ == "__main__":
    runner = Runner()
    runner.run()