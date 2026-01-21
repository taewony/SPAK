import os
import io
import sys
import litellm
import importlib.util
from typing import Dict, Any, Optional
from .semantic_kernel import Handler, Effect, perform
from .effects import Generate, ExecuteCode, ReadFile, WriteFile, ListFiles, Recurse, LLMRequest, Math, Listen, Reply, SendMessage, SubTask, ReasoningTrace, TraceLog, GrepFiles

# Safe Execution Imports (from recursive-llm wisdom)
from RestrictedPython import compile_restricted_exec, safe_globals, limited_builtins, utility_builtins
from RestrictedPython.Guards import guarded_iter_unpack_sequence, safer_getattr
from RestrictedPython.PrintCollector import PrintCollector

class LiteLLMHandler(Handler):
    def __init__(self, default_model: str = "ollama/qwen3:8b"):
        self.default_model = default_model

    def handle(self, effect: Effect) -> Any:
        if isinstance(effect, Generate):
            req: LLMRequest = effect.payload
            
            # Simple Retry Logic
            max_retries = 3
            import time
            
            for attempt in range(max_retries):
                try:
                    response = litellm.completion(
                        model=req.model or self.default_model,
                        messages=req.messages,
                        stop=req.stop
                    )
                    return response.choices[0].message.content
                except Exception as e:
                    print(f"⚠️ [LiteLLM] Error (Attempt {attempt+1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        time.sleep(2) # Wait 2 seconds before retry
                    else:
                        raise e # Re-raise on final failure
                        
        raise NotImplementedError

class SafeREPLHandler(Handler):
    """
    Highly secure Python REPL Handler inspired by recursive-llm.
    Uses RestrictedPython to prevent malicious or accidental system damage.
    """
    def __init__(self):
        self.env: Dict[str, Any] = {}
        self.max_output_chars = 2000

    def handle(self, effect: Effect) -> Any:
        if isinstance(effect, ExecuteCode):
            return self._execute_safe(effect.payload.code)
        raise NotImplementedError

    def _execute_safe(self, code: str) -> str:
        # Build restricted globals (The wisdom from recursive-llm)
        restricted_globals = safe_globals.copy()
        restricted_globals.update(limited_builtins)
        restricted_globals.update(utility_builtins)
        
        # Add guards
        restricted_globals['_iter_unpack_sequence_'] = guarded_iter_unpack_sequence
        restricted_globals['_getattr_'] = safer_getattr
        restricted_globals['_getitem_'] = lambda obj, index: obj[index]
        restricted_globals['_getiter_'] = iter
        restricted_globals['_print_'] = PrintCollector

        # Add safe modules
        import re, math, json
        restricted_globals.update({'re': re, 'math': math, 'json': json})

        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = captured_output = io.StringIO()

        try:
            byte_code = compile_restricted_exec(code)
            if byte_code.errors:
                return f"Compilation Error: {', '.join(byte_code.errors)}"

            exec(byte_code.code, restricted_globals, self.env)
            output = captured_output.getvalue()
            
            # Handle prints
            if '_print' in self.env and hasattr(self.env['_print'], 'txt'):
                output += ''.join(self.env['_print'].txt)

            if not output.strip():
                return "Executed successfully (no output)."
            
            return output[:self.max_output_chars]
        except Exception as e:
            return f"Runtime Error: {str(e)}"
        finally:
            sys.stdout = old_stdout

class FileSystemHandler(Handler):
    def handle(self, effect: Effect) -> Any:
        if isinstance(effect, ReadFile):
            with open(effect.payload.path, 'r', encoding='utf-8') as f:
                return f.read()
        elif isinstance(effect, ListFiles):
            import glob
            return [f for f in glob.glob(os.path.join(effect.payload.dir_path, "**"), recursive=True) if os.path.isfile(f)]
        elif isinstance(effect, WriteFile):
            dirname = os.path.dirname(effect.payload.path)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
            with open(effect.payload.path, 'w', encoding='utf-8') as f:
                f.write(effect.payload.content)
            return None
        elif isinstance(effect, GrepFiles):
            import subprocess
            req = effect.payload
            
            # Pattern and path
            pattern = req.pattern
            dir_path = req.dir_path or "."
            file_pattern = req.file_pattern or "*"
            
            if os.name == 'nt':
                # Windows findstr
                # findstr /s /m /i "pattern" dir\*.txt
                cmd = f'findstr /s /m /i "{pattern}" "{os.path.join(dir_path, file_pattern)}"'
                try:
                    # Use shell=True for flexible path/wildcard handling on Windows
                    result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
                    return [line.strip() for line in result.stdout.splitlines() if line.strip()]
                except Exception as e:
                    return [f"Error (findstr): {str(e)}"]
            else:
                # Unix grep
                cmd = ["grep", "-r", "-l", pattern, dir_path]
                if req.file_pattern:
                    cmd.append(f"--include={req.file_pattern}")
                
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    return [line.strip() for line in result.stdout.splitlines() if line.strip()]
                except Exception as e:
                     return [f"Error (grep): {str(e)}"]

        raise NotImplementedError

class MathHandler(Handler):
    def handle(self, effect: Effect) -> Any:
        if isinstance(effect, Math):
            op = effect.payload.op
            a = effect.payload.a
            b = effect.payload.b
            if op == "add": return a + b
            if op == "sub": return a - b
            if op == "mul": return a * b
            if op == "div": return a / b if b != 0 else float('inf')
            raise ValueError(f"Unknown operation: {op}")
        raise NotImplementedError

class UserInteractionHandler(Handler):
    def __init__(self, input_queue: Optional[list] = None):
        self.input_queue = input_queue or []

    def handle(self, effect: Effect) -> Any:
        if isinstance(effect, Listen):
            if self.input_queue:
                return self.input_queue.pop(0)
            # Truly interactive input for REPL usage
            prompt = effect.payload.prompt or "Agent is listening... "
            return input(prompt)
        if isinstance(effect, Reply):
            print(f"\n[Coach]: {effect.payload.message}")
            return "Replied"
        raise NotImplementedError

class MessageBusHandler(Handler):
    def handle(self, effect: Effect) -> Any:
        if isinstance(effect, SendMessage):
            msg = effect.payload
            prefix = "[BROADCAST]" if msg.broadcast else f"[TO: {msg.recipient}]"
            print(f"🚌 {prefix} {msg.content}")
            return "Sent"
        raise NotImplementedError

class RecursiveAgentHandler(Handler):
    """
    Handles the 'Recurse' effect by spawning a new isolated Runtime.
    Dynamically maps natural language queries to agent functions using LLM.
    """
    def __init__(self, model_name: str = "ollama/qwen3:8b"):
        self.model_name = model_name

    def handle(self, effect: Effect) -> Any:
        if isinstance(effect, Recurse):
            task: SubTask = effect.payload
            spec_path = task.spec_path
            query = task.query
            
            print(f"🔄 [RecursiveHandler] Spawning Sub-Agent from '{spec_path}'...")
            print(f"    Goal: {query}")
            
            if not spec_path or not os.path.exists(spec_path):
                 return f"Error: Spec file '{spec_path}' not found."

            try:
                from .compiler import Compiler
                from .runtime import Runtime
                import kernel.semantic_kernel as sk
                
                # 1. Parse the target spec to understand its capabilities
                compiler = Compiler()
                spec_ast = compiler.compile_file(spec_path)
                
                # 2. Use LLM to decide which component/function to call and with what arguments
                # This is a 'Meta-Reasoning' step
                mapping_prompt = f"""
                You are a meta-orchestrator. A user wants to: "{query}"
                Using the following Agent Specification, decide which component and function to call.
                
                SPECIFICATION:
                Name: {spec_ast.name}
                Components: {[(c.name, [f.name for f in c.functions]) for c in spec_ast.components]}
                
                Return a JSON object:
                {{
                  "component": "ComponentName",
                  "function": "FunctionName",
                  "arguments": {{"arg1": val1, ...}}
                }}
                """
                
                response = litellm.completion(
                    model=self.model_name,
                    messages=[{"role": "user", "content": mapping_prompt}],
                    response_format={ "type": "json_object" }
                )
                import json
                decision = json.loads(response.choices[0].message.content)
                
                comp_name = decision['component']
                func_name = decision['function']
                args = decision['arguments']

                print(f"    Selected: {comp_name}.{func_name}({args})")

                # 3. Instantiate Runtime and execute
                sub_runtime = Runtime()
                # Register standard handlers
                sub_runtime.register_handler(LiteLLMHandler(default_model=self.model_name))
                sub_runtime.register_handler(FileSystemHandler())
                sub_runtime.register_handler(MathHandler())
                
                parent_runtime = sk._active_runtime
                sk._active_runtime = sub_runtime
                
                try:
                    module_path = os.path.join("src", f"{comp_name.lower()}.py")
                    if not os.path.exists(module_path):
                        return f"Error: Implementation for {comp_name} not found at {module_path}. Build it first."

                    spec = importlib.util.spec_from_file_location(comp_name, module_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    cls = getattr(module, comp_name)
                    agent_instance = cls()
                    
                    method = getattr(agent_instance, func_name)
                    result = method(**args)
                    return str(result)
                    
                finally:
                    sk._active_runtime = parent_runtime

            except Exception as e:
                return f"Sub-Agent Execution Failed: {e}"
            
        raise NotImplementedError

class ReasoningHandler(Handler):
    def handle(self, effect: Effect) -> Any:
        if isinstance(effect, ReasoningTrace):
            log = effect.payload
            # In a real system, this would write to a structured audit log (TraceIR)
            print(f"🧠 [Trace] Thought: {log.thought}")
            if log.plan:
                print(f"   [Trace] Plan: {log.plan}")
            return None
        raise NotImplementedError

class MockLLMHandler(Handler):
    def handle(self, effect: Effect) -> Any:
        if isinstance(effect, Generate):
            # Echo the last message content
            if effect.payload.messages:
                last_msg = effect.payload.messages[-1]['content']
                return f"Mock LLM Response: {last_msg[-50:]}"
            return "Mock Response"
        raise NotImplementedError
