import os
import io
import sys
import litellm
from typing import Dict, Any, Optional
from .semantic_kernel import Handler, Effect
from .effects import Generate, ExecuteCode, ReadFile, WriteFile, Recurse, LLMRequest

# Safe Execution Imports (from recursive-llm wisdom)
from RestrictedPython import compile_restricted_exec, safe_globals, limited_builtins, utility_builtins
from RestrictedPython.Guards import guarded_iter_unpack_sequence, safer_getattr
from RestrictedPython.PrintCollector import PrintCollector

class LiteLLMHandler(Handler):
    def __init__(self, default_model: str = "qwen2.5:3b"):
        self.default_model = default_model

    def handle(self, effect: Effect) -> Any:
        if isinstance(effect, Generate):
            req: LLMRequest = effect.payload
            response = litellm.completion(
                model=req.model or self.default_model,
                messages=req.messages,
                stop=req.stop
            )
            return response.choices[0].message.content
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
        elif isinstance(effect, WriteFile):
            os.makedirs(os.path.dirname(effect.payload.path), exist_ok=True)
            with open(effect.payload.path, 'w', encoding='utf-8') as f:
                f.write(effect.payload.content)
            return None
        raise NotImplementedError
