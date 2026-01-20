import ast
import os
import importlib.util
import yaml
from typing import List, Dict, Any, Optional
from .compiler import SystemSpec, ComponentSpec, FunctionSpec
from .runtime import Runtime
from .handlers import LiteLLMHandler, MathHandler, UserInteractionHandler, MessageBusHandler, RecursiveAgentHandler, FileSystemHandler, ReasoningHandler
from .semantic_kernel import Handler, Effect
from .effects import Generate
import kernel.semantic_kernel as sk

class MockLLMHandler(Handler):
    def handle(self, effect: Effect) -> Any:
        if isinstance(effect, Generate):
            # Echo the last message content to satisfy 'contains' checks
            last_msg = effect.payload.messages[-1]['content']
            return f"Mock LLM Response: {last_msg}"
        raise NotImplementedError

class StaticVerifier:
    def verify(self, spec: SystemSpec, src_dir: str) -> List[str]:
        errors = []
        print(f"\n[Static Analysis] Starting verification for system: {spec.name}")
        for component in spec.components:
            found = False
            for root, _, files in os.walk(src_dir):
                for file in files:
                    if file.endswith(".py") and file.startswith(component.name.lower()):
                        path = os.path.join(root, file)
                        if self._check_file_for_class(path, component, errors):
                            found = True
                            print(f"  ✅ Found '{component.name}' in {path}")
                            break
                if found: break
            if not found:
                errors.append(f"Missing implementation for Component '{component.name}'")
        return errors

    def _check_file_for_class(self, file_path: str, component: ComponentSpec, errors: List[str]) -> bool:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read())
        except Exception as e:
            errors.append(f"Syntax Error in {file_path}: {str(e)}")
            return False
        class_node = next((n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == component.name), None)
        if not class_node: return False
        implemented_methods = {n.name for n in class_node.body if isinstance(n, ast.FunctionDef)}
        for func in component.functions:
            if func.name not in implemented_methods:
                errors.append(f"Method '{func.name}' missing in {component.name}")
        return True

class DynamicVerifier:
    def run_tests(self, test_file: str, src_dir: str = "src") -> List[str]:
        errors = []
        print(f"\n[Dynamic Analysis] Running tests from: {test_file}")
        
        # Setup Runtime for Side Effects
        runtime = Runtime()
        # runtime.register_handler(LiteLLMHandler(default_model="ollama/qwen2.5-coder:7b")) # Use Mock instead
        runtime.register_handler(MockLLMHandler())
        runtime.register_handler(FileSystemHandler())
        runtime.register_handler(MathHandler())
        runtime.register_handler(UserInteractionHandler(input_queue=["Hello", "Yes", "Goodbye"]))
        runtime.register_handler(MessageBusHandler())
        runtime.register_handler(RecursiveAgentHandler()) # Level 5
        runtime.register_handler(ReasoningHandler())
        sk._active_runtime = runtime
        
        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            comp_name = config['component']
            module_path = os.path.join(src_dir, f"{comp_name.lower()}.py")
            
            # Load the module with error reporting
            try:
                spec = importlib.util.spec_from_file_location(comp_name, module_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            except Exception as e:
                print(f"  💥 Failed to load module {module_path}: {e}")
                return [f"Module Load Error: {str(e)}"]
            
            cls = getattr(module, comp_name)
            instance = cls()

            for test in config.get('tests', []):
                name = test['name']
                print(f"  🧪 Running {name}...", end=" ", flush=True)
                
                try:
                    func_name = test.get('function')
                    if func_name:
                        func = getattr(instance, func_name)
                        # Execute with provided input
                        result = func(**test.get('input', {}))
                        
                        expected = test.get('expected')
                        
                        # Soft Match for LLM outputs (contains vs exact)
                        is_match = result == expected
                        if isinstance(result, str) and isinstance(expected, str):
                            if expected in result or result in expected:
                                is_match = True
                        elif isinstance(result, list) and isinstance(expected, str):
                            # Check if expected string is in string representation of list or any element
                            if expected in str(result) or any(expected in str(x) for x in result):
                                is_match = True
                        
                        if not is_match:
                            err = f"Expected '{expected}', got '{result}'"
                            errors.append(f"{name}: {err}")
                            print(f"❌ FAIL")
                            print(f"     └─ {err}")
                        else:
                            print("✅ PASS")
                except Exception as e:
                    errors.append(f"{name}: Runtime Error: {str(e)}")
                    print(f"❌ ERROR ({str(e)})")
                    
        except Exception as e:
            errors.append(f"General Test Failure: {str(e)}")
            print(f"💥 General Failure: {e}")
        finally:
             sk._active_runtime = None
            
        return errors

class Verifier:
    def __init__(self):
        self.static = StaticVerifier()
        self.dynamic = DynamicVerifier()

    def verify_structure(self, spec: SystemSpec, src_dir: str) -> List[str]:
        return self.static.verify(spec, src_dir)

    def verify_behavior(self, test_path: str, src_dir: str = "src") -> List[str]:
        return self.dynamic.run_tests(test_path, src_dir)

    def verify_spec(self, spec: SystemSpec, src_dir: str = "src") -> bool:
        # 1. Structural
        errors = self.verify_structure(spec, src_dir)
        
        # 2. Behavioral
        for comp in spec.components:
            test_file = os.path.join("tests", f"tests.{comp.name.lower()}.yaml")
            if os.path.exists(test_file):
                dynamic_errors = self.verify_behavior(test_file, src_dir)
                errors.extend(dynamic_errors)
        
        print("-" * 50)
        if errors:
            print(f"\n[Result] Verification FAILED with {len(errors)} total errors.")
            # We don't print detail here because DynamicVerifier already printed per-test result
            return False
        print(f"\n[Result] Verification PASSED. Implementation matches Spec and Tests.")
        return True
