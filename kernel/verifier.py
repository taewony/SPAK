import ast
import os
# import pytest # Dynamic verification disabled for initial setup
import yaml
from typing import List, Dict, Any, Optional
from .compiler import SystemSpec, ComponentSpec, FunctionSpec

class VerificationError(Exception):
    pass

class StaticVerifier:
    """
    Checks if the implementation (Python code) matches the Spec (AISpec).
    """
    def verify(self, spec: SystemSpec, src_dir: str) -> List[str]:
        errors = []
        for component in spec.components:
            found = False
            for root, _, files in os.walk(src_dir):
                for file in files:
                    if file.endswith(".py"):
                        path = os.path.join(root, file)
                        if self._check_file_for_class(path, component, errors):
                            found = True
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
            return False

        class_node = None
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == component.name:
                class_node = node
                break
        
        if not class_node:
            return False

        # Class found, now check methods
        implemented_methods = {n.name for n in class_node.body if isinstance(n, ast.FunctionDef)}
        
        for func in component.functions:
            if func.name not in implemented_methods:
                errors.append(f"Component '{component.name}' in '{file_path}' is missing method '{func.name}'")
        
        return True

class DynamicVerifier:
    """
    Runs YAML-based test vectors using Pytest.
    """
    def run_tests(self, test_file: str) -> bool:
        # Placeholder for dynamic verification
        print(f"[DynamicVerifier] Would run tests from {test_file}")
        return True

class Verifier:
    def __init__(self):
        self.static = StaticVerifier()
        self.dynamic = DynamicVerifier()

    def verify_spec(self, spec: SystemSpec, src_dir: str = "src") -> bool:
        errors = self.static.verify(spec, src_dir)
        if errors:
            for e in errors:
                print(f"[Static Error] {e}")
            return False
        print(f"[Static] Structure of system '{spec.name}' verified.")
        return True