import sys
import os

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kernel.dsl.parser import DSLParser
from kernel.dsl.ast import Task, ToolStep, LLMStep

def test_parse_hello():
    parser = DSLParser()
    spec = parser.parse_file("examples/hello.dsl")
    task = spec.task
    
    print(f"Parsed Task: {task.name}")
    assert task.name == "HelloSPAK"
    assert len(task.steps) == 3
    
    s1 = task.steps[0]
    assert isinstance(s1, ToolStep)
    assert s1.id == "s1"
    assert s1.params['cmd'] == "echo 'Hello from Shell'"
    assert s1.output_var == "shell_out"
    
    s2 = task.steps[1]
    assert isinstance(s2, LLMStep)
    assert s2.id == "s2"
    assert s2.role == "greeter"
    assert s2.output_var == "greeting"
    
    print("SUCCESS: DSL Parsing verified.")

if __name__ == "__main__":
    test_parse_hello()
