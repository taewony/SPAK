from kernel.compiler import Compiler

def test_system_model_parsing():
    spec_content = """
meta {
    name = "TestAgent"
}

system_model TestModel {
    axiom: "This is an axiom."
    heuristic: "This is a heuristic."
    prediction: "This is a prediction."
}

system TestSystem {
    component TestComponent {
        description: "Test description"
    }
}
    """
    compiler = Compiler()
    system_spec = compiler.compile(spec_content)
    
    print(f"Agent Name: {system_spec.metadata.get('name')}")
    if system_spec.system_model:
        print(f"Model Name: {system_spec.system_model.name}")
        for stmt in system_spec.system_model.statements:
            print(f"  [{stmt.type}] {stmt.content}")
    else:
        print("No system model found!")

if __name__ == "__main__":
    test_system_model_parsing()
