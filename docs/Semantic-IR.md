# Part 1: Semantic IR Definition (AISpec)

This document defines the **Semantic Intermediate Representation (AISpec)**, a formal specification system for AI Software Engineering. It adopts a "Nix-style" approach where **Spec (Design)** and **Test (Verification)** are decoupled from implementation language but drive the entire build process.

## 1. Philosophy: The "Spec-Driven" Architecture

In this system, we do not write code first. We write **Specs** and **Tests**.
-   **Formal Spec (AISpec)**: A DSL defining *Structure*, *Interfaces*, *Constraints*, and *Invariants*.
-   **Test Vectors (YAML)**: Data defining *Input*, *Expected Output*, and *State Transitions*.
-   **Verification**: An automated process (Pytest + LLM) that checks if an Implementation (Python/Agent) satisfies the Spec and passes the Tests.

## 2. File System Hierarchy (The "Nix" Style)

The file system structure itself represents the system architecture.

```text
/specs
  SPEC.root.md                 # Root system definition
  SPEC.root.agent.md           # Agent subsystem
  SPEC.root.agent.memory.md    # Memory component
/tests
  tests.root.yaml              # High-level integration tests
  tests.root.agent.yaml        # Agent behavioral tests
  tests.root.agent.memory.yaml # Memory unit tests
```

## 3. AISpec Grammar (Lark)

The content of `SPEC.*.md` files is parsed using this Lark grammar. It allows for defining systems, components, functions, and formal constraints.

```lark
// AISpec: A DSL for Formal System Specification
start: system_def

system_def: "system" NAME "{" component_def* "}"

component_def: 
    | "component" NAME "{" member_def* "}"
    | "import" NAME

member_def:
    | "description" ":" STRING ";"
    | "state" NAME "{" field_list "}"
    | "function" NAME "(" param_list ")" "->" type_ref ";"
    | "invariant" ":" logic_expr ";"
    | "constraint" ":" logic_expr ";"

field_list: (NAME ":" type_ref)*
param_list: (NAME ":" type_ref ("," NAME ":" type_ref)*)?

type_ref: NAME | "List" "[" type_ref "]" | "Map" "[" type_ref "," type_ref "]"

logic_expr: /[^\;]+/  // Simplified for prototype: raw logic string

NAME: /[a-zA-Z_]\w*/
STRING: /"[^"]*"/
%import common.WS
%ignore WS
```

### Example Spec (`SPEC.agent.memory.md`)
```aispec
system AgentMemory {
    component ContextTree {
        description: "Hierarchical storage for infinite context";
        
        state Node {
            id: String
            content: String
            children: List[Node]
        }

        invariant: "Node ID must be unique";
        invariant: "Tree depth must not exceed max_depth";

        function add_context(content: String, parent_id: String) -> Node;
        function get_context(selector: String) -> String;
    }
}
```

## 4. Test Vector Specification (YAML)

Tests are language-agnostic. They define **What** to test, not **How**.

### Example Test (`tests.agent.memory.yaml`)
```yaml
system: AgentMemory
component: ContextTree
tests:
  - name: test_add_root_context
    function: add_context
    input:
      content: "Initial System Prompt"
      parent_id: "root"
    expected:
      state_change: 
        node_count: 1
      return:
        id: "node_0"
        
  - name: test_depth_invariant
    function: add_context
    setup:
      # Pre-condition: Create a deep tree
      depth: 10
    input:
      content: "Too deep"
      parent_id: "leaf_node"
    expected:
      error: "InvariantViolation: Tree depth must not exceed max_depth"
```

## 5. Verification Process (The "Build" Loop)

The Build-Agent does not just "write code". It performs a **Verified Lift**:

1.  **Parse**: Load `SPEC.*.md` -> AST.
2.  **Load Tests**: Load `tests.*.yaml`.
3.  **Check**:
    *   **Structural Consistency**: Does the Python class `ContextTree` exist? Does it have method `add_context`? (Checked via AST analysis).
    *   **Behavioral Correctness**: Run `pytest` using the YAML vectors against the Python implementation.
4.  **Refine**: If verification fails, the Build-Agent reads the error (Pytest output) and the Spec, then rewrites the Python code.

## 6. REPL Integration

The system includes a specialized REPL for managing this lifecycle.

```python
# pseudo-code for REPL commands
> load_spec SPEC.agent.memory.md
[System] Spec loaded. Defined: ContextTree

> verify_structure
[Check] Class 'ContextTree' ... OK
[Check] Method 'add_context' ... OK
[Check] Invariant 'Node ID unique' ... WARNING (No explicit check found in code)

> run_tests tests.agent.memory.yaml
[Pytest] test_add_root_context ... PASS
[Pytest] test_depth_invariant ... FAIL
   -> Detail: RecursiveError not raised.

> implement --fix
[Agent] Reading failure... Generating fix for depth check...
[Agent] Code updated.

> run_tests
[Pytest] All passed.
```