# Design Specification: Spec-Driven Build Agent System

## 1. Overview

This system is a **Spec-Driven Development Environment**. It reverses the traditional workflow: instead of writing code and then docs, we write **Formal Specifications (AISpec)** and **Test Vectors** first. The Build-Agent then acts as a "Compiler" that transforms these specs into verified Python code.

## 2. System Architecture

### 2.1 The Specification Layer (`/specs`)
- **Format**: `AISpec` (Lark DSL)
- **Role**: Defines the "Truth". If the code disagrees with the Spec, the code is wrong.
- **Structure**: Recursive file hierarchy (`SPEC.system.subsystem.md`).

### 2.2 The Verification Layer (`/tests`)
- **Format**: YAML
- **Role**: Defines the "Behavior". Language-agnostic input/output pairs.
- **Tools**: `pytest` adapter that reads YAML and executes Python functions.

### 2.3 The Runtime Layer (`src/`)
- **Format**: Python
- **Role**: Implementation details.
- **Constraint**: Must strictly adhere to interfaces defined in Specs.

### 2.4 The Build-Agent (The "Kernel")
The active process that mediates between Spec, Test, and Code.

#### Core Loop:
1.  **Monitor**: Watch for changes in `/specs` or `/tests`.
2.  **Parse**: Convert AISpec to AST.
3.  **Verify Structure**: Check if Python classes/functions match the AST signature.
4.  **Verify Behavior**: Run `pytest` with YAML data.
5.  **Act**:
    -   If code is missing: **Generate** skeleton.
    -   If tests fail: **Repair** code (using LLM).
    -   If spec changes: **Refactor** code.

## 3. Component Design

### 3.1 `compiler.py` (The Parser)
-   **Input**: `SPEC.*.md` content.
-   **Output**: Python objects (`SystemSpec`, `ComponentSpec`, `FunctionSpec`).
-   **Tech**: `lark-parser`.

### 3.2 `verifier.py` (The Auditor)
-   **Static Check**: Uses `ast` module to inspect Python code. compares it against `Spec` objects.
    -   *Error*: "Class `ContextTree` missing method `get_context`".
-   **Dynamic Check**: Generates temporary `test_*.py` files from `tests.*.yaml` and runs `pytest`.

### 3.3 `builder.py` (The Coder)
-   **Role**: Interface to LLM (Ollama).
-   **Prompt Strategy**:
    -   "You are implementing a spec."
    -   Input: Spec AST + Failed Test Output.
    -   Output: Python Code.

### 3.4 `repl.py` (The UI)
-   Interactive shell for the developer.
-   Commands:
    -   `load <spec_file>`
    -   `verify`
    -   `implement`
    -   `test`

## 4. Implementation Roadmap

### Phase 1: The Core (Current Goal)
-   Implement `compiler.py` to parse AISpec.
-   Implement `verifier.py` to run structural checks.
-   Implement `repl.py` to expose these tools.

### Phase 2: The Agent
-   Connect `builder.py` to Ollama.
-   Enable "Auto-Implement" loop.

### Phase 3: Recursive Self-Build
-   Write `SPEC.build_agent.md`.
-   Have the system rebuild itself.

## 5. Example Workflow

1.  **User**: Creates `specs/SPEC.math.md`
    ```aispec
    system Math {
        component Calculator {
            function add(a: Int, b: Int) -> Int;
        }
    }
    ```
2.  **User**: Creates `tests/tests.math.yaml`
    ```yaml
    system: Math
    component: Calculator
    tests:
      - function: add
        input: {a: 1, b: 2}
        expected: 3
    ```
3.  **REPL**: `> load specs/SPEC.math.md`
4.  **REPL**: `> implement`
    -   Agent sees missing `src/math.py`.
    -   Agent generates class `Calculator` with method `add`.
5.  **REPL**: `> verify`
    -   Agent runs tests. Returns PASS.