# SPAK-POC: Step-wise Dual Loop Agent System

## 0. Vision
To build a **Deterministic Agent Kernel** where execution is governed by an explicit **DSL (Domain Specific Language)**, and LLM reasoning is isolated to specific interaction points (`llm.query`). This architecture separates the "System Model" (defined in DSL) from the "Latent Simulation" (performed by LLM), enabling robust debugging, step-wise execution, and self-improvement.

## 1. Architecture

### 1.1 The Components
1.  **DSL (The Contract)**:
    *   Defines `Task`, `Step`, `Tool`, `LLM Interaction`, and `Invariants`.
    *   Serves as the "Intermediate Representation" (IR) between the Outer Loop and the Kernel.
2.  **Kernel (The Engine)**:
    *   **Step Machine**: A Python VM that executes the DSL step-by-step.
    *   **Trace Logger**: Records every state change, prompt, and result to `trace.json`.
    *   **Suspend/Resume**: Critical for the "Gemini CLI Simulation". Allows the kernel to pause at `llm.query`, wait for input (simulation), and resume.
3.  **Outer Loop (The Improver)**:
    *   Analyzes `trace.json`.
    *   Refines the DSL (Grammar or Script) to optimize agent behavior.

### 1.2 The LLLM Simulation (Latent Logic / Large Language Model Simulation)
*   **Goal**: To verify agent logic without non-deterministic API calls.
*   **Mechanism**:
    *   The Kernel halts at `llm.query`.
    *   The "Latent Space" is simulated by the developer (via CLI) or a mock script.
    *   The "Symbolic Space" (Kernel) resumes processing the simulated response.

## 2. Roadmap

### Phase 1: Core Kernel & DSL Foundation (The "Body")
*   [x] **DSL Grammar Definition (`grammar.lark`)**:
    *   Define syntax for `task`, `step`, `tool.run`, `llm.query`.
    *   Define `system_model` blocks for constraints.
    *   Define `evaluation` blocks for metrics.
*   [x] **AST & Compiler**:
    *   Implement parser to convert `.dsl` text into Python AST objects.
*   [x] **Step Machine Skeleton**:
    *   Create the main loop that iterates through AST nodes.
    *   Implement `ExecutionContext` for variable storage.

### Phase 2: Simulation & Interaction (The "Brain" Interface)
*   [x] **LLM Step & Suspend Logic**:
    *   Implement `llm.query` handling: Render prompt -> Save State -> Exit/Suspend.
*   [x] **Gemini CLI Simulation Mode**:
    *   CLI command to resume execution (`spak step --resume`).
    *   Input mechanism for simulated LLM responses (e.g., reading from `response.txt`).
*   [x] **Trace Logging**:
    *   Implement structured JSON logging for replayability.
    *   Implement `metrics.json` for quantitative evaluation.

### Phase 3: Execution & Tools (The "Hands")
*   [x] **Variable Binding**:
    *   Allow outputs from tools to be stored in variables (`output_var`).
    *   Allow variables to be used in prompts (`{{var}}`).
*   [x] **Real Tool Dispatcher**:
    *   Implemented `tool.read` and `tool.write` for file I/O.
    *   Implemented `tool.run` with `--live` flag for real subprocess execution.
*   [x] **Real Backend Integration**:
    *   Implemented `OllamaBackend` (localhost:11434).

### Phase 4: MatMul Real-World Verification (Claim 1: Reliability)
*   **Objective**: Validate `matmul_improved_spak.py` on actual GPU hardware (RTX 5070/3090).
*   **Tasks**:
    1.  [ ] **Execution**: Run the generated code on a GPU node using `python spak_v2.py run ... --live`.
    2.  [ ] **Validation**: Confirm `np.allclose(SPAK_Output, PyTorch_Output)` passes.
    3.  [ ] **Benchmarking**: Record speedup ($T_{pytorch} / T_{spak}$) across multiple matrix sizes (4096, 8192, 16384).
    4.  [ ] **Data Collection**: Secure `trace.json` and execution logs as **Experiment 1 Back Data**.

### Phase 5: Batch & FMHA Engineering (Claim 2: Complexity & Generalization)
*   **BatchMatMul (BMM)**:
    *   Generalize 2D optimizations to 3D tensors.
    *   Verify Correctness/Monotonic Speedup invariants.
*   **Fused Multi-Head Attention (FMHA)**:
    *   Implement Online Softmax & Fusion.
    *   Prove capability to handle complex, non-linear kernels.

## 3. Current Focus: Phase 4 (GPU Execution)
*   **Immediate Action**: Deploy the codebase to the GPU server and run the live benchmark.
