# SPAK v2: Neuro-Symbolic Agent Engine (Current Implementation)

**Status**: ✅ Functional Neuro-Symbolic Kernel (v2.0)
**Date**: 2026-01-28

The SPAK v2 Kernel is now a fully functional **Neuro-Symbolic Agent Engine**. It successfully offloads **"Latent Reasoning"** to a local LLM (Ollama) while maintaining the **"Symbolic Structure"** of the DSL.

---

## 0. Design Goals (Achieved)

| Metric | Status | Implementation |
| :--- | :--- | :--- |
| **Execution** | ✅ Done | Python VM (`StepMachine`) + Lark Parser |
| **Reasoning** | ✅ Done | Ollama (`qwen2.5:7b`) & Manual Simulation |
| **Control** | ✅ Done | Explicit DSL with `llm.query` & `tool.run` |
| **Planning** | ✅ Done | Inner Loop Planning (LLM selects tools dynamically) |
| **Evaluation** | ✅ Done | Quantitative Metrics via `evaluation` block |

---

# 1. System Architecture

```
┌──────────────────────────────────────────────┐
│                OUTER LOOP (Todo)             │
│  - Analyzes metrics.json & trace.json        │
│  - Optimizes DSL structure (Self-Improvement)│
└───────────────▲──────────────────────────────┘
                │ Feedback (Metrics)
                │
┌───────────────┴──────────────────────────────┐
│              SPAK KERNEL v2 (Python)         │
│  - Entry: spak_v2.py                         │
│  - Parser: Lark based DSL Compiler           │
│  - VM: StepMachine with Suspend/Resume       │
│  - Memory: Context (Variables) + Trace Log   │
└───────────────▲──────────────────────────────┘
                │
                │ 1. Render Prompt
                │ 2. Bind Variables ({{var}})
                │
┌───────────────┴──────────────────────────────┐
│              LLM BACKENDS                    │
│  - Mode A: Simulation (Manual 'response.txt')│
│  - Mode B: Ollama (Local 'qwen2.5:7b')       │
└──────────────────────────────────────────────┘
```

---

# 2. DSL Specification (v2)

The DSL now supports **System Models**, **Tasks**, **Steps**, and **Evaluation**.

## 2.1 Core Structure

```dsl
system_model SysAdminAgent {
  axiom: "Availability is the highest priority."
  heuristic: "Investigate before taking destructive actions."
}

task IncidentResponse {
  step alert: tool.run {
    cmd: "echo 'ALERT: High Latency'"
    output_var: alert_ctx
  }

  step planner: llm.query {
    role: "SRE"
    prompt_template: "Context: {{alert_ctx}}. Decide next command."
    output_var: next_cmd
  }

  step executor: tool.run {
    cmd: "{{next_cmd}}"  # Dynamic Tool Execution
    output_var: result
  }

  evaluation {
    check heuristic_compliance: llm.query {
      role: "Judge"
      prompt_template: "Did {{next_cmd}} follow the heuristic?"
      output_var: score
    }
  }
}
```

## 2.2 Execution Flow

1.  **Parse**: `DSLParser` compiles `.dsl` to AST.
2.  **Initialize**: `StepMachine` creates `ExecutionContext`.
3.  **Run Loop**:
    *   **Tool Step**: Executes shell command, captures output.
    *   **LLM Step**:
        *   If `backend="sim"`: Suspends, waits for `response.txt`.
        *   If `backend="ollama"`: Calls API, resumes automatically.
4.  **Evaluate**: After task completion, runs `evaluation` block to generate `metrics.json`.

---

# 3. Kernel Implementation Details

## 3.1 File Structure

```
D:\code\SPAK\
├── spak_v2.py              # CLI Entry Point
├── kernel/
│   ├── dsl/
│   │   ├── grammar.lark    # Strict Grammar Definition
│   │   ├── parser.py       # AST Transformer
│   │   └── ast.py          # Data Classes
│   └── vm.py               # The Neuro-Symbolic Engine
├── examples/
│   ├── sysadmin_agent.dsl  # Planning Demo
│   └── eval_test.dsl       # Evaluation Demo
└── trace.json              # Execution Log
```

## 3.2 Key Features Implemented

*   **Variable Binding**: `{{variable}}` syntax allows data to flow between Tools (Symbolic) and LLM (Latent).
*   **Inner Loop Planning**: The LLM can output a command string that the Kernel executes in the next step, enabling "Agentic" behavior within a deterministic framework.
*   **Quantitative Evaluation**: Integrated `evaluation` block allows "LLM-as-a-Judge" to score runs automatically.

---

# 4. Roadmap & Status

### Phase 1: Core Kernel (Completed ✅)
*   Lark Grammar defined.
*   AST & Compiler implemented.
*   Step Machine (VM) functional.

### Phase 2: Simulation & Interaction (Completed ✅)
*   Suspend/Resume logic for manual debugging.
*   JSON structured logging (`trace.json`, `context.json`).

### Phase 3: Backend & Planning (Completed ✅)
*   **Ollama Integration**: Fully autonomous local execution.
*   **Dynamic Planning**: Agents can select tools at runtime.
*   **Metrics**: `metrics.json` generation.

### Phase 4: Reliability & Optimization (Next 🚀)
*   **Batch Benchmarking**: Run DSLs $N$ times to measure statistical reliability.
*   **Outer Loop**: Implement the "Optimizer" that reads `metrics.json` and patches the DSL.