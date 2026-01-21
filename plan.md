# Trace & Logging Strategy and Project Roadmap

This document outlines the logging strategy for the Agent Runtime, the theoretical "Control Loop Model" for Systematic Intelligence Engineering, and the strategic roadmap for evolving the SPAK Kernel.

## 1. Trace Command Usage

The CLI provides the `trace` command to inspect the execution history of the most recent agent run.

*   **`trace`**: Displays **only** the high-level cognitive trace. This includes the Agent's "Thought" and "Plan". It filters out low-level system calls like LLM generation, file I/O, or raw inputs to focus on the reasoning process.
*   **`trace all`**: Displays **everything**. This includes `ReasoningTrace`, `Generate` (LLM inputs/outputs), `Listen`, `Reply`, and all other system effects. Use this for deep debugging of context windows or raw IO.

## 2. Log Types

The Runtime captures several types of effects. The `trace` command filters these based on relevance:

### High-Level (Visible in `trace`)
*   **`ReasoningTrace`**: The primary artifact of the `think` process.
    *   **Thought**: The natural language reasoning (Why am I doing this?).
    *   **Plan**: The structured JSON-like object describing the intended action (What am I going to do?).

### Low-Level (Visible in `trace all`)
*   **`Generate`**: The raw request to the LLM. Contains the full message history and the raw text response.
*   **`Listen` / `Reply`**: User interaction events.
*   **`ReadFile` / `WriteFile`**: File system operations.

## 3. Think vs. Plan vs. Revise Strategy

To ensure clarity in Agent behavior and specification, we distinguish these concepts as follows:

### A. Think (`ReasoningTrace.thought`)
*   **Definition**: The latent reasoning process where the Agent analyzes the situation, context, and goal.
*   **Implementation**: This is the narrative part of the log.
*   **Role**: **Audit**. Thought explains *why*, but does not execute. It can be wrong without crashing the system, but must align with the Plan for verification.

### B. Plan (`ReasoningTrace.plan`)
*   **Definition**: The explicit, structured output of the thinking process. This determines the control flow.
*   **Implementation**: A dictionary/object within the log.
*   **Role**: **Control**. The Plan *must* be correct because it drives the execution.

### C. Revise (Plan Update)
*   **Definition**: A specific type of planning action where the *existing* plan is modified.
*   **Logging Strategy**:
    *   When an Agent decides to change its course, it **MUST** emit a `ReasoningTrace` before taking action.
    *   The `plan` field should explicitly indicate a revision.
    *   **Verification**: This allows RTCT (Round-Trip Consistency Test) to verify that a "Revise" thought actually leads to a change in the Agent's internal state or behavior.

---

## 4. Theoretical Model: The Cognitive Control Loop

The SPAK Agent architecture is fundamentally a **Control Loop** applied to Intelligence.

| Control Theory | Agent Function | LLM Role |
| :--- | :--- | :--- |
| **State Estimation** | `think` | **Cognitive State Estimation**: Where am I? What is the user intent? |
| **Policy Selection** | `plan` | **Decision Making**: Selects the next Domain Operation. |
| **Actuation** | `perform` | **Execution**: System Effects (IO, Tools). |
| **Policy Update** | `revise` | **Learning/Correction**: Modifying the strategy based on outcome. |

**The Core RTCT Requirement:**
> Latent (Thought) $\to$ Symbolic (Plan) $\to$ Executable (Effect) $\to$ Outcome

---

## 5. Architectural Strategy: Meta-IR vs. Domain-IR

We explicitly reject the idea of a "Single Universal PlanIR" for all agents. Instead, we adopt a layered approach common in compilers and robotics.

### Layer 1: Meta-IR (The Control Flow Grammar)
Universal across all agents (Coach, Analyst, Coder).
*   **Structure:** Sequential, Branching, Revision, Termination.
*   **Semantics:** "If X happens, switch state to Y."

### Layer 2: Domain-IR (The Semantic Operators)
Specific to the agent's domain.
*   **CoachingAgent:** `ask`, `revise_goal`, `suggest_action`.
*   **ResearchAnalyst:** `retrieve`, `cluster`, `outline`, `synthesize`.
*   **PersonaChatBot:** `adopt_persona`, `reply_in_character`.

**Engineering Goal:** The `Builder` must synthesize the *Domain-IR* from the Spec, while the `Runtime` enforces the *Meta-IR*.

---

## 6. Strategic Roadmap: Systematic Intelligence Engineering

### Phase 1: Instrumentation & Grounding (Completed)
*   **Focus:** Level 0-3 (Static to Planning Agents).
*   **Achievement:** Established the "Think-Plan-Act" loop and RTCT verification.
*   **Key Insight:** "LLM is a syscall provider; Cognition is a symbolic control problem."

### Phase 2: Domain Specification & Composition (Current Focus)
*   **Focus:** Level 4 (Multi-Agent) and `ResearchAnalyst`.
*   **Goal:** Define the **Domain-IR** for complex agents.
    *   **TODO:** Explicitly list `domain_operations` in Agent Specs.
    *   **TODO:** Update `Builder` to auto-synthesize Python orchestration code from Spec `workflow {}` blocks.
    *   **TODO:** Verify `MessageBus` effectiveness in decoupling components (`Librarian` vs `Analyst`).

### Phase 3: Recursive & Fractal Intelligence (Next)
*   **Focus:** Level 5 (Recursive Solvers).
*   **Goal:** Recursion as a precise effect.
    *   **TODO:** Prove that a Child Agent's memory does not pollute the Parent's context window (Context Efficiency).
    *   **TODO:** Implement `Recurse` RTCT: Verify that the Parent correctly interpreted the Child's output.

### Phase 4: Semantic Invariant Testing (Advanced RTCT)
*   **Focus:** Moving from "Structural Match" to "Semantic Equivalence".
*   **Goal:** Verify that the *Predicted Effect* matches the *Actual System Effect*.
    *   **Example (Coach):** Does the `suggest_action` actually reduce the semantic distance to the `goal`?
    *   **Example (Coder):** Does `optimize_kernel` actually monotonically increase performance?
    *   **Role of Meta-LLM:** This phase enables the Meta-LLM to analyze trace failure patterns and optimize the agent's code/prompts (Self-Improvement). **We do not introduce Meta-LLM optimization until this instrumentation phase is mature.**
