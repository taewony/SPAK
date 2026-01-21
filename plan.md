# Trace & Logging Strategy

This document outlines the logging strategy for the Agent Runtime, specifically focusing on the distinction between cognitive processes (`think`), structured planning (`plan`), and raw system execution.

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
*   **Example**: "The user wants to change the topic. I should update the plan to reflect this new interest."

### B. Plan (`ReasoningTrace.plan`)
*   **Definition**: The explicit, structured output of the thinking process. This determines the control flow.
*   **Implementation**: A dictionary/object within the log.
*   **Example**: `{"action": "update_topic", "target": "coding"}`

### C. Revise (Plan Update)
*   **Definition**: A specific type of planning action where the *existing* plan is modified.
*   **Logging Strategy**:
    *   When an Agent decides to change its course, it **MUST** emit a `ReasoningTrace` before taking action.
    *   The `plan` field should explicitly indicate a revision.
    *   **Suggested Structure**:
        ```python
        perform(ReasoningTrace(TraceLog(
            thought="The previous plan is no longer valid because the user declined the offer. I need to propose an alternative.",
            plan={
                "action": "revise",
                "reason": "user_decline",
                "old_strategy": "direct_sale",
                "new_strategy": "consultative_approach"
            }
        )))
        ```
    *   **Verification**: This allows RTCT (Round-Trip Consistency Test) to verify that a "Revise" thought actually leads to a change in the Agent's internal state or behavior.

## 4. Implementation Checklist

- [x] CLI `trace` command filters for `ReasoningTrace`.
- [x] CLI `trace all` command shows full history.
- [ ] Agents implement `think` loops that emit `ReasoningTrace` before acting.
- [ ] Agents use structured `plan` payloads (especially for `revise` actions) to enable automated consistency checking.
