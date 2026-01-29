# SPAK (Semantic Programmable Agent Kernel)

**The Semantic Programmable Agent Kernel (SPAK)** is a bi-level optimization framework designed to bridge the gap between high-level human intent and low-level execution logic. It operates as a **dual-loop system** where a **Semi-formal DSL (Domain Specific Language)** serves as the evolving "contract" between a strategic reasoning engine (Outer Loop) and a deterministic execution environment (Inner Loop/Kernel).

Unlike traditional agentic workflows that rely on probabilistic prompt chaining, SPAK employs an **abductive reasoning process**: it observes execution traces to infer the optimal system architecture, refining the DSL iteratively until the system behavior aligns with the engineering goals.

### Core Architecture & Flow

1. **Outer Loop (The Strategist):** The LLM analyzes the **Agent Trace Log** from previous runs. Using **abductive reasoning**, it hypothesizes necessary structural changes (e.g., changing a sync process to async) and refines the **DSL Specification**.
2. **The Kernel (The Grounding Layer):** The kernel parses the new DSL, validating it against strict schemas to prevent hallucinations. It then compiles the DSL into executable agent configurations.
3. **Inner Loop (The Tactician):** The instantiated agents execute tasks within the constraints of the DSL. The kernel captures every action, state change, and result into a **Trace Log**, which is fed back to the Outer Loop.

### Key Components & Characteristics

* **Abductive Reasoning Cycle:** The system treats engineering not as a one-off generation task but as a hypothesis testing loop. It continuously perceives the gap between *Goal* and *Trace*, reasons about the root cause (DSL structure vs. Parameter tuning), and refines the DSL accordingly.
* **Semi-formal DSL as Intermediate Representation (IR):** The DSL acts as a **cognitive scaffolding** that defines the search space. It prunes the infinite possibilities of natural language into a finite, manageable set of architectural choices, ensuring that agent behaviors are interpretable and reproducible.
* **Deterministic Kernel Execution:** While the reasoning (Outer Loop) is probabilistic, the execution (Inner Loop) is governed by a programmable kernel. This ensures that the generated DSL is strictly enforced, providing reliability and type safety.
* **Systematic Intelligence Engineering:** SPAK transforms "prompt engineering" into a disciplined engineering approach. It supports formal verification (e.g., satisfying invariants defined in DSL) and enables **search-based optimization** for complex problem solving.
* **Self-Correction via Grounding:** By coupling the LLM with a runtime kernel, the system enables **grounded self-correction**. Agents do not just "think" they are right; they "prove" it through execution traces.

---

## DSL Requirements (The Semi-formal Contract)

To function as a valid Intermediate Representation (IR) for the SPAK architecture, the DSL must meet the following criteria:

- **Completeness of Specification:** It must explicitly define the minimal system elements required for execution, including **Tasks**, **Items (Resources)**, **Invariants (Constraints)**, and **Success Criteria**.
- **Abductive Recoverability:** The DSL must be reverse-inferable from the **Trace Log**. An observer analyzing the execution trace should be able to reconstruct the DSL state that generated it.
- **Context-Free Reproducibility:** The DSL must be self-contained. A stateless LLM or Kernel should be able to execute the agent loop solely based on the DSL, without relying on hidden prompts or conversational history.

---

## Agent Loops (Runtime Mechanism)

Agent Loops within SPAK are the distinct runtime cycles responsible for **Tactical Execution (Inner Loop)** and **Strategic Refinement (Outer Loop)**. They transform the static DSL into dynamic behavior.

### 1. The Inner Loop: Tactical Execution

* **Role:** **"Do things right."** This loop operates *under* the constraints of the current DSL.
* **Process:**
* **Execution:** The agent performs actions using specific tools (APIs, RAG) defined in the DSL.
* **Evaluation:** The kernel checks if the immediate output meets the *Criteria* defined in the DSL.
* **Iteration:** If the criteria are not met (e.g., a code syntax error), the agent retries or adjusts parameters *within* the allowed DSL scope.


* **Output:** A comprehensive **Trace Log** detailing actions, latencies, and errors.

### 2. The Outer Loop: Strategic Refinement

* **Role:** **"Do the right things."** This loop operates *over* the DSL.
* **Process:**
* **Observation:** The LLM analyzes the **Trace Log** from the Inner Loop.
* **Abduction:** It identifies why the goal was not met (e.g., "The trace shows a timeout because the architectural pattern is sequential").
* **Refinement:** It modifies the **DSL Specification** (e.g., switching the pattern to `MapReduce`).


* **Outcome:** A new hypothesis (DSL) to be tested by the kernel.

---
### Application in Systematic Intelligence Engineering

* **Intent-Driven System Synthesis:** Automatically synthesizing complex software architectures (e.g., Microservices, PLC control logic) by iteratively refining the DSL based on unit test results and performance metrics.
* **Scientific Discovery & Optimization:** Managing multi-agent collaborations where the "Hypothesis" is formulated as a DSL and the "Experiment" is the agent loop execution, creating a closed-loop discovery engine.
* **Reliable Orchestration:** Providing a control plane (like LangGraph or TB-CSPN) where the flow of control is not hallucinated but strictly defined by the kernel-parsed DSL.

---
### Enterprise Implications

This dual-loop structure allows SPAK to handle scenarios requiring both precision and adaptability:

* **Automated Root Cause Analysis (RCA):** The Outer Loop can effectively act as an automated debugger, refining system configurations (DSL) until the issue resolves.
* **Dynamic Resource Allocation:** Agents can autonomously adjust their own resource constraints (defined in DSL) based on real-time performance traces.