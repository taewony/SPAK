# SPAK: Spec-Driven Programmable Agent Kernel
## The Operating System for Systematic Intelligence Engineering

**"From Prompting to Programming: A Formal Kernel for Verifiable Agents"**

SPAK is a **Programmable Agent Kernel** designed to teach and enforce rigorous engineering practices in AI Agent development. Unlike frameworks that focus on "chaining prompts," SPAK treats Agent Logic as a formal artifact that must be **compiled from a Specification (AgentSpec)**, **verified against Invariants**, and **executed within an Effect-Isolated Runtime**.

This project serves as a reference implementation for **"Systematic Intelligence Engineering"**, providing a structured curriculum from simple functions to recursive, self-improving systems, and proving the "Round-Trip Consistency" of latent reasoning.

---

## 🏗 Core Architecture

> **"An Agent is an Endofunctor on a Semantic Category."**

SPAK operates on four fundamental pillars:

1.  **Semantic Specification (AgentSpec):** A Domain-Specific Language (DSL) that defines the agent's **State Space** (Category) and **Transitions** (Morphisms), not just its prompts.
2.  **Algebraic Effect Runtime:** Separates **Policy** (LLM Decisions) from **Mechanism** (IO/Tools). The Agent *requests* an effect; the Kernel *decides* how to handle it (Execute, Mock, or Deny).
3.  **Traceability & Verification:** The Kernel captures the **Reasoning Trace** (Latent Thought) as a formal artifact (`TraceLog`), enabling mathematical verification of "Intent Preservation" (Round-Trip Consistency Test).
4.  **Recursive Fractal Design:** The system is capable of infinite scalability via **Recursive Sub-Kernels**. A parent agent can spawn an isolated child agent to solve a sub-problem with a fresh context window.

---

## 📚 The Agent Curriculum (Maturity Levels)

SPAK implements a graded curriculum to demonstrate the evolution of agent complexity. All levels are implemented and verifiable in this repository.

| Level | Agent Type | Key Concept | Spec File | Status |
| :--- | :--- | :--- | :--- | :--- |
| **0** | **Static Responder** | Input $\to$ Output | `specs/level0.agent.spec.md` | ✅ Ready |
| **1** | **Context-Aware Bot** | State Persistence | `specs/level1.agent.spec.md` | ✅ Ready |
| **2** | **Tool-Use Agent** | Algebraic Effects (Math) | `specs/level2.agent.spec.md` | ✅ Ready |
| **3** | **Planning Agent** | Workflows & Traceability | `specs/level3.agent.spec.md` | ✅ Ready |
| **4** | **Multi-Agent System** | Collaboration (MsgBus) | `specs/level4.agent.spec.md` | ✅ Ready |
| **5** | **Recursive Solver** | Isolation (Sub-Kernel) | `specs/level5.agent.spec.md` | ✅ Ready |

### Understanding Agent Maturity

*   **Level 0 (Morphism):** Pure Input-Output mapping. No memory.
*   **Level 1 (Objects):** Immutable Semantic State (Memory/History).
*   **Level 2 (Side-Effect Isolation):** External world interaction via Kernel-Mediated Effects.
*   **Level 3 (Endofunctor):** Goal-oriented planning loops with **Reasoning Trace** and **Round-Trip Verification**.
*   **Level 4 (Category Composition):** Multi-agent systems sharing a Message Bus.
*   **Level 5 (Recursive Kernel):** Fractal scalability via recursive sub-agent spawning.

### Why "Curriculum-Based"?

In the context of SPAK, **"Curriculum-Based"** means that the platform is structured as a graded pedagogical path designed to teach **Agent Engineering** through incremental complexity.

Instead of throwing a developer into a complex framework (like LangChain or AutoGPT) where everything happens at once, SPAK forces you to master specific architectural concepts one level at a time. Each level represents a module in the curriculum, introducing one new "semantic pillar" of agent design.

**Why this is a "Curriculum" and not just a "Library":**

1.  **Scaffolding:** You cannot build a Level 5 Recursive Agent until you understand Level 2 Effects, because recursion in SPAK is implemented as a specific type of Effect (`Recurse`).
2.  **Formal Requirements:** To "pass" a level, your agent must satisfy the specific **Invariants** and **Success Criteria** defined in that level's specification (`.spec.md`). 
3.  **Conceptual Mapping:** It maps standard software engineering patterns to mathematical foundations (Category Theory). For example, it teaches that "Memory" is not just a database, but a transition between two immutable "Objects" in a category.
4.  **The "Round-Trip" Exam:** The curriculum includes a final verification step at each level: the **Consistency Test**. This proves that the student (or the LLM) didn't just write code that "looks right," but code that is **operationally consistent with the intent.**

---

## 🛠 Usage & Workflow

The system runs a **REPL-driven Build Loop**.

### 1. Basic Interactive Flow
```bash
# Start the Kernel Shell
$ python spak.py

# Load Specifications
(kernel) > load specs

# Activate a System (e.g., Level 3 Coach)
(kernel) > use CoachingAgent

# Run the Agent Interactively
(kernel) > run Coach
✅ Coach instantiated as 'app'.
>>> app.configure_session("Master Python")
>>> app.start_session()
[Coach]: Hello! How are you feeling?
[User]: I want to learn fast.
[Coach]: That's great spirit! Let's start with basics...
```

### 2. Verification Flow (TDD)
```bash
# Verify Implementation matches Spec and Tests
(kernel) > verify
[Static Analysis] Checking structure... OK
[Dynamic Analysis] Running tests... PASS

# Build/Repair (Auto-Coding)
(kernel) > build
[Builder] Synthesizing missing components...
✅ Synthesized src/coach.py
```

### 3. Advanced Verification (Round-Trip Consistency)
This unique feature verifies if the Agent's *latent reasoning* matched the *symbolic plan*.

```bash
# After running an agent...
(kernel) > trace
[Trace Table Displayed]

# Verify consistency against a PlanIR
(kernel) > consistency plans/research.plan.yaml
📊 Score: 100.0%
✅ PASSED: Execution Semantic Intent matches Plan.
```

---

## 🌍 Why "Kernel"?

In Computer Science, a **Kernel** manages resources, provides abstraction, and enforces isolation. SPAK does exactly this for Agents:

1.  **Resource Management:** Manages the **Context Window** as a scarce resource (via Recursion/Memory).
2.  **Isolation:** Protects the host system by sandboxing **Effect Execution** (e.g., `SafeREPLHandler`).
3.  **Abstraction:** Provides a standard Syscall interface (`perform Effect`) for LLMs, replacing fragile prompt engineering.
4.  **Auditability:** Enforces **Reasoning Trace** logging for every decision, ensuring no "hidden thoughts".

---

## 📝 Citation

If you use SPAK for research or education, please cite:

> **"SPAK: A Formally Verified, Spec-Driven Kernel for Curriculum-Based AI Agent Synthesis"** (Draft, 2026)
