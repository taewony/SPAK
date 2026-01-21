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

### 3. Advanced Verification: The Round-Trip Consistency Test (RTCT)
This unique feature provides the **Methodological Validation** claimed in the SPAK whitepaper. It verifies if the Agent's *latent reasoning* (unseen thoughts) structurally maps to its *symbolic execution* (actual actions).

*   **Structural Grounding:** Proves the Agent followed a deterministic procedure rather than just "guessing" a correct-looking response.
*   **Semantic Drift Detection:** Identifies when the Agent's internal intent deviates from the formal Specification (`PlanIR`).
*   **Intent Recovery Rate:** Quantifies how much of the original plan was successfully recovered from the execution trace.

```bash
# After running an agent (e.g., CoachingAgent)...
(kernel) > trace
# Inspect the 'mind' of the agent (Thoughts vs. Plans)

# Verify consistency against a PlanIR (The formal 'Law' of the workflow)
(kernel) > consistency plans/coaching.plan.yaml
⚖️ [Consistency] Verifying trace against PlanIR: 'Coaching Session Workflow'
📊 Score: 100.0%
✅ PASSED: Execution Semantic Intent matches Plan.
```

 1. The Core Scientific Claim: "Structural Grounding"
  The Paper argues: It is not enough for an Agent to produce a correct text response (the "Output"). To be
  scientifically valid and reliable, the Agent's latent reasoning (its private thoughts) must structurally map to its
  explicit actions (system calls).

  The Check validates:
   * Latent Space: The "Thought" logs (ReasoningTrace.thought).
   * Symbolic Space: The "Action" logs (ReasoningTrace.plan and actual tool calls).

  If the consistency command returns PASSED, you have evidence that the LLM didn't just "guess" the right answer; it
  followed the correct procedure to get there.

  2. "Methodological Validation" vs. "Output Evaluation"
  Most LLM benchmarks just check: Did the user get the right answer?
  Your Consistency Check asks: Did the Agent follow the Plan?

   * PlanIR (`coaching.plan.yaml`): This is the "Law" or the "Specification". It represents the ideal algorithmic flow
     (e.g., "First establish context, then loop, then analyze").
   * Trace (`runtime.trace`): This is the "Reality". It captures what the probabilistic model actually did.

  The "Consistency Check" calculates the semantic distance between the Law and Reality.

  3. How it works exactly (The Mechanics)

  When you ran consistency plans/coaching.plan.yaml, the kernel performed these 3 steps:

   1. Intent Extraction: It looked at the YAML to see what should happen.
       * Expectation: "In the 'Session Start' phase, the agent MUST use action init AND it MUST think about
         'establishing a baseline'."
   2. Trace Alignment: It scanned the actual logs from your Coach run.
       * Reality: It found a log entry where plan={'action': 'init'} and thought="...establish a baseline...".
   3. Verification:
       * Because Action matched (Structural Consistency)...
       * AND Keywords matched (Semantic Consistency)...
       * Result: The step is Verified.
	   
> **Claim:** This check elevates LLMs from "Black Boxes" to "Systematic Components" by proving that latent reasoning is operationally consistent with explicit symbolic procedures.

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
