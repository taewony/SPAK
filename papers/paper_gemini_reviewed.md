# SPAK: A Spec-Driven Foundation for Systematic Intelligence Engineering

## Abstract

The rapid advancement of Large Language Models (LLMs) has catalyzed the development of autonomous agents; however, current systems rely heavily on ad-hoc prompt engineering, resulting in fragility, unverifiability, and a lack of scalability. We propose **Systematic Intelligence Engineering (SIE)**, a principled framework that decouples heuristic reasoning from deterministic execution. To operationalize SIE, we present the **Spec-driven Programmatic Agent Kernel (SPAK)**, a runtime environment that treats agent behavior as a formal, compilable artifact rather than probabilistic text generation. SPAK enforces a **Dual Verification Framework**, ensuring both *Operational Consistency* (alignment of thought and action) and *Domain Invariant Compliance* (adherence to safety constraints). We validate SPAK through a **6-Level Agent Curriculum**, demonstrating that a single kernel can support a progression from simple static responders to recursive, self-improving solvers. This work establishes the architectural foundation for **Autonomous Engineering Systems (AES)** capable of complex, high-reliability tasks. The reference implementation and 6-level curriculum are available at: https://github.com/taewony/SPAK

---

## 1. Introduction

The paradigm of AI engineering is shifting from chat-based interactions to agent-based automation. While frameworks like LangChain and AutoGen facilitate rapid prototyping, they often entangle probabilistic reasoning with execution logic. This leads to "black-box" behaviors where failures are difficult to trace, and safety constraints are easily bypassed by hallucination.

**Core Thesis:** Reliability in autonomous systems is an architectural problem, not merely a model capability problem. Robust intelligence emerges only when neural heuristics are strictly isolated from mechanized execution via explicit specifications.

To address this, we introduce **SPAK (Spec-driven Programmatic Agent Kernel)**. Unlike libraries that act as wrappers around API calls, SPAK functions as an operating system kernel for agents, managing context, enforcing permissions, and validating state transitions against a formal schema.

**Key Contributions:**

1. **The SIE Framework:** A methodology for building reliable systems via the separation of neural creativity and symbolic enforcement.
2. **SPAK Architecture:** A kernel design implementing Algebraic Effects and Algebraic Data Types (ADTs) for agent isolation.
3. **Dual Verification Mechanism:** A formal approach to validating agent actions against both intent and domain laws.
4. **6-Level Curriculum Evaluation:** A systematic validation method demonstrating architectural scalability across varying levels of agent complexity.

---

## 2. Related Work

### 2.1 Agent Orchestration Frameworks

Current frameworks such as **LangChain** [Chase, 2022], **LlamaIndex** [Liu, 2023], and **Microsoft AutoGen** [Wu et al., 2023] focus on ease of composition and multi-agent conversation. While effective for scripting and prototyping, they predominantly rely on "Chain-of-Thought" prompting without formal runtime guarantees. SPAK differentiates itself by prioritizing **correctness constraints** (via `AgentSpec`) over compositional flexibility, treating the agent definition as a compilable contract rather than a Python script.

### 2.2 Neuro-Symbolic AI

Neuro-symbolic approaches attempt to combine neural networks with logic programming [Garcez et al., 2020]. Classic methods often require differentiable logic layers or specialized training. SPAK adopts a pragmatic **"System 1 / System 2"** approach: it does not restrict the *internal* reasoning of the LLM (System 1) but strictly enforces the *external* interfaces and side effects via a symbolic DSL (System 2). This aligns with recent work on **Toolformer** [Schick et al., 2023] but adds a layer of formal invariant checking.

### 2.3 Formal Verification in Software Engineering

We draw upon principles of **Design by Contract (DbC)** [Meyer, 1992] and **Runtime Verification** [Leucker & Schallhart, 2009]. Unlike static verification (which is intractable for the probabilistic state space of LLMs), SPAK treats correctness as a **runtime property** enforced by the kernel. The kernel acts as the "Runtime Monitor" that blocks actions violating the contract defined in `AgentSpec`.

### 2.4 Autonomous Engineering Systems (AES)

Recent initiatives like **ADRS (AI-Driven Research for Systems)** [Ananthanarayanan et al., 2025] demonstrate the potential of AI agents to automate the discovery and optimization of computer systems algorithms. While ADRS focuses on the evolutionary search for better algorithms, SPAK provides the underlying kernel architecture to ensure such autonomous systems operate within strictly defined safety and correctness boundaries.

---

## 3. The Systematic Intelligence Engineering (SIE) Framework

SIE structures intelligent systems into a hierarchy of control, moving from raw execution to meta-cognitive evolution.

### 3.1 Hierarchical Architecture

```
Layer 4: Meta-Level Supervision (Self-Improvement)
   │     • Log Analysis / Spec Evolution / Heuristic Patching
   ▼
Layer 3: Systematic Intelligence Engineering (SIE) Methodology
   │     • Design Patterns / Lifecycle Management
   ▼
Layer 2: Autonomous Engineering System (AES)
   │     • Domain Specific Agents (e.g., ADRS, GPU Optimizer, Web Builder)
   ▼
Layer 1: SPAK Kernel (Runtime)
         • Execution / Verification / Isolation
```

### 3.2 Design Principles

1. **Specification Primacy:** Agent behavior must be defined in **AgentSpec**, a machine-readable DSL, not natural language prompts.
2. **Heuristic-Mechanism Separation:** The "Brain" (LLM) proposes actions; the "Body" (Kernel) validates and executes them.
3. **Immutable State:** Agent memory utilizes value semantics, enabling perfect time-travel debugging and auditability.

---

## 4. The SPAK Architecture

SPAK acts as the sandbox and execution engine. It borrows concepts from Functional Programming to ensure safety.

### 4.1 AgentSpec: The Domain Specific Language

Agents are defined using **AgentSpec**, which captures:

* **Domain Model:** Algebraic Data Types (Sum/Product types) ensuring illegal states are unrepresentable.
* **Workflow:** A state-machine representation of the agent's lifecycle.
* **Invariants:** Logic predicates that must hold true.

**Listing 1: AgentSpec Example (Reviewer Component)**
```agentspec
component Reviewer {
    description: "The Taster. Evaluates artifacts against success criteria.";
    
    // Domain Invariant: Engineering Law
    invariant: "Final artifact must contain all required sections (Motivation, Background, Methodology, Results)"
    invariant: "No hallucinated citations allowed (Result coverage > 0.9)"
    
    function evaluate(draft: String, criteria: List<String>) -> EvaluationResult;
}
```

### 4.2 Algebraic Effects for I/O Isolation

Agents cannot directly access the filesystem or network. They must `yield` an effect (e.g., `WriteFile`). The Kernel intercepts this effect, checks permissions against the Spec, and determines whether to execute, mock, or reject the request. This prevents "Prompt Injection" attacks from manifesting as system-level breaches.

---

## 5. Dual Verification Framework

A core innovation of SPAK is the gating of all agent outputs through two distinct verification layers.

### 5.1 Definition 1: Operational Consistency ( $\Phi_{op}$ )

This verifies that the agent's *symbolic output* matches its *latent reasoning*.
Given a reasoning trace $R$ and a structured plan $P$:

$$ \Phi_{op}(R, P) \iff \forall s \in P, \exists t \in R : \text{aligns}(t, s) $$

* **Purpose:** Ensures the agent "knows what it is doing" and isn't guessing.
* **Algorithm:** The `ConsistencyVerifier` extracts steps from the `PlanIR` (derived from the Spec) and aligns them with `ReasoningTrace` events. It checks if the "Thought" contains required semantic keywords and if the "Action" matches the planned operation.

### 5.2 Definition 2: Domain Invariant Compliance ( $\Phi_{dom}$ )

This verifies that the execution result $E$ satisfies all safety constraints $I$ defined in AgentSpec.

$$ \Phi_{dom}(E, I) \iff \forall i \in I : i(E) \text{ is True} $$

* **Purpose:** Ensures the output is safe and compliant with engineering standards, regardless of the agent's reasoning.

### 5.3 Execution Condition

The Kernel executes an action if and only if:

$$ \text{Execute}(a) \iff \Phi_{op}(R, P) \land \Phi_{dom}(State', I) $$

---

## 6. Experimental Validation: The 6-Level Curriculum

To demonstrate the universality of SPAK, we implemented a graded curriculum of agents, all running on the same kernel instance.

### 6.1 Curriculum Structure

| Level | Agent Type | Architectural Concept | SPAK Feature Utilized |
| --- | --- | --- | --- |
| **0** | **Static Responder** | Input $\to$ Output Morphism | Stateless execution |
| **1** | **Context-Aware Bot** | State Persistence | Immutable History management |
| **2** | **Tool-Use Agent** | Side Effects | Algebraic Effect Handlers |
| **3** | **Planning Agent** | Workflow Control | Traceability & `Think-Act` loop |
| **4** | **Multi-Agent System** | Composition | Inter-agent Message Bus |
| **5** | **Self-Improving** | Meta-Recursion | Dynamic Spec Evolution |

### 6.2 Quantitative Evaluation (Preliminary)

We evaluated the **"Knowledge Chef" (Level 4)** system against a baseline prompt-only agent on a task of synthesizing technical reports from 5 documents.

* **Method:** We injected 50 "Invalid" tasks (missing data, requesting restricted actions) and 50 "Valid" tasks.
* **Reliability (Invariant Enforcement):**
    *   **Baseline:** Generated hallucinatory content for **42%** of invalid tasks.
    *   **SPAK:** Successfully rejected **100%** of invalid tasks via `InvariantFailure` exceptions.
* **Operational Consistency:**
    *   SPAK detected "Semantic Drift" (where the agent's plan diverged from its action) in **15%** of early training runs, allowing for targeted prompt refinement.
* **Overhead:** The Dual Verification added an average of **120ms** latency per step, which is negligible (< 5%) compared to the LLM inference latency.

This confirms that the kernel provides a hard safety guarantee that probabilistic models alone cannot achieve.

---

## 7. Case Study: The Knowledge Chef (Level 4)

We deployed a specific Level 4 implementation, the **"Knowledge Chef,"** to synthesize technical reports from raw documentation.

### 7.1 Architecture

The system composes four specialized agents:

1. **Librarian:** Discovers sources (Constraint: Must cite valid URLs).
2. **Analyst:** Extracts typed `Insight` structs.
3. **Writer:** Generates prose based on Insights.
4. **Reviewer:** Validates the prose against the original `Insight` structs.

### 7.2 Results

In our empirical evaluation, the Knowledge Chef was deployed using the **ollama/qwen3:8b** model accelerated by an **NVIDIA GeForce RTX 5070** GPU. The workflow executed seamlessly, producing the following terminal status: `🎉 [ChiefEditor] Paper published to: final_paper.md; Result: Workflow Complete`.

The Reviewer agent, enforcing the **"Source Fidelity Invariant,"** successfully rejected hallucinations where the Writer invented details not present in the Analyst's output. The resulting documents maintained **100% citation accuracy** relative to the provided context, as verified by manual audit of the `ReasoningTrace`.

---

## 8. Discussion: Towards Self-Improving Systems

The ultimate goal of SIE is Level 5: **Self-Improvement**. By logging verification failures (e.g., "Invariant X violated 5 times"), the Meta-Supervisor can:

1. **Patch Heuristics:** Update the prompt strategy for the failing sub-agent (e.g., "Add instruction to check date format").
2. **Relax Specs:** If an invariant is too strict (false positive), the system can propose a Spec amendment.
3. **Code Repair:** For Level 2+ agents, the system can rewrite the Python tool implementation to fix runtime errors.

In our Level 5 experiments, the `MetaSolver` successfully spawned a sub-agent to fix a broken calculation tool, demonstrating the viability of this closed-loop evolution.

---

## 9. Conclusion

We have presented **SPAK**, a kernel that brings software engineering rigor to AI agents. By formalizing the interface between neural creativity and symbolic execution, we enable the construction of systems that are verifiable by design. The successful deployment of the 6-Level Curriculum demonstrates that SPAK provides a scalable foundation for the next generation of **Autonomous Engineering Systems**.

---

## References

- Ananthanarayanan, G., et al. (2025). Barbarians at the Gate: How AI is Upending Systems Research. arXiv preprint arXiv:2510.06189.
- Chase, H. (2022). LangChain: Building applications with LLMs. GitHub: https://github.com/hwchase17/langchain.
- Garcez, A. D., et al. (2020). Neurosymbolic AI: The 3rd Wave. arXiv preprint arXiv:2012.05876.
- Leucker, M., & Schallhart, C. (2009). A brief history of runtime verification. Journal of Logic and Algebraic Programming, 78(5), 293-303.
- Liu, J. (2023). LlamaIndex: Interface between LLMs and your data. https://www.llamaindex.ai.
- Meyer, B. (1992). Design by Contract. IEEE Computer, 25(10), 40-51.
- OpenAI. (2023). GPT-4 Technical Report. arXiv preprint arXiv:2303.08774.
- Plotkin, G. D., & Power, A. J. (2003). Algebraic operations and generic effects. Applied Categorical Structures, 11, 69-94.
- Schick, T., et al. (2023). Toolformer: Language Models Can Teach Themselves to Use Tools. arXiv preprint arXiv:2302.04761.
- Weng, L. (2023, June 23). LLM-powered Autonomous Agents. Lil'Log. https://lilianweng.github.io/posts/2023-06-23-agent/.
- Wu, Q., et al. (2023). AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation. arXiv preprint arXiv:2308.08155.

---

## Appendices

### Appendix A: AgentSpec Grammar (EBNF)

```ebnf
system_def  ::= "system" NAME "{" component_def* "}"
component_def ::= "component" NAME "{" member_def* "}" 
                | "workflow" NAME "(" param_list ")" "{" step* "}"
member_def  ::= "function" NAME "(" params ")" "->" type
              | "invariant" ":" STRING
step        ::= "step" NAME "{" logic "}"
```

### Appendix B: Functional Programming Primitives

**Algebraic Effects:** SPAK implements effects as `Effect[T]` data classes. The runtime `perform(Effect)` suspends execution, bubbling the effect to the nearest handler in the stack, similar to Extensible Effects in Haskell or Koka. This ensures that agent logic remains pure and testable, as side effects are interpreted only at the boundary.

```python
class Effect[T]:
    pass

def perform(effect: Effect[T]) -> T:
    # ... runtime implementation ...
    pass
```
