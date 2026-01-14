# Appendix A. AgentSpec v0.3 Language Specification

## A.1 Overview

**AgentSpec** is a domain-specific specification language for defining the **semantic structure, behavioral constraints, and effect boundaries** of programmable agents.
Unlike prompt-centric agent frameworks, AgentSpec treats agent behavior as a **compilable semantic artifact**, enabling static validation, runtime verification, and semantic-preserving code generation.

AgentSpec serves three roles simultaneously:

1. **Semantic Specification Language**
   for modeling agent intent, state, transitions, and invariants.
2. **Intermediate Representation (Semantic IR)**
   between high-level intent and executable implementations.
3. **Verification Boundary**
   for constraining large language models (LLMs) within predictable and auditable execution environments.

---

## A.2 Design Principles

AgentSpec is designed around the following principles:

1. **Semantic First**
   Meaning, intent, and constraints are primary; implementation is secondary.
2. **State–Transition Semantics**
   Agents are modeled as explicit state transition systems.
3. **Effect Isolation**
   All side effects are explicitly declared and mediated by the runtime.
4. **Gradual Formalization**
   Specifications may evolve from natural language to executable predicates.
5. **Verification-Driven Development**
   Correctness is enforced through invariants, tests, and effect auditing.

---

## A.3 Core Language Constructs

AgentSpec consists of the following top-level constructs:

```
AgentSpec ::= Agent
Agent      ::= metadata
               imports?
               effects*
               types*
               state
               invariants*
               functions*
               workflows*
               tests*
```

Each construct is described below.

---

## A.4 Metadata

Metadata provides contextual information for tooling, versioning, and reproducibility.

```agentspec
meta {
  name        = "Meta-Build-Agent"
  version     = "0.2"
  description = "Spec-driven agent build and verification kernel"
  author      = "Anonymous"
}
```

---

## A.5 Effects

### A.5.1 Effect Declarations

Effects represent all **observable interactions with the external world**, including:

* File system access
* Network calls
* Memory mutation
* Tool invocation
* Logging and monitoring

```agentspec
effect FileIO {
  operation read(path: String) -> Result<String, Error>
  operation write(path: String, content: String) -> Result<Unit, Error>
}

effect Network {
  operation request(url: String, method: String, body: String?) -> Result<Response, Error>
}
```

### A.5.2 Effect Semantics

* Effects are **declarative** and **non-executable**.
* Effects define *what may happen*, not *how it happens*.
* The runtime decides:
  * whether to allow an effect,
  * how to execute it,
  * whether to mock, replay, or deny it.
* **Return Type:** Operations SHOULD return a `Result` type to handle failures gracefully without exceptions.

---

## A.6 Types (Algebraic Data Types)

AgentSpec supports **algebraic data types (ADTs)** for modeling semantic domains.

```agentspec
type Node =
  | Root
  | Section(id: String, title: String)
  | Paragraph(id: String, content: String)

type Result<T, E> =
  | Ok(value: T)
  | Err(error: E)
```

Types are:

* Immutable
* Serializable
* Used for state, function inputs, and outputs

---

## A.7 State

The `state` block defines the **persistent semantic state** of the agent.

```agentspec
state {
  nodes        : Map<String, Node>
  current_node : String?
  history      : List<Event>
}
```

### State Properties

* State is immutable from the agent’s perspective.
* All state changes occur via declared transitions.
* State evolution is recorded as an event log (event-sourcing compatible).
* **Implementation Note:** Maps to `pyrsistent` PMap/PVector or Frozen Dataclasses in Python.

---

## A.8 Invariants

Invariants define **semantic constraints that must always hold**.

### A.8.1 Natural Language Invariants (Level 1)

```agentspec
invariant "All node IDs must be unique."
```

### A.8.2 Logical Invariants (Level 2)

```agentspec
invariant forall n1, n2 in state.nodes:
  n1.id == n2.id implies n1 == n2
```

### A.8.3 Executable Invariants (Level 3)

```agentspec
invariant python {
  def check(state):
      # Must be pure and safe to execute
      return len(state.nodes) == len(set(state.nodes.keys()))
}
```

The system supports **gradual refinement** across these levels.

---

## A.9 Functions (Semantic Transitions)

Functions represent **pure semantic transitions** over state.

```agentspec
function add_node(node: Node)
  requires node.id not in state.nodes
  ensures state.nodes[node.id] == node
  effects { StateWrite }
  returns Result<Node, Error>
```

### Function Properties

* Functions are **pure by default**
* Side effects must be explicitly declared
* Preconditions (`requires`) and postconditions (`ensures`) are optional but encouraged

---

## A.10 Workflows (Composite Transitions)

Workflows define **structured multi-step behaviors**, analogous to programs.

```agentspec
workflow build_section(title: String) {
  step create_section
  step add_paragraph
  step validate_structure
}
```

Workflows:

* Compose functions and effects
* Define higher-level agent behavior
* Are subject to invariant and effect checks

---

## A.11 Tests (Verification Artifacts)

Tests are **executable specifications** that validate agent behavior.

```agentspec
test "Add section increases node count" {
  given initial_state
  when add_node(Section("s1", "Intro"))
  then state.nodes.size == 1
}
```

Tests form the **primary feedback loop** for agent correctness.

---

## A.12 Semantic IR Mapping

AgentSpec compiles into a **Semantic Intermediate Representation (Semantic IR)** consisting of:

* State schema
* Transition graph
* Effect signatures
* Invariant set
* Verification artifacts

This IR is:

* Language-agnostic
* Replayable
* Suitable for reverse engineering and code synthesis

---

## A.13 Reverse Engineering Support

AgentSpec supports *semantic lifting* from existing codebases by extracting:

* State variables → `state`
* Side effects → `effect`
* Control flow → `workflow`
* Assertions/tests → `invariant`

This enables **semantic-preserving reimplementation** rather than literal translation.

---

## A.14 Execution Model Summary

1. AgentSpec is parsed and validated.
2. Semantic IR is constructed.
3. Verification artifacts are generated.
4. The Meta-Build-Agent runtime:
   * mediates effects,
   * enforces invariants,
   * logs execution traces,
   * supports replay and repair.

---

## A.15 Limitations and Scope

AgentSpec v0.2 intentionally does **not**:

* Specify LLM architectures
* Control model training
* Guarantee optimal reasoning quality

Its focus is **agent reliability, reproducibility, and semantic clarity**.

---

## A.16 Summary

AgentSpec v0.2 provides a formal yet flexible language for modeling agent semantics, enabling AI agents to be engineered with the same rigor as traditional software systems—while remaining compatible with the probabilistic nature of large language models.

---

# Appendix C. Implementation Considerations (v1.0 Roadmap)

## C.1 Parser & Compiler
*   **Lark Grammar:** The grammar must handle nested blocks (`state`, `effect`, `invariant`) robustly.
*   **Type Mapping:** ADTs (`type Node = ...`) should map to Python `dataclasses` (frozen=True) with a discriminative union pattern or `typing.Union`.
*   **Result Type:** Since Python lacks a native `Result` monad, use a lightweight implementation (e.g., `returns` library or a custom class) to enforce error handling without `try-except` chaos.

## C.2 Runtime Kernel
*   **Effect Handling:** Use Python `generators` (`yield Effect(...)`) as the primary mechanism for effect suspension and resumption. This avoids "callback hell" and keeps agent logic linear and synchronous-looking.
*   **Sandboxing:** For `invariant python` blocks, strictly use `RestrictedPython` to prevent access to `os`, `sys`, or network calls. The verification step must not become a security hole.
*   **State Persistence:** Use `pyrsistent` for the state map/list to ensure O(1) snapshots for time-travel debugging and replay.

## C.3 LLM Integration
*   **Structured Output:** When the Builder Agent generates implementation code, use JSON mode or strict prompt templates to ensure the output matches the Semantic IR signatures.
*   **Prompting Strategy:** Inject the Semantic IR as a "Context" into the system prompt: "You are implementing an agent defined by this Spec. Do not deviate from the interfaces."

## C.4 Distribution
*   **Package Structure:**
    *   `sbak-kernel`: The runtime and effect system.
    *   `sbak-compiler`: The Lark parser and codegen.
    *   `sbak-cli`: The REPL and build tools.
*   **Self-Hosting:** The ultimate test is defining `SPEC.sbak.md` and having the current implementation rebuild parts of itself.