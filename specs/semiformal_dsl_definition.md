# Semiformal DSL Specification: The SPAK "System Definition Language" (v2.1)

**Version:** 0.2.1
**Status:** Draft / Proposal
**Context:** Extended Loop Definitions for Agent, Service, and Engineering contexts.

---

## 1. Core Philosophy: The Loop Taxonomy

The SPAK architecture recognizes that not all "loops" are created equal. We differentiate them based on **Determinism**, **Latency**, and **Reasoning Capability**.

| Loop Type | Role | Determinism | Latency | Reasoning (LLM) | Typical Use |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Agent Loop** | **Strategist** | Low (Probabilistic) | High (Seconds) | **Yes** (Heavy) | Planning, Debugging, Architecture Design |
| **Service Loop** | **Operator** | High (Strict) | Low (Milliseconds) | **No** (Logic only) | API Serving, Request Handling, Event Routing |
| **Engineering Loop** | **Experimenter** | High (Repeatable) | Variable (Minutes) | **No** (Search only) | Auto-tuning, Benchmarking, A/B Testing |

## 2. Updated Loop Definitions

### 2.1 Agent Loop (`agent_loop`)
The "Brain". It observes state, reasons about it, and modifies the system or artifacts. It pauses for "Thought".

```spak-dsl
agent_loop Architect {
    step "Analyze Failure" {
        llm.query(role="Expert", prompt="Why did the kernel crash?")
    }
    step "Refine Plan" {
        tool.write(file="plan.md")
    }
}
```

### 2.2 Service Loop (`service_loop`)
The "Hands". It waits for external triggers and executes pre-compiled logic. It *never* calls the LLM directly (to ensure latency/cost guarantees).

```spak-dsl
service_loop RequestHandler {
    on user_request(query) {
        tool.search(query)
        emit response
    }
    
    on error(e) {
        emit log_entry
    }
}
```

### 2.3 Engineering Loop (`engineering_loop`)
The "Optimizer". It explores a parameter space defined by the Agent. It is a specialized Service Loop focused on finding maxima in a high-dimensional space.

```spak-dsl
engineering_loop AutoTuner {
    // 1. Define Search Space
    parameter TileSize: [64, 128, 256]
    parameter Warps: [4, 8]

    // 2. Define Experiment
    measure {
        cmd: "python benchmark.py --tile {{TileSize}} --warps {{Warps}}"
        metric: "tflops"
        objective: "maximize"
    }
}
```

---

## 3. Polymorphic Interaction

The Loops interact via **Traces** and **Artifacts**.

1.  **Agent -> Engineering**: The Agent writes the `benchmark.py` script (Artifact).
2.  **Engineering -> Trace**: The Engineering Loop runs and emits `TraceItem.Performance`.
3.  **Trace -> Agent**: The Agent observes the Trace, deduces "Memory Bound", and modifies `benchmark.py` for the next run.

This separation ensures that expensive LLM tokens are not wasted on the "Inner Loop" of trial-and-error, while the rigid execution environment is not confused by probabilistic instructions.