# **SPAK: The Semantic Programmable Agent Kernel**

## **A Unified Architecture for Autonomous Engineering Systems (AES)**

**Abstract**

The development of Autonomous Engineering Systems (AES) is currently limited by the "Grounding Gap"—the disconnect between the probabilistic creativity of Large Language Models (LLMs) and the deterministic rigor required for physical engineering tasks. This document proposes **SPAK (Semantic Programmable Agent Kernel)**, a unified architecture that bridges this gap. By synthesizing the **Systematic Intelligence Engineering (SIE)** methodology with **Latent Space Topology** theory, SPAK treats the Agent not as a chatbot, but as a compilable specification. The system utilizes a tailored Domain-Specific Language (DSL) to act as a navigational manifold for the LLM's latent space, executed within a "Soft Sandbox" kernel. We define the architecture required to scale this system from simple assistance (Level 1\) to autonomous GPU hardware optimization (Level 5).

## **1\. Design Philosophy: The "Cognitive Loop" as a State Machine**

We reject the standard "Chain-of-Thought" loop in favor of a formalized **Cognitive State Machine**.

* **Theory (from Semantic Coupling):** The LLM is a stateless function $f(Context, Query) \\rightarrow Response$. The "Agent" is merely a trajectory through the LLM's latent space.  
* **Practice (from SPAK):** This trajectory must be constrained by a formal SystemModel (Axioms, Heuristics) and executed by a Kernel that enforces safety invariants.

### **1.1 The 4-Layer Control Hierarchy**

To support the 5 Levels of Autonomy, the system is strictly layered:

|

| **Layer** | **Component** | **Responsibility** | **Inputs/Outputs** |

| **L4** | **Meta-Supervisor** | Evolution & Compilation. Analyzes execution logs to patch/optimize the DSL. | Logs $\\rightarrow$ Updated DSL |

| **L3** | **SPAK DSL** | The "Constitution." Defines the System Model, Task Graph, and Memory Policy. | User Intent $\\rightarrow$ Semantic Spec |

| **L2** | **The Kernel** | The Runtime OS. Manages State, Memory, Tool Sandboxing, and Validation. | Spec \+ Triggers $\\rightarrow$ Actions |

| **L1** | **Worker-LLM** | Stateless Inference. Pure semantic projection. | Context $\\rightarrow$ ADT Items |

## **2\. The SPAK DSL Specification**

We merge the **Task-Graph** (from Paper A) with the **System Model** (from Paper B) into a single, cohesive grammar.

**Core Concept:** The DSL defines *what* the agent thinks about (System Model) and *how* it moves (Transitions).

### **2.1 DSL Template (spak\_agent.lark)**

// The Unified SPAK Grammar

agent\_def: "AGENT" name "LEVEL" int "{" config system\_model task\_graph "}"

// 1\. Configuration & Memory Policy

config: "CONFIG" "{" 

    "MEMORY" ("FIFO" | "SUMMARIZE" | "FLUSH\_ON\_TRANSITION") 

    "SANDBOX" ("RESTRICTED" | "DOCKER")

"}"

// 2\. The System Model (Domain Physics from SPAK)

system\_model: "SYSTEM\_MODEL" name "{"

    ("AXIOM" string)\* // Hard constraints injected into System Prompt

    ("HEURISTIC" string)\* // Hints triggered on failure

"}"

// 3\. The Task Graph (Execution Flow from Semantic Coupling)

task\_graph: "START" "-\>" task\_name (task)+

task: "TASK" task\_name "{"

    "GOAL" string

    "TOOLS" "\[" tool\_list "\]"

    "VALIDATOR" validator\_class?  // The Level 5 Hardware Bridge

    "NEXT" transition\_map

"}"

transition\_map: "{" (trigger "-\>" task\_name)\* "}"

trigger: "success" | "fail" | "metric\_improved" | "timeout"

### **2.2 Example: Level 5 GPU Tuner Spec**

AGENT Cuda\_Optimizer LEVEL 5 {

  CONFIG { MEMORY FLUSH\_ON\_TRANSITION, SANDBOX RESTRICTED }

  SYSTEM\_MODEL GPU\_Physics {

    AXIOM "Shared memory banks have 32-bit width."

    HEURISTIC "If memory bound, try fusing Softmax."

  }

  START \-\> baseline\_profile

  TASK baseline\_profile {

    GOAL "Compile and measure naive implementation."

    TOOLS \[nvcc, nsight\]

    VALIDATOR CudaProfiler

    NEXT { success \-\> optimize\_loop }

  }

  TASK optimize\_loop {

    GOAL "Refactor kernel for online-softmax."

    TOOLS \[read\_ast, write\_cuda\]

    VALIDATOR CudaProfiler

    NEXT { 

       metric\_improved \-\> optimize\_loop,

       plateau \-\> finalize,

       fail \-\> optimize\_loop // Triggers Heuristic automatically

    }

  }

}

## **3\. The Kernel Architecture (Runtime)**

The Kernel is the "Soft Sandbox" and "Event Loop." It does not run the LLM; it *orchestrates* it.

### **3.1 Core Modules**

1. **SpecLoader**: Parses the DSL. Extracts the TaskGraph for the Router and the SystemModel for the Context Builder.  
2. **StateRouter**: Maintains the pointer to the current TASK.  
   * *Logic:* current\_task \= transition\_map\[validator\_result\]  
3. **TraceManager (New)**: Structured Logger for Meta-Supervision.  
   * Records every cognitive step: Input Context, LLM Response (Raw), Parsed ADT, Tool Output, Validator Signal.  
   * *Format:* JSONL (JSON Lines) for easy machine parsing by Gemini.  
4. **ContextBuilder**: Constructs the prompt for the Worker-LLM.  
   * *Input:* System Model Axioms \+ Current Task Goal \+ (Filtered) Memory.  
   * *Output:* List of Open Responses Items.  
5. **SandboxExecutor**: The "Body."  
   * Executes tools (nvcc, python\_repl) in an isolated environment.  
   * *Security:* Enforces the ban on direct I/O unless explicitly allowed by the TOOLS list.  
6. **Validator (The Level 5 Bridge)**:  
   * A plugin system that runs external checks (Unit Tests, Profilers, Linters).  
   * Returns structured signals: Metric(latency=12ms), Status(FAIL).

### **3.2 The Execution Cycle (The Cognitive Loop)**

1. **Inject:** Kernel builds context from SystemModel \+ Task.GOAL.  
2. **Plan (Latent):** Worker-LLM generates ReasoningItem and FunctionCallItem.  
3. **Act (Sandbox):** Kernel executes FunctionCallItem (e.g., write\_cuda).  
4. **Observe (Validator):** Validator runs (e.g., nvcc).  
5. **Trace (Log):** TraceManager captures the entire tuple $(State\_t, Action\_t, Reward\_t, State\_{t+1})$.  
6. **Refine (Router):** \* If Validator returns FAIL: Kernel injects HEURISTIC from System Model into next Context (Self-Correction).  
   * If Validator returns SUCCESS: Router moves to NEXT task.

## **4\. Meta-Supervision Interface (Gemini CLI)**

To close the outer loop, Gemini CLI interacts with the Kernel via standard commands.

### **4.1 Verification Command**

spak verify \--artifact \<path\> \--spec \<agent.spak\>

This command validates if the final output complies with the SYSTEM\_MODEL axioms.

* **Process:**  
  1. Load SYSTEM\_MODEL from DSL.  
  2. Load generated artifact (code/report).  
  3. Run ComplianceVerifier (LLM-based judge or Static Analysis).  
  4. Return Compliance Report (Pass/Fail \+ Violation Details).

### **4.2 Trace Analysis Command**

spak analyze \--trace \<session\_id\>.jsonl

Gemini analyzes the JSONL log to identify:

* **Loop Detection:** Agent stuck in fail \-\> retry cycle.  
* **Prompt Inefficiency:** Contexts that yielded low-quality reasoning.  
* **Tool Failures:** Tools that consistently returned errors.

## **5\. Implementation Strategy (Migration Plan)**

To migrate from the existing codebase to SPAK, we recommend a "Kernel-First" approach.

### **Phase 1: The Protocol (Week 1\)**

* Define the **Algebraic Data Types (ADTs)**.  
* Migrate all LLM interaction code to output strict Item objects (Reasoning, ToolCall, Message).  
* *Goal:* Isolate the "Worker-LLM" as a pure function.

### **Phase 2: The Spec (Week 2\)**

* Implement the **Lark Parser** for the unified DSL.  
* Port the existing "Knowledge Chef" logic into a .spak file.  
* *Goal:* Delete hardcoded Python logic and replace it with the DSL file.

### **Phase 3: The Validator (Week 3\)**

* Implement the Validator abstract base class.  
* Build the CudaProfiler for the Level 5 experiment.  
* *Goal:* Enable the hardware feedback loop.

### **Phase 4: Meta-Supervision (Future)**

* Implement the outer loop where a superior model edits the .spak file based on run logs.

## **6\. Success Metrics**

| **Metric** | **Definition** | **Target (L5)** |

| **Consistency** | Constraint Satisfaction Rate (CSR) of generated artifacts. | \> 99% |

| **Self-Correction** | % of failures resolved autonomously via Heuristics. | \> 80% |

| **Efficiency** | Token usage per successful artifact (Context flushing enabled). | \< 50% vs Baseline |

| **Latent Alignment** | Perplexity of generated code when guided by DSL keywords. | Low |

