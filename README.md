# Meta Build-Agent Kernel (MBAK)
## to specify and build the Build-Agent itself and build other domain agent
**"A Formal Spec-Driven, Recursively Self-Improving Agent Kernel"**

This project implements a **Programmable and Verifiable Agent Kernel**. Unlike traditional agent frameworks that focus on "orchestration" (chaining prompts), this system focuses on **"Correctness & Synthesis"**. It treats Agent Logic as an artifact that must be **compiled from a formal specification (AISpec)**, **verified against test vectors**, and **executed within an effect-isolated runtime**.

## 🏗 Core Definition

> **"It is like Terraform for AI Agents, but instead of cloud infrastructure, it manages Software Logic."**

The system operates on three fundamental pillars:

1.  **Compiler, Not Interpreter:** It "compiles" a **Formal Spec (`SPEC.md`)** into an **Executable Agent (`src/*.py`)** using an LLM as the code generator.
2.  **Effect-Isolated Runtime:** It separates **Policy** (Decision) from **Runtime** (Execution) using **Algebraic Effects**. Agents yield intents; the Kernel handles them.
3.  **Recursive Fractal Design:** The system is capable of building itself. A "Build Agent" can define a sub-spec, spawn a "Sub-Build Agent" to implement it, verify it, and merge it back.

---

## 📚 Terminology & Concepts

### 1. The "Inside vs. Outside" Architecture

This system is designed to bridge the gap between High-Level Intent and Low-Level Execution.

| Realm | **Outside the System** | **Inside the System** |
| :--- | :--- | :--- |
| **Agent** | **The Architect** | **The Contractor** |
| **Intelligence** | Frontier Models (GPT-4o, Claude 3.5) + Humans | Local Models (Llama 3, Gemma 2, Qwen 2.5) |
| **Role** | Define **Specs** and **Constraints**. | Implement Logic, Fix Bugs, Pass Tests. |
| **Artifact** | `SPEC.*.md`, `tests.*.yaml` | `src/*.py`, `build/*.html`, `trace.json` |

### 2. Semantic IR (AISpec)

The **Intermediate Representation (IR)** of our system. It is a Domain Specific Language (DSL) that defines *what* an agent should do, without defining *how*.

*   **Format:** `SPEC.system.component.md` (Lark-based DSL)
*   **Content:** Interfaces, State Definitions, Invariants, Capabilities.
*   **Example:**
    ```aispec
    system WebBuilder {
        component Header {
            state { title: String, color: Hex }
            invariant: "Contrast ratio > 4.5";
        }
    }
    ```

### 3. Verification Vectors

Language-agnostic definitions of expected behavior. These are the "Unit Tests" for the Spec.

*   **Format:** `tests.system.component.yaml`
*   **Content:** Input State -> Expected Output State / Effect.
*   **Example:**
    ```yaml
    input: { title: "Hello" }
    expected: { html: "<h1>Hello</h1>" }
    ```

### 4. Algebraic Effects

Side effects are treated as data values. Agents do not `open()` files or `request()` URLs directly. They yield an **Effect Object**.

*   `Effect`: An intent (e.g., `WriteFile`, `Browser.Screenshot`).
*   `Handler`: The kernel component that executes the intent (e.g., `FileSystemHandler`, `PlaywrightHandler`).
*   **Benefit:** 100% Deterministic Replayability.

---

## 🛠 Usage & Workflow

The system runs a **REPL-driven Build Loop**:

```bash
# Start the Kernel Shell
$ python spec_repl.py

# 1. Load a Specification (The Blueprint)
(kernel) > load specs/SPEC.web_agent.md
[Kernel] Loaded System: WebBuilder

# 2. Verify Structural Integrity
(kernel) > verify structure
[Check] Component 'Header' ... MISSING implementation.

# 3. Auto-Implement (The Build)
(kernel) > implement
[Builder] Generating code for 'Header' using Local LLM... Done.

# 4. Verify Behavior (The Test)
(kernel) > verify behavior
[Pytest] test_header_rendering ... FAIL (Contrast ratio too low)

# 5. Recursive Repair
(kernel) > repair
[Builder] Reading error log... Adjusting CSS color... Done.
(kernel) > verify behavior
[Pytest] test_header_rendering ... PASS
```

---

## 🌍 Generic Application Domains

While built for Software Engineering, this kernel is a **Universal Factory** for any domain that requires "Spec -> Artifact -> Verification".

| Domain | Spec (AISpec) | Artifact | Verification |
| :--- | :--- | :--- | :--- |
| **Web Dev** | UI Component / User Flow | HTML / React / Vue | Playwright / Lighthouse |
| **Data Eng** | Schema / Pipeline Logic | SQL / Airflow DAGs | Data Quality Tests |
| **DevOps** | Infrastructure State | Terraform / K8s Manifests | Policy-as-Code (OPA) |
| **Reverse Eng** | Decompiled Source | High-Level AISpec | Structural Similarity |

### LLM-Based Reverse Engineering
The system can run in reverse: **Code -> Spec**.
1.  **Input:** Legacy Source Code.
2.  **Process:** The Kernel uses an LLM to extract the "Implied Spec" (State, Invariants).
3.  **Output:** `SPEC.legacy.md`.
4.  **Value:** This Spec can then be used to re-implement the system in a modern stack (e.g., COBOL -> Spec -> Python).

---

## 📂 Project Structure

```text
/
├── compiler.py        # AISpec Parser (Lark)
├── verifier.py        # Static & Dynamic Verification Engine
├── builder.py         # LLM Code Generator & Repairer
├── runtime.py         # Effect-Isolated Execution Kernel
├── spec_repl.py       # Interactive Shell (The UI)
├── specs/             # The Source of Truth (AISpecs)
│   ├── SPEC.root.md
│   └── ...
├── tests/             # The Guardrails (YAML Vectors)
│   ├── tests.root.yaml
│   └── ...
└── src/               # The Generated Implementation
    └── ...
```

---

## 🚀 Getting Started

1.  **Install Dependencies:** `pip install -r requirements.txt` (requires `lark`, `pytest`, `pyyaml`)
2.  **Run the Shell:** `python spec_repl.py`
3.  **Define your first Spec:** Create `specs/SPEC.hello.md`.
4.  **Build it:** Run `load specs/SPEC.hello.md` then `implement`.

---

## Appendix: Generalizing to Other Domains

Because the system is "Spec In -> Artifact Out", it can apply to almost anything:

| Domain | Spec (AISpec) | Artifact | Verification |
| :--- | :--- | :--- | :--- |
| **Web Dev** | UI Component / User Flow | HTML / React / Vue | Playwright / Storybook / Lighthouse |
| **Data Eng** | Schema / Pipeline Logic | SQL / Airflow DAGs | Data Quality Tests (GreatExpectations) |
| **DevOps** | Infrastructure State | Terraform / K8s Manifests | `terraform plan` / Policy-as-Code (OPA) |
| **Research** | Experiment Protocol | Python / Jupyter Notebook | Statistical Significance Check |

### Conclusion: The Build-Agent is generic.

By defining the system as **"Programmable and Verifiable"**, you can use it as a **"Universal Agent Factory"**.
*   The **"Blueprints"** (AISpec) change based on the product (Web, Backend, Data).
*   The **"Quality Control"** (Verification) changes based on the product.
*   But the **"Factory Floor"** (The Agent Kernel: Parse -> Build -> Verify -> Refine) remains exactly the same.

---

## Appendix: Why "Kernel"?

"Kernel"이라는 용어는 이 시스템을 단순한 **'도구(Tool)'**나 **'프레임워크(Framework)'**가 아닌, **'자원 관리 및 실행 제어의 핵심 엔진'**으로 정의하기 때문에 사용합니다.

컴퓨터 과학에서 **커널(Kernel)**이라고 부르기 위해서는 다음과 같은 **핵심 역할(Criteria)**을 수행해야 합니다. 우리 시스템이 이 조건들을 어떻게 만족하는지 설명해 드립니다.

### **1. 자원 관리 및 추상화 (Resource Abstraction)**

*   **OS Kernel의 역할:** CPU, 메모리, I/O 장치(하드웨어)를 추상화하여 프로세스에 제공합니다.
*   **MBAK Kernel의 역할:** **LLM(지능 자원)**과 **도구(File, Network)**를 추상화하여 에이전트(프로세스)에 제공합니다.
    *   에이전트는 직접 `open()`을 호출하지 못합니다. 커널에게 `FileSystemEffect`를 요청해야 합니다.
    *   에이전트는 어떤 LLM이 도는지 모릅니다. 커널이 `LLM.generate` 요청을 받아 적절한 모델(Llama 3, GPT-4)에 배분합니다.

### **2. 격리 및 보호 (Isolation & Protection)**

*   **OS Kernel의 역할:** 프로세스 간 메모리를 격리하고, 잘못된 프로그램이 시스템 전체를 망가뜨리지 않도록(Protection Ring) 보호합니다.
*   **MBAK Kernel의 역할:** **에이전트의 논리(Policy)와 실행(Runtime)을 격리**합니다.
    *   에이전트가 "파일 시스템 전체 삭제"라는 코드를 생성해도, 커널(Runtime)이 그 Effect를 거부하거나 Mocking하면 실행되지 않습니다.
    *   **Sandboxing:** 에이전트가 생성한 코드는 커널이 관리하는 격리된 환경(REPL Sandbox) 내에서만 돕니다.

### **3. 스케줄링 및 제어 (Scheduling & Control)**

*   **OS Kernel의 역할:** 여러 프로세스에 CPU 시간을 분배하고 실행 순서를 제어합니다.
*   **MBAK Kernel의 역할:** **재귀적 작업(Task)의 순서를 제어**합니다.
    *   에이전트가 하위 에이전트를 `recurse()`로 호출할 때, 커널은 스택을 관리하고, 깊이(Depth) 제한을 걸고, 무한 루프를 방지합니다.
    *   Verification 실패 시 재시도(Retry) 횟수를 제어합니다.

### **4. 시스템 호출 인터페이스 (Syscall Interface)**

*   **OS Kernel의 역할:** `read()`, `write()`, `fork()` 같은 표준화된 인터페이스(Syscall)를 제공합니다.
*   **MBAK Kernel의 역할:** **Algebraic Effects**가 바로 Syscall입니다.
    *   `perform(WriteFile)`
    *   `perform(GenerateCode)`
    *   `perform(Verify)`
    *   이들은 에이전트가 커널에 요청하는 표준화된 시스템 호출입니다.

### **결론: 왜 "Framework"가 아니고 "Kernel"인가?**

*   **Framework**는 개발자가 코드를 채워 넣는 **'틀'**입니다. (예: LangChain - 개발자가 Chain을 짬)
*   **Kernel**은 실행을 통제하고 자원을 관리하는 **'엔진'**입니다. (예: MBAK - 에이전트가 Spec을 던지면 커널이 안전하게 돌림)

우리는 에이전트가 마음대로 행동하지 못하도록 **엄격하게 통제하고(Runtime), 자원을 추상화(Effects)하고, 안전을 보장(Verification)**하기 때문에 **"Build-Agent Kernel"**이라는 용어가 정확합니다.

**즉, 우리는 "에이전트를 위한 운영체제(OS)의 핵심부"를 만든 것입니다.**