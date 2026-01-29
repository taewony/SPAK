 **Step-wise Agent System 전체 구조설계서 (POC v1)**

사용자 의도에 맞게:

* ✅ step-wise execution (human-in-the-loop)
* ✅ LLM interaction point를 DSL step에 명시
* ✅ LLM call은 즉시 실행하지 않고 “질의 생성 → 중단 → 다음 step에서 응답 반영”
* ✅ Ollama / Cloud / Gemini CLI simulation switchable
* ✅ Lark 기반 DSL parsing
* ✅ Python VM 기반 deterministic execution
* ✅ Trace JSON file logging
* ✅ Outer loop가 DSL grammar + script 자체를 개선

---

# ✅ SPAK-POC Agent System 구조설계서 (Step-wise Dual Loop)

## 0. Design Goals

| 항목              | 설계 원칙                                        |
| --------------- | -------------------------------------------- |
| Execution       | Deterministic kernel + explicit step machine |
| Reasoning       | LLM은 only at DSL-declared interaction points |
| Autonomy        | step-wise → semi-auto → full auto 전환 가능      |
| Learning        | Outer loop가 DSL + script 구조 수정               |
| Reproducibility | Trace → DSL recoverable                      |
| Modularity      | LLM backend hot-swap 가능                      |

---

# 1. 전체 시스템 계층 구조

```
┌──────────────────────────────────────────────┐
│                OUTER LOOP (LLM)              │
│  - Trace analysis                            │
│  - DSL grammar evolution                     │
│  - DSL script patch generation               │
└───────────────▲──────────────────────────────┘
                │ DSL_patch
                │
┌───────────────┴──────────────────────────────┐
│              AGENT KERNEL (Python)           │
│  - DSL Parser (Lark)                         │
│  - DSL → AST                                 │
│  - Step Scheduler                            │
│  - Invariant Checker                         │
│  - LLM Interaction Broker                    │
│  - Tool Dispatcher                           │
│  - Trace Logger (JSON files)                 │
└───────────────▲──────────────────────────────┘
                │
                │ step execution
                │
┌───────────────┴──────────────────────────────┐
│          STEP-WISE AGENT LOOP (VM)           │
│  - Execute until LLM step encountered        │
│  - Emit LLM query & suspend                  │
│  - Resume on user trigger                    │
└───────────────▲──────────────────────────────┘
                │
                │ LLM calls
┌───────────────┴──────────────────────────────┐
│              LLM BACKENDS                    │
│  - Ollama (local)                            │
│  - Cloud API                                 │
│  - Gemini CLI simulator                      │
└──────────────────────────────────────────────┘
```

---

# 2. DSL 설계: Interaction Point 명시

## 2.1 DSL 핵심 개념

DSL은 반드시 다음을 포함:

* task
* step
* llm_call step
* tool_call step
* invariant
* success criteria

---

## 2.2 DSL 예시 (v0)

```dsl
task BuildFeature {

  step s1: tool.run {
    cmd: "pytest tests/"
  }

  step s2: llm.query {
    role: "coder"
    prompt_template: "Fix failing tests:\n{{last_tool_output}}"
    output_var: fix_patch
  }

  step s3: tool.apply_patch {
    patch: fix_patch
  }

  success:
    file_contains("result.txt", "PASS")
}
```

---

## 2.3 LLM Interaction Semantics

### llm.query step 의미:

Kernel 행동:

1. context assemble
2. prompt render
3. trace에 query 저장
4. execution suspend
5. CLI returns control to user

Trace:

```json
{
  "step": "s2",
  "type": "llm.query",
  "prompt": "...",
  "status": "waiting_response"
}
```

---

## 2.4 Resume Semantics

User:

```bash
spak step --resume
```

Kernel:

* LLM backend 호출
* response 저장
* variable binding
* step 완료
* 다음 step 계속

---

# 3. Kernel 내부 구조

## 3.1 Modules

```
kernel/
 ├─ dsl/
 │   ├─ grammar.lark
 │   ├─ parser.py
 │   └─ ast.py
 ├─ vm/
 │   ├─ step_machine.py
 │   ├─ context.py
 │   └─ variables.py
 ├─ llm/
 │   ├─ backend_base.py
 │   ├─ ollama.py
 │   ├─ cloud.py
 │   └─ gemini_cli_sim.py
 ├─ tools/
 │   ├─ shell.py
 │   ├─ fs.py
 │   └─ patch.py
 ├─ invariant/
 │   └─ checker.py
 ├─ trace/
 │   └─ logger.py
 └─ main.py
```

---

## 3.2 Step Machine FSM

```
READY
 └─> RUNNING
        ├─ tool_step → continue
        ├─ llm_step → SUSPEND
        └─ invariant_fail → FAIL
SUSPEND
 └─> WAIT_INPUT
        └─ resume → RUNNING
SUCCESS | FAIL
```

---

## 3.3 Context Model

```python
class ExecutionContext:
    variables: dict
    last_tool_output: str
    step_pointer: int
    task_state: enum
```

Context snapshot도 trace에 저장 → rollback 가능.

---

# 4. LLM Backend Abstraction

## 4.1 Interface

```python
class LLMBackend:
    def generate(self, prompt: str, context: dict) -> str:
        ...
```

---

## 4.2 Config Example

```yaml
llm:
  mode: ollama
  model: qwen2.5:7b
```

또는

```yaml
llm:
  mode: gemini_cli_sim
```

---

## 4.3 Gemini CLI Simulation Mode

실제 LLM 호출 대신:

* prompt 출력
* user가 직접 답변 입력
* trace에 response 저장

→ 디버깅 및 실험에 매우 중요

---

# 5. Trace Logging 구조 (JSON)

## 5.1 파일 단위

```
runs/
 └─ 2026-01-28_12-30-01/
      ├─ trace.json
      ├─ context.json
      └─ metrics.json
```

---

## 5.2 Trace Event Schema

```json
{
  "timestamp": "...",
  "task": "BuildFeature",
  "step": "s2",
  "type": "llm.query",
  "prompt": "...",
  "response": null,
  "status": "waiting"
}
```

---

## 5.3 Recoverability 보장

Trace → reconstruct 가능:

* step pointer
* variables
* pending llm query

---

# 6. Outer Loop: DSL Self-Improvement

Outer loop LLM input:

* grammar.lark
* current DSL script
* trace logs
* invariant violations
* performance metrics

Output:

* grammar patch (optional)
* DSL script patch

---

## 6.1 Outer Loop Prompt Contract

LLM에게 요구:

```
Return:
1. grammar_patch (diff)
2. dsl_patch (diff)
3. rationale
```

Kernel은:

* patch 적용
* schema re-validate
* next run

---

# 7. Autonomous Mode Transition

config:

```yaml
execution:
  mode: stepwise | semi_auto | full_auto
```

| Mode      | Behavior                      |
| --------- | ----------------------------- |
| stepwise  | llm.query마다 중단                |
| semi_auto | N step 자동                     |
| full_auto | continuous until success/fail |

---

# 8. 왜 이 구조가 연구적으로 중요한가

이 구조는:

* agent ≠ LLM
* system behavior = DSL + Kernel
* LLM = hypothesis generator

즉:

> Agent Engineering → Search over Programs

으로 전환됩니다.

특히 GPU task orchestration, scheduling DSL, invariant verification까지 그대로 확장 가능합니다.

---

# 9. POC 구현 우선순위 (권장)

### Phase 1 (2~3일)

* Lark grammar
* step machine
* tool step
* trace logging

### Phase 2

* llm.query step
* Ollama backend
* stepwise resume

### Phase 3

* invariant checker
* outer loop patch generation

---

# 10. 다음 단계 제안
**grammar + AST + step VM 구조**부터 잡아야 전체가 안정됩니다.

### 1. StepMachine Python skeleton

### 2. Gemini CLI simulation mode test

### 3. DSL grammar.lark 초안

### 4. LLM backend interface + Ollama impl

### 5. Trace JSON schema

### LAST. The effectiveness check about this Dual Loop DSL centric Agent System