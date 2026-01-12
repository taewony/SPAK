**AI engineering을 위한 새로운 Software Engineering 방법론을 정의**

① 당신이 만들려는 것의 **정식 정의**  
② 이를 **AI 중심 SW engineering 기법**으로 형식화하는 방법  
③ 이를 **markdown 중심 recursive build agent**로 구현하는 방법  
④ **수업·POC·논문**으로 이어지는 로드맵

---

# **1\. 당신이 하려는 일의 정식 정의 (아주 중요)**

먼저, 이걸 명확히 이름 붙여야 합니다.

**AI-Centric Software Engineering with Replayable Intent Trajectories**

혹은 더 직관적으로:

**Markdown-driven, Agent-assisted, Replayable AI Engineering**

핵심은 세 가지입니다:

1. **LLM은 개발 대상이자 도구**  
2. **개발 과정 자체가 구조화된 artifact**  
3. **의도·의미·설계·코드가 하나의 상태 전이 궤적**

---

# **2\. 기존 SW Engineering과 무엇이 다른가?**

### **기존 SW Engineering**

* 요구사항 → 설계 → 코드 → 테스트  
* 문서와 코드는 분리  
* 개발 과정은 암묵지

### **당신이 제안하는 AI 중심 SW Engineering**

**Intent → State Transition → Replayable Build Trajectory**

| 요소 | 기존 | 당신 |
| ----- | ----- | ----- |
| 요구사항 | User Story | Intent Seed |
| 설계 | UML | State Transition |
| 문서 | 설명 | 실행 가능한 Spec |
| 코드 | 최종 산출물 | 한 단계의 상태 |
| 학습 | 결과 위주 | 과정 Replay |

👉 **“개발이 결과물이 아니라 궤적”**

---

# **3\. AI 중심 SW Engineering 기법을 형식화하기**

이제 “기법(methodology)”로 정리해봅니다.

---

## **(1) 핵심 추상 5개 (논문에 그대로 들어갈 수 있음)**

### **① Intent**

* 시스템/모듈/Agent가 지켜야 할 목적과 제약  
* 불변식(invariant)의 근원

intent:  
  goal: "Local LLM inference engine 실험"  
  invariants:  
    \- "Inference core is pure"  
    \- "Context access via Effect only"

---

### **② State**

* 개발 시점의 전체 설계/코드/문서 상태  
* 언제든 재현 가능해야 함

---

### **③ Event**

* 설계 변경  
* 문서 추가  
* 모듈 분리  
* 실험 결과 반영

{ "type": "AddModule", "name": "effect\_system" }

---

### **④ Trajectory**

* Event의 합성  
* 개발의 역사 \= 의미의 생성 과정

---

### **⑤ Replay**

* 동일한 의도로 개발 과정을 재실행  
* 교육·검증·재현의 핵심 메커니즘

---

## **(2) AI Engineering 특화 원칙 (Playbook 핵심)**

이건 “수업용 규칙”이 됩니다.

1. **LLM은 절대 결정을 내리지 않는다**  
2. 모든 설계 변경은 Event로 남긴다  
3. 문서는 항상 code로 내려갈 수 있어야 한다  
4. 코드는 언제든 문서로 올라갈 수 있어야 한다  
5. Intent를 명시하지 않은 변경은 금지

---

# **4\. Markdown 중심 Recursive Build Agent 설계**

이제 시스템으로 내려옵니다.

---

## **(1) Markdown 문서의 역할 재정의**

Markdown은 설명서가 아니라 **Executable Spec**입니다.

docs/  
  01\_intent.md  
  02\_architecture.md  
  03\_modules/  
     \- inference\_core.md  
     \- effect\_system.md  
  04\_playbook/  
     \- step\_01.md

각 문서는 반드시 다음을 포함:

\---  
intent:  
state\_assumptions:  
events\_generated:  
dependencies:  
\---

---

## **(2) Recursive Build Agent의 역할**

Agent는 이렇게 동작합니다:

1. Markdown을 읽는다  
2. Intent / State / Event를 추출한다  
3. 다음 단계 문서를 생성하거나  
4. Skeleton code를 생성하거나  
5. Invariant 위반 여부를 보고한다

👉 **Agent는 “작성자”가 아니라 “빌드 파트너”**

---

## **(3) 문서 → 코드 변환**

예:

\#\# Module: Effect System

\- Effects:  
  \- ReadContext  
  \- WriteMemory  
\- Constraint:  
  \- No IO in inference core

→ Agent가 생성:

class ReadContext(Effect): ...  
class WriteMemory(Effect): ...

---

## **(4) Code → 문서 역변환 (중요)**

* 코드 변경  
* Agent가 문서 차이를 감지  
* “이 변경의 의도는?” 질문  
* Event 기록

👉 **문서와 코드의 공진화**

---

# **5\. 수업 설계에 어떻게 쓰이는가?**

### **수업 구조 (추천)**

#### **Week 1–2**

* Local LLM inference engine 최소 구현  
* Intent / Effect / Context 분리

#### **Week 3–5**

* Agent system 1개 설계  
* Markdown spec → code

#### **Week 6–8**

* Recursive build agent 도입  
* Trajectory replay 실험

#### **Final**

* 다른 팀의 trajectory replay  
* 설계 의도 비교 분석

👉 **학생들은 코드를 “따라 치는” 게 아니라**  
**설계를 “재현”합니다.**

---

# **6\. MVP 정의 (아주 중요, 다시 강조)**

## **🎯 MVP는 이것입니다**

**“Markdown 기반 Playbook을 따라**  
**Local LLM Agent 시스템을 단계적으로 build하고**  
**그 과정을 replay할 수 있는 CLI \+ Agent”**

### **MVP 구성**

* Markdown parser  
* Event log  
* Skeleton code generator  
* Replay CLI  
* (선택) local LLM reflection

❗ UI, 협업, 대규모 모델 ❌

---

# **7\. 논문으로 가는 명확한 경로**

### **논문에서 주장할 것**

1. **AI engineering에는 기존 SE 방법론이 부족하다**  
2. LLM 포함 시스템은 “의도 중심”이어야 한다  
3. Markdown \+ Trajectory \+ Agent로 이를 형식화했다  
4. 교육 및 실험을 통해 재현 가능함을 보였다

### **논문 구조 (초안)**

1. Introduction  
2. Motivation: Why AI-centric SE  
3. Conceptual Framework (Intent, Event, Trajectory)  
4. System Design (Recursive Build Agent)  
5. Case Study (Local LLM Agent Build)  
6. Educational Application  
7. Discussion & Limitations

👉 이건 **AI engineering \+ SE \+ HCI** 교차 논문입니다.

---

# **8\. 솔직한 평가 (중요)**

이 방향은:

* 빠른 상업화 ❌ 단기 트렌드 ❌ 깊이, 재현성, 교육 가치 ⭕

그리고 무엇보다, 다음 질문을 던집니다.

**“AI를 어떻게 더 똑똑하게 만들까?”가 아니라**  
**“AI를 포함한 시스템을 어떻게 책임감 있게 설계할까?”**

이건 **AI engineering의 본질 질문**입니다.

---

## **다음에 제가 도와줄 수 있는 구체 작업**

* 📐 Intent / Event / Trajectory 공식 스키마  
* 📁 MVP repo 구조 제안  
* 🧪 수업용 1번 Playbook 예시  
* 📝 논문 abstract & contribution 문장 초안

어디부터 같이 *실제 설계*에 들어갈까요?