# **1\. “agent.dsl을 최소 서술(minimal description)로 자동 정규화**

이건 한마디로 말하면:

**agent의 의도를 유지하면서,**  
**복원·비교·검증에 필요한 정보만 남기고**  
**나머지를 모두 제거한 \*표준형(spec normal form)\*을 만드는 과정**야.

수학에서의 *minimal generating set* 이나  
컴파일러에서의 \*IR (Intermediate Representation)\*에 가깝다.

---

## **1.1 왜 이게 필요한가?**

agent.dsl은 보통:

* 사람마다 다르게 씀  
* 중복 설명이 많음  
* 암묵적 가정이 섞임  
* domain-specific jargon이 많음

이 상태로는:

* trace ↔ spec alignment 비교 ❌  
* Bayesian 업데이트 ❌  
* 재현성 평가 ❌

그래서 \*\*“의미 보존 \+ 서술 최소화”\*\*가 필요해.

---

## **1.2 Minimal Agent Spec (MAS)의 정의**

**MAS는 다음 성질을 만족해야 한다:**

1. ❌ 실행 순서(step-by-step)를 강제하지 않는다  
2. ❌ LLM 내부 reasoning을 가정하지 않는다  
3. ✅ Task / Item / Invariant / Criteria만 남긴다  
4. ✅ trace로부터 *역추론 가능*해야 한다  
5. ✅ 다른 LLM이 다시 실행 가능한 수준이다

---

## **1.3 자동 정규화 프로세스 (개념적 파이프라인)**

Raw agent.dsl  
   ↓  
Semantic Parsing (LLM-assisted)  
   ↓  
Redundancy Elimination  
   ↓  
Implicit Assumption Extraction  
   ↓  
Constraint / Invariant Canonicalization  
   ↓  
Minimal Agent Spec (MAS)

### **핵심 포인트**

* **LLM은 “요약”이 아니라 “정규화(normalization)”를 한다**  
* 표현은 줄이되, *구조는 더 명확해진다*

---

## **1.4 MAS의 canonical form (권장)**

AGENT\_SPEC\_NORMAL\_FORM {

  INTENT { ... }

  TASKS { ... }

  ITEMS { ... }

  INVARIANTS { ... }

  SUCCESS\_CRITERIA { ... }

}

이 형식은:

* LLM-friendly  
* diff-friendly  
* alignment metric-friendly

---

# **2\. 이제 본론: 주어진 3개 domain을 다루는 agent loop \+ kernel 설계**

당신이 준 intent들을 다시 보자:

1. 🌐 Web source merge agent  
2. 📄 Fact-based extraction agent  
3. ⚡ GPU kernel synthesis & validation agent

이 셋은 **표면은 다르지만 kernel 관점에서는 공통 구조**를 가진다.

---

## **3\. 공통 추상: “Evidence-Grounded Synthesis Agent”**

이게 핵심 통찰이야.

### **공통점**

* heterogeneous input  
* intermediate reasoning 필요  
* verification 단계 존재  
* 결과는 artifact (page / doc / code)

즉:

**Observe → Propose → Validate → Synthesize**

---

## **4\. 이 domain들을 위한 Minimal Agent Spec (정규화된 형태)**

### **4.1 Intent Layer (공통)**

INTENT {  
  "Produce a synthesized artifact from heterogeneous inputs  
   while preserving semantic fidelity and verifiable grounding."  
}

---

### **4.2 Task Layer (kernel-reusable)**

TASK DecomposeInput  
TASK ExtractEvidence  
TASK ProposeCandidate  
TASK ValidateCandidate  
TASK SynthesizeFinalArtifact

💡 GPU agent에서도:

* ExtractEvidence \= access pattern / memory traffic  
* Validate \= benchmark

---

### **4.3 Item Layer (domain-agnostic \+ specialization)**

ITEM Evidence {  
  source: Reference  
  content: Text  
}

ITEM Candidate {  
  representation: Text | Code | StructuredData  
  assumptions: List\<Text\>  
}

ITEM ValidationResult {  
  passed: Boolean  
  metrics: Optional\<Map\>  
}

---

### **4.4 Invariants (agent 성격의 핵심)**

INVARIANTS {  
  "All outputs must be traceable to extracted evidence."  
  "No unverifiable extrapolation is allowed."  
  "Validation must precede final synthesis."  
}

GPU domain에선 자연스럽게:

* “Performance claims must be empirically validated.”

---

### **4.5 Success Criteria**

SUCCESS\_CRITERIA {  
  semantic\_fidelity: "No loss of core meaning from inputs"  
  grounding: "Every claim/code decision linked to evidence"  
  efficiency: "Within allowed execution/tool budget"  
}

---

## **5\. 이제 핵심: 이 agent를 실행하는 Kernel 요구사항**

이제 kernel.spec 관점으로 내려가자.

---

## **6\. Kernel의 필수 책임 (요구사항)**

### **6.1 Trace-Complete Execution**

* 모든 Task transition 기록  
* 모든 Item 생성/변형 기록  
* 모든 Tool / Function call 기록

👉 trace는 **재현의 원자 단위**

---

### **6.2 LLM Interaction Model**

response \= f(context\_snapshot, query\_fragment)

* LLM은 stateless  
* kernel이 context를 관리  
* context에는:  
  * active TASK  
  * available ITEMS  
  * applicable INVARIANTS

---

### **6.3 Plan is advisory, not authoritative**

* LLM이 Plan을 제안  
* kernel이 invariant 위반 여부 검사  
* 위반 시:  
  * plan 수정 요청  
  * 또는 task 재진입

---

### **6.4 Validation Gate (중요)**

kernel은 반드시 다음을 보장해야 한다:

**ValidateCandidate Task를 건너뛰고**  
**SynthesizeFinalArtifact로 갈 수 없다**

이건 kernel-level rule이어야 함.

---

## **7\. Kernel Loop (개념적)**

while not SUCCESS:  
  provide current TASK \+ ITEMS to LLM  
  receive proposal (Plan / Candidate)  
  check invariants  
  if requires tool:  
    execute tool  
    log result  
  if validation failed:  
    re-enter ProposeCandidate

LLM은 “지능”이고  
Kernel은 “법률”이다.

---

## **8\. 이 설계의 진짜 의미**

이 구조를 쓰면:

* agent.dsl → 실행 → trace  
* trace \+ kernel.spec → agent.dsl (복원)  
* 복원된 spec ↔ 원 spec 비교 가능

즉:

**agent는 이제 ‘설계된 물체’가 된다**

학습된 블랙박스가 아니라.

---

## **9\. 마지막 요약 (중요)**

### **“Minimal description 정규화”란?**

* agent.dsl을 **복원 가능한 설계 최소 단위**로 줄이는 것

### **Kernel 설계의 핵심은?**

* LLM을 자유롭게 두되  
* **Invariant \+ Validation을 kernel이 강제**

### **이게 궁극적으로 여는 문은?**

* Self-reflective  
* Self-improving  
* Auditable  
* Reproducible  
  **Autonomous Engineering System**

