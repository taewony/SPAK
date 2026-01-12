
교수님께서 구상하시는 아이디어는 **"AI-Native Software Engineering (AISE)"**의 핵심인 "자연어 명세(Spec)가 곧 실행 가능한 코드(Code)가 되는 파이프라인"을 구축하는 것입니다. 이는 전통적인 **문학적 프로그래밍(Literate Programming)**을 LLM 시대에 맞게 재해석한 매우 훌륭한 연구 주제입니다.

학생들이 개별 에이전트를 만드는 동안, 교수님께서는 이들을 관장하는 **"Meta-Build Agent"**를 설계하고 논문으로 발전시킬 수 있도록, **Markdown 중심의 재귀적 빌드 시스템(Markdown-Driven Recursive Build System)** 방법론을 제안합니다. 이 방법론의 이름을 가칭 \*\*"MaDD (Markdown-Driven Development) Pipeline"\*\*이라 가이드 함. 

---

### **1\. 핵심 철학: "Markdown is the Intermediate Representation (IR)"**

컴파일러 수업에서 IR(중간 표현)이 소스 코드와 기계어 사이를 잇듯이, 여기서는 **Markdown 문서가 사람의 의도(Intent)와 Python 코드 사이의 IR 역할**을 합니다.

* **사람의 역할:** Markdown 문서에 의도, 설계, 제약조건을 작성.  
* **Build Agent의 역할:** Markdown을 파싱하여 Code를 생성하고, 실행 결과를 다시 Markdown에 기록(Traceability).  
* **결과물:** 문서와 코드가 완벽하게 동기화된 Repository.

---

### **2\. 디렉토리 구조 및 파일 시스템 설계**

이 시스템은 명세(Spec)와 구현(Impl)이 1:1로 매핑되는 구조를 가집니다.

Plaintext

Project\_Root/  
├── playbook/              \# \[Source of Truth\] 사람이 작성 & AI가 업데이트  
│   ├── 00\_master\_plan.md  \# 전체 시스템 아키텍처 및 진행 상황  
│   ├── 01\_planner.md      \# 모듈별 상세 명세  
│   ├── 02\_worker.md  
│   └── logs/              \# AI가 수행한 사고 과정(CoT) 기록  
├── src/                   \# \[Generated\] Build Agent가 생성하는 코드  
│   ├── \_\_init\_\_.py  
│   ├── planner.py  
│   └── worker.py  
└── build\_agent.py         \# \[The Builder\] 교수님이 작성할 핵심 엔진

---

### **3\. Markdown 문서 형식화 (Spec Standardization)**

AI가 정확하게 코드로 변환하기 위해서는 Markdown 작성 규칙(Protocol)이 필요합니다. 이를 논문에서는 \*\*"Prompt Schema"\*\*라고 정의할 수 있습니다.

**예시: playbook/01\_planner.md**

Markdown

\# Module: Planner Agent  
Status: \[Draft | Implementing | Tested\]  
Version: 1.0

\#\# 1\. Intent (의도)  
사용자의 자연어 요청을 입력받아, 실행 가능한 하위 Task List(JSON)로 분해한다.

\#\# 2\. Architecture & Interface  
\- **\*\*Input:\*\*** \`user\_query\` (str)  
\- **\*\*Output:\*\*** \`task\_list\` (List\[Dict\])  
\- **\*\*Dependencies:\*\*** \`ollama\_client\`

\#\# 3\. Logic Description (상세 설계)  
1\. System Prompt에 현재 날짜와 사용 가능한 도구 목록을 주입한다.  
2\. LLM에게 "Step-by-step"으로 생각하도록 지시한다.  
3\. 출력은 반드시 JSON 포맷이어야 한다. 예외 발생 시 재시도 로직을 포함한다.

\#\# 4\. Constraints (제약 사항)  
\- Ollama 모델은 \`llama3\`를 기본으로 사용한다.  
\- Timeout은 30초로 설정한다.

\---  
\#\# 5\. Generated Code (Auto-filled by Build Agent)  
\> \[\!NOTE\]  
\> 아래 코드는 Build Agent에 의해 2026-01-11에 생성되었습니다.  
(이곳에 파이썬 코드가 자동으로 링크되거나 요약되어 들어갑니다)

---

### **4\. Build Agent의 단계적 동작 (The Recursive Process)**

교수님께서 만드실 build\_agent.py는 다음 4단계 루프를 돕니다. 이것이 시스템의 핵심 알고리즘입니다.

**Step 1: Spec Parsing (명세 해석)**

* playbook/\*.md 파일들을 읽습니다.  
* Regex나 LLM을 이용해 Intent, Interface, Logic 섹션을 추출하여 Context에 적재합니다.

**Step 2: Code Synthesis (코드 합성)**

* 추출된 정보를 바탕으로 LLM(Ollama)에게 코드를 요청합니다.  
* *Prompt:* "너는 Senior Python 개발자야. 다음 명세(Intent, Logic)를 바탕으로 src/planner.py를 작성해. Dependencies를 고려해서 import 구문을 작성해."

**Step 3: Verification & Feedback (검증 및 피드백)**

* 생성된 코드를 가상 환경에서 실행(Linting 또는 단위 테스트)해봅니다.  
* 에러 발생 시, 에러 로그를 다시 LLM에게 보여주고 코드를 수정(Refactoring)합니다.

**Step 4: Back-Annotation (역방향 기록)**

* 성공적으로 코드가 생성되면, Markdown 문서의 Status를 \[Implemented\]로 변경합니다.  
* 생성된 코드의 경로와 핵심 주석을 Markdown의 Generated Code 섹션에 업데이트합니다. **(이 부분이 Traceability의 핵심입니다)**

---

### **5\. 논문 및 연구 관점의 차별화 포인트**

이 시스템을 논문으로 발전시킬 때 강조해야 할 **Contribution**은 다음과 같습니다.

1. **Dual-Process Engineering (이중 공정 엔지니어링):**  
   * 기존 SE는 코드만 관리했지만, 본 제안은 **Natural Language Spec**과 **Executable Code**가 상호 의존성을 가지며 진화(Co-evolution)하는 모델을 제시합니다.  
2. **Traceability by Design (설계에 의한 추적성):**  
   * 코드가 왜 이렇게 짜였는지에 대한 "이유(Why)"가 Markdown에 영구히 보존됩니다. 코드 변경 시 Markdown이 먼저 변경되어야 하므로 문서의 최신성이 보장됩니다.  
3. **Local LLM Feasibility:**  
   * 거대 모델(GPT-4) 없이, 명확하게 구조화된(Structured) Markdown Context만 있다면 Local LLM(Ollama)으로도 복잡한 시스템 구축이 가능함을 증명합니다.

### **6\. 교수님의 Next Step (Action Plan)**

가장 먼저 하셔야 할 일은 \*\*Build Agent의 "Seed(씨앗)"\*\*를 만드는 것입니다.

1. **Playbook 템플릿 정의:** 위에서 제안한 Markdown 형식을 확정합니다.  
2. **Bootstrap:** build\_agent.py 자체의 명세도 Markdown(playbook/00\_builder.md)으로 먼저 작성해 보십시오.  
3. **Manual Start:** 첫 번째 빌드 에이전트 코드는 손수(또는 LLM 도움을 받아) 작성하시되, 이 에이전트가 **"자신의 코드를 읽고 스스로 개선할 수 있는지"** 테스트해보십시오. (재귀적 개선)

이 구조가 마음에 드신다면, \*\*"Build Agent가 사용할 Markdown 템플릿의 상세 양식"\*\*이나 **"Build Agent의 초기 Python 코드 스켈레톤"** 중 어느 것을 먼저 만들어 드릴까요?

