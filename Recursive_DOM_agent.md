네, **Windows PC 환경**에서 **Ollama의 Tool Calling(Function Calling)** 기능을 활용하면 이 아이디어를 완벽하게 구현할 수 있습니다. 특히 최신 Llama 3.1이나 Llama 3.2 모델은 도구 사용 능력이 매우 뛰어납니다.

학생들을 위한 **"Project: Recursive DOM Agent (RDA)"** 개발 개요서(Outline)를 작성해 드립니다. 이 문서를 커리큘럼의 기초 자료로 활용하세요.

---

# **📂 Project: Recursive DOM Agent (RDA) on Windows**

## **1\. 프로젝트 개요 (Project Overview)**

* **목표:** LLM의 Context Window 한계를 극복하기 위해, 비정형 텍스트 데이터를 **DOM(Document Object Model)** 구조로 변환하고, LLM이 **CSS Selector**를 도구(Tool)로 사용하여 필요한 정보를 능동적으로 탐색 및 재귀적(Recursive)으로 처리하는 에이전트를 개발한다.  
* **핵심 개념:**  
  * **Context as a Database:** 텍스트를 읽는 대상이 아닌 '조회(Query)'의 대상으로 취급.  
  * **Tool Use:** LLM이 스스로 Python 함수(select, map\_reduce)를 호출.  
  * **Recursive Processing:** 큰 문제를 작은 단위(Node)로 쪼개어 하위 에이전트에게 위임.

## **2\. 개발 환경 (Environment Setup)**

학생들이 보유한 Windows PC(RTX 4070)를 기준으로 합니다.

* **OS:** Windows 10/11 (PowerShell 또는 WSL2 권장)  
* **Language:** Python 3.10 이상  
* **Core Engine:** [Ollama for Windows](https://ollama.com/download/windows)  
* **Target Model:**  
  * llama3.1:8b (Main Controller \- 도구 사용 능력이 좋음)  
  * llama3.2:3b (Sub-Worker \- 단순 요약 등 빠른 처리에 적합)  
* **Key Libraries:**  
  * ollama: LLM 통신 및 도구 호출  
  * beautifulsoup4 & lxml: DOM 파싱 및 CSS Selector 엔진  
  * rich: 터미널 UI 시각화 (Tree 구조 출력용)

---

## **3\. 주차별 개발 단계 (Development Roadmap)**

### **Phase 1: 기반 환경 구축 (Infrastructure)**

Ollama를 설치하고 Python에서 제어하는 기초 단계입니다.

1. **Ollama Setup:** Windows용 Ollama 설치 및 GPU 가속 확인 (ollama run llama3.1 실행).  
2. **API Binding:** Python import ollama를 통해 텍스트 생성 테스트.  
3. **Data Preparation:** 긴 Markdown 문서(예: 강의록, 매뉴얼)를 준비하고, 이를 Python 스크립트로 **HTML/XML 포맷으로 변환**하여 저장하는 전처리기(Preprocessor) 작성.

### **Phase 2: 가상 환경(Environment) 클래스 구현**

LLM이 접속할 "데이터베이스(DOM)"를 구축합니다. LLM 없이 순수 Python 로직으로 작동해야 합니다.

1. **DOM Loader:** BeautifulSoup을 이용해 XML 데이터를 메모리에 로드.  
2. **Tools Implementation:**  
   * get\_structure(): 문서의 목차(ID, 태그, Title)만 리턴하는 함수 (토큰 절약용).  
   * read\_node(selector): 특정 CSS Selector에 해당하는 본문 텍스트를 리턴하는 함수.  
   * *Unit Test:* 교수가 제공한 selector를 입력했을 때 정확한 텍스트가 나오는지 테스트.

### **Phase 3: Ollama Tool Binding (The Brain)**

Llama 3.1 모델에게 Phase 2에서 만든 파이썬 함수들을 "도구"로 쥐어주는 단계입니다.

1. **Tool Definition:** get\_structure, read\_node 함수를 Ollama가 이해할 수 있는 JSON Schema 형태로 정의.  
2. **Chat Loop (REPL):**  
   * 사용자 질문 입력 \-\> Ollama에게 전달 (with Tools).  
   * Ollama가 도구 호출 요청(tool\_calls) \-\> Python이 해당 함수 실행.  
   * 함수 실행 결과 \-\> 다시 Ollama에게 전달 (Role: tool).  
   * Ollama가 최종 답변 생성.

### **Phase 4: 재귀적(Recursive) 기능 구현 (The Magic)**

단순 조회를 넘어, 영상에서 본 RLM(Recursive LM) 개념을 적용합니다.

1. **map\_reduce 도구 추가:**  
   * 입력: selector (예: section.case\_study), query (예: "이 사례의 핵심 원인 분석해줘").  
   * 동작:  
     1. Selector로 $N$개의 노드를 찾음.  
     2. for 루프를 돌며 각 노드 내용에 대해 \*\*새로운 ollama.chat 세션(Sub-Agent)\*\*을 생성하여 질문.  
     3. $N$개의 답변을 리스트로 모아서 반환.  
2. **Controller Logic:** 메인 LLM이 "전체 챕터 요약" 요청을 받으면, 스스로 map\_reduce 도구를 호출하도록 유도.

### **Phase 5: 데모 및 시각화 (Visualization)**

1. **CLI Dashboard:** rich 라이브러리를 사용하여, 현재 LLM이 DOM 트리의 어느 부분을 보고 있는지, 어떤 하위 에이전트가 돌고 있는지 터미널에 시각적으로 표시.  
2. **Final Test:** "전체 매뉴얼에서 '보안'과 관련된 내용만 찾아서, 각 항목별 조치 사항을 표로 만들어줘"와 같은 복합 질의 수행.

---

## **4\. 핵심 코드 구조 예시 (Python Draft)**

학생들에게 제공할 수 있는 **Starter Code**의 골격입니다.

Python

import ollama  
from bs4 import BeautifulSoup  
import json

\# 1\. Environment (DOM Manager)  
class DocumentEnv:  
    def \_\_init\_\_(self, xml\_content):  
        self.soup \= BeautifulSoup(xml\_content, 'xml')  
      
    def get\_structure(self):  
        """문서의 뼈대(ID와 Title)만 반환"""  
        \# (구현 생략: 본문을 제외한 태그 구조만 문자열로 리턴)  
        pass

    def read\_content(self, selector):  
        """특정 노드의 내용을 조회"""  
        selected \= self.soup.select(selector)  
        return "\\n".join(\[tag.get\_text() for tag in selected\])

    def map\_reduce(self, selector, sub\_query):  
        """\[Recursive\] 각 노드에 대해 하위 에이전트 실행"""  
        results \= \[\]  
        targets \= self.soup.select(selector)  
          
        print(f"🔄 Spawning {len(targets)} sub-agents...")  
        for target in targets:  
            \# 하위 에이전트는 가벼운 모델(llama3.2) 사용 가능  
            response \= ollama.chat(  
                model='llama3.2',  
                messages=\[  
                    {'role': 'system', 'content': '분석가는 주어진 텍스트만 보고 답합니다.'},  
                    {'role': 'user', 'content': f"Context: {target.get\_text()}\\n\\nTask: {sub\_query}"}  
                \]  
            )  
            results.append(response\['message'\]\['content'\])  
        return json.dumps(results)

\# 2\. Tool Definitions for Ollama  
my\_tools \= \[  
    {  
        'type': 'function',  
        'function': {  
            'name': 'get\_structure',  
            'description': '문서의 전체 목차 구조를 확인합니다. 탐색 전에 반드시 먼저 호출해야 합니다.',  
            'parameters': {'type': 'object', 'properties': {}}  
        }  
    },  
    {  
        'type': 'function',  
        'function': {  
            'name': 'read\_content',  
            'description': 'CSS Selector를 사용하여 특정 섹션의 내용을 읽습니다.',  
            'parameters': {  
                'type': 'object',  
                'properties': {  
                    'selector': {'type': 'string', 'description': 'CSS Selector (예: chapter\#1 \> section)'}  
                },  
                'required': \['selector'\]  
            }  
        }  
    },  
    {  
        'type': 'function',  
        'function': {  
            'name': 'map\_reduce',  
            'description': '반복적인 요소들(예: 모든 섹션)에 대해 동일한 작업을 수행하고 결과를 모읍니다.',  
            'parameters': {  
                'type': 'object',  
                'properties': {  
                    'selector': {'type': 'string', 'description': '반복할 대상의 CSS Selector'},  
                    'sub\_query': {'type': 'string', 'description': '각 대상에게 수행할 지시사항'}  
                },  
                'required': \['selector', 'sub\_query'\]  
            }  
        }  
    }  
\]

\# 3\. Main REPL Loop (Simplified)  
def run\_agent(user\_query, env):  
    messages \= \[{'role': 'user', 'content': user\_query}\]  
      
    while True:  
        \# LLM에게 질문 \+ 도구 목록 전달  
        response \= ollama.chat(model='llama3.1', messages=messages, tools=my\_tools)  
        msg \= response\['message'\]  
          
        \# 도구 호출이 없으면 최종 답변으로 간주하고 종료  
        if not msg.get('tool\_calls'):  
            print(f"🤖 Agent: {msg\['content'\]}")  
            break  
              
        \# 도구 호출 처리  
        messages.append(msg) \# 대화 내역에 추가  
        for tool in msg\['tool\_calls'\]:  
            fn\_name \= tool\['function'\]\['name'\]  
            args \= tool\['function'\]\['arguments'\]  
            print(f"🛠️ Tool Call: {fn\_name}({args})")  
              
            \# 실제 파이썬 함수 실행  
            if fn\_name \== 'get\_structure':  
                result \= env.get\_structure()  
            elif fn\_name \== 'read\_content':  
                result \= env.read\_content(args\['selector'\])  
            elif fn\_name \== 'map\_reduce':  
                result \= env.map\_reduce(args\['selector'\], args\['sub\_query'\])  
              
            \# 결과 반환  
            messages.append({'role': 'tool', 'content': str(result)})

\# 실행 예시  
\# env \= DocumentEnv(xml\_data)  
\# run\_agent("챕터 1의 내용을 요약해줘", env)

---

## **5\. 기대 효과 및 평가 포인트**

* **Engineering:** 단순히 프롬프트만 치는 것이 아니라, LLM을 시스템의 모듈(함수)로 취급하는 사고방식을 배양.  
* **Optimization:** llama3.1:8b와 llama3.2:3b를 적재적소에 섞어 쓰는(Orchestrator-Worker 패턴) 최적화 경험.  
* **Interpretability:** LLM이 왜 그런 답을 냈는지, 어떤 Tool을 호출했고 어떤 데이터를 참조했는지 로그(Tool Call)를 통해 명확히 디버깅 가능.

이 프로젝트는 4070 PC 한 대만으로도 로컬에서 엔터프라이즈급 에이전트 아키텍처를 모사할 수 있는 훌륭한 실습이 될 것입니다.

아주 좋은 문제 설정이에요. 지금 질문은 \*\*“Recursive LM 논문에서 말하는 환경 E \+ context C를, DOM Tree \+ CSS selector처럼 구조화해서, REPL 기반으로 누적 추론을 수행하는 시스템을 어떻게 구현할 것인가?”\*\*이고,  
\*\*RTX 4070 / L40S \+ Ollama \+ 소형 LLaMA(3B급)\*\*을 전제로 한 **교육용 AI 엔지니어링 아키텍처**를 묻고 있습니다.

아래에서는 **개념적 대응 → 시스템 아키텍처 → 데이터 구조 → REPL/Selector 설계 → Ollama 연동 → 수업용 확장 포인트** 순서로 설명하겠습니다.

---

## **1\. 개념적 대응 (DOM/CSS ↔ Recursive LM)**

먼저 비유를 명확히 정리해보면 학생들에게도 설명이 쉬워집니다.

| Web 개념 | Recursive LM 개념 |
| ----- | ----- |
| DOM Tree | Context Tree (C-Tree) |
| DOM Node | Context Node (cᵢ) |
| CSS Selector | Context Selector DSL |
| querySelector | context.select() |
| JS 실행 환경 | Python REPL Environment E |
| innerHTML 변경 | context node transform / annotate |
| Virtual DOM diff | Context versioning / snapshot |

즉:

**Context C를 “긴 문자열”로 취급하지 않고,**  
**구조적·계층적·참조 가능한 Tree로 취급한다**  
→ 이것이 Recursive LM의 핵심 구현 포인트

---

## **2\. 전체 시스템 아키텍처 (교육용 현실 버전)**

┌──────────────────────────┐  
│  Student (REPL / UI)     │  
└────────────┬─────────────┘  
             ▼  
┌──────────────────────────┐  
│ Python REPL Environment  │  ← E  
│  \- context\_tree          │  
│  \- results\_store         │  
│  \- selector DSL          │  
│  \- call\_rlm()            │  
└────────────┬─────────────┘  
             ▼  
┌──────────────────────────┐  
│  Local LLM Backend       │  
│  (Ollama \+ LLaMA 3B)     │  
│                          │  
│  M(q, C\_selected)        │  
└──────────────────────────┘

핵심은:

* **Python REPL이 “Environment E”**  
* **Context는 Tree로 메모리에 상주**  
* **LLM은 도구처럼 호출**  
* **Recursive call은 Python에서 spawn**

---

## **3\. Context Tree 설계 (DOM-like)**

### **3.1 기본 Context Node**

from dataclasses import dataclass, field  
from typing import List, Dict, Any

@dataclass  
class ContextNode:  
    id: str  
    type: str                  \# e.g. "section", "paragraph", "claim"  
    text: str  
    children: List\["ContextNode"\] \= field(default\_factory=list)  
    meta: Dict\[str, Any\] \= field(default\_factory=dict)

예시 트리:

root  
 ├── c1 (Introduction)  
 │    ├── c1.1 (Motivation)  
 │    └── c1.2 (Problem Statement)  
 ├── c2 (Method)  
 │    ├── c2.1 (Recursive Call)  
 │    └── c2.2 (Environment E)  
 └── c3 (Experiments)

👉 논문, 코드, 로그, 대화 기록 모두 동일 구조로 저장 가능

---

## **4\. CSS-like Context Selector DSL**

### **4.1 최소한의 Selector 문법 (교육용)**

\#id  
type  
type \> type  
type\[type=section\]

### **4.2 Python 구현 예시**

class ContextTree:  
    def \_\_init\_\_(self, root: ContextNode):  
        self.root \= root

    def select(self, selector: str) \-\> List\[ContextNode\]:  
        \# 아주 단순한 구현 (수업용)  
        if selector.startswith("\#"):  
            return self.\_find\_by\_id(self.root, selector\[1:\])  
        else:  
            return self.\_find\_by\_type(self.root, selector)

    def \_find\_by\_id(self, node, id\_):  
        if node.id \== id\_:  
            return \[node\]  
        result \= \[\]  
        for c in node.children:  
            result.extend(self.\_find\_by\_id(c, id\_))  
        return result

    def \_find\_by\_type(self, node, type\_):  
        result \= \[\]  
        if node.type \== type\_:  
            result.append(node)  
        for c in node.children:  
            result.extend(self.\_find\_by\_type(c, type\_))  
        return result

학생 과제:

* `section > paragraph`  
* `paragraph[meta.score > 0.7]`  
* XPath-like selector 확장

---

## **5\. RLMM(q, C) 구현 구조**

### **5.1 Environment E (Python REPL)**

class RLMEnvironment:  
    def \_\_init\_\_(self, context\_tree, llm):  
        self.context\_tree \= context\_tree  
        self.llm \= llm  
        self.memory \= {}

    def call(self, query, selector):  
        nodes \= self.context\_tree.select(selector)  
        context\_text \= "\\n".join(n.text for n in nodes)

        response \= self.llm.generate(query, context\_text)  
        return response

---

## **6\. Recursive Call (논문의 핵심)**

### **6.1 Sub-RLM Spawn**

def recursive\_call(env, query, selector):  
    \# 1\. select sub-context  
    nodes \= env.context\_tree.select(selector)

    \# 2\. transform context (optional)  
    sub\_context \= ContextTree(  
        root=ContextNode(  
            id="subroot",  
            type="subcontext",  
            text="",  
            children=nodes  
        )  
    )

    \# 3\. new isolated environment  
    sub\_env \= RLMEnvironment(sub\_context, env.llm)

    \# 4\. run  
    result \= sub\_env.call(query, "paragraph")

    \# 5\. feed back  
    env.memory\[f"result\_{selector}"\] \= result  
    return result

이게 바로:

**RLMM(q̂, Ĉ) → 결과를 E로 반환**

---

## **7\. Ollama \+ LLaMA 3B 연동 (현실적)**

### **7.1 Ollama 실행**

ollama run llama3.2:3b

### **7.2 Python 호출 예시**

import subprocess  
import json

class OllamaLLM:  
    def generate(self, query, context):  
        prompt \= f"""  
Context:  
{context}

Query:  
{query}  
"""  
        result \= subprocess.run(  
            \["ollama", "run", "llama3.2:3b"\],  
            input=prompt,  
            text=True,  
            capture\_output=True  
        )  
        return result.stdout

👉 RTX 4070에서도 충분히 실습 가능  
👉 L40S 서버에서는 multi-user REPL 가능

---

## **8\. 누적 추론 & 결과 저장 (DOM diff 느낌)**

env.memory\["claims"\] \= \[\]  
env.memory\["claims"\].append({  
    "selector": "\#c2",  
    "result": output  
})

확장하면:

* 결과를 **새 ContextNode로 다시 트리에 삽입**  
* reasoning trace를 meta로 저장  
* versioned context (Git-like)

---

## **9\. 수업용 커리큘럼 확장 아이디어**

### **Level 1**

* Context Tree 만들기  
* Selector 구현  
* LLM 단일 호출

### **Level 2**

* Recursive call  
* Sub-context isolation  
* 결과 feed-back

### **Level 3**

* Context transform (summarize → store)  
* Agent-free planning (LLM이 selector 결정)

### **Level 4 (연구 수준)**

* Selector를 LLM이 생성  
* Cost-aware recursion  
* Context cache / eviction policy

---

## **10\. 핵심 메시지 (학생들에게)**

**Recursive LM은 “에이전트 프레임워크”가 아니라**  
**“LLM \+ 구조화된 메모리 \+ 실행 환경”의 문제다.**

DOM과 CSS를 이해하면,  
LLM Context Engineering도 이해할 수 있다.

---

원하시면 다음 단계로:

* **이 구조를 과제로 쓸 수 있는 미니 프로젝트**  
* **논문 Figure 1을 그대로 재현하는 코드**  
* **cuTile / GPU 수업과 연결하는 방식**  
* **Selector DSL을 formal language로 정의**

중 무엇을 더 깊게 볼지 정해봅시다.

아주 좋은 포인트를 짚으셨어요.  
\*\*“기술적으로 가능하다”\*\*보다 훨씬 중요한 질문이  
👉 **“학생들이 흥미를 잃지 않고, 실패를 감당하면서, 점진적으로 깊어질 수 있느냐”** 인데,  
Ollama는 이 조건을 **교육용 LLM 인프라**로 거의 최적에 가깝게 만족합니다.

아래에서는 **Ollama를 ‘도구’가 아니라 ‘수업 장치’로 쓰는 방법**을 중심으로 정리하겠습니다.

---

# **1\. Ollama를 수업에 쓰기 좋은 이유 (교육 관점)**

### **1️⃣ “내 컴퓨터에 AI가 있다”는 감각**

* OpenAI API와 달리  
  * ❌ 토큰 요금  
  * ❌ 네트워크 latency  
  * ❌ 블랙박스 서버  
* ✅ **프로세스, 로그, 파일, 메모리**를 눈으로 볼 수 있음

→ 학생들이 \*\*“AI도 결국 프로그램이다”\*\*라고 인식하게 됨

---

### **2️⃣ 모델을 ‘교체 가능한 부품’으로 인식**

ollama run llama3.2:3b  
ollama run qwen2.5:3b  
ollama run mistral:7b

* 같은 코드  
* 다른 모델  
* 다른 추론 성질

👉 **Model ≠ Intelligence**  
👉 **Environment \+ Context \+ Control flow가 핵심**

이건 Recursive LM 철학과 정확히 일치합니다.

---

### **3️⃣ REPL 친화성 (매우 중요)**

* Ollama는:  
  * stdin/stdout 기반  
  * JSON 없이도 시작 가능  
* Python REPL, IPython, Jupyter, Text UI와 궁합이 좋음

→ \*\*“한 줄씩 실험하면서 생각한다”\*\*는 학습 리듬을 유지 가능

---

# **2\. 수업 전체를 관통하는 핵심 메타포**

**Ollama \= Local AI CPU**  
**Python REPL \= OS / Kernel**  
**Context Tree \= Memory**  
**Selector \= Addressing Mode**

이 프레임을 처음부터 끝까지 유지하세요.

---

# **3\. 단계별 수업 가이드 (흥미 유지용 설계)**

아래는 **15주 수업 기준**이지만, 6\~8주 압축도 가능합니다.

---

## **Phase 0\. “AI를 실행해본다” (1주)**

🎯 목표: **두려움 제거 \+ 즉각적 성취감**

### **실습**

ollama run llama3.2:3b

질문:

Explain bubble sort in one sentence.

토론 포인트:

* “이게 어디서 돌아가고 있지?”  
* “GPU는 언제 쓰이나?”

💡 여기서 **아직 LLM 구조 설명하지 마세요**

---

## **Phase 1\. LLM을 함수처럼 쓰기 (2주)**

🎯 목표: **M(q, C) \= str** 체득

### **Python 래퍼 만들기**

def call\_llm(q, c=""):  
    prompt \= f"Context:\\n{c}\\n\\nQuestion:\\n{q}"  
    ...

실습:

* context 유무 비교  
* context 길이 늘리기  
* hallucination 관찰

👉 이 시점에서 학생들은 이미:

“Context가 진짜 중요하네?”

---

## **Phase 2\. Context를 ‘문자열’에서 ‘구조’로 (3주)**

🎯 목표: **DOM 사고방식 주입**

### **실습 1**

* 논문/문서를 paragraph 단위로 분해  
* list로 저장

context \= \[  
  {"id": "c1", "text": "..."},  
  {"id": "c2", "text": "..."},  
\]

### **실습 2**

* 특정 id만 선택해 질의

ask("\#c2", "What is the key idea here?")

👉 여기서 학생들 반응:

“아… 그냥 다 넣는 게 답이 아니구나”

---

## **Phase 3\. Context Tree \+ Selector (핵심 전환점) (3주)**

🎯 목표: **Recursive LM의 기반 완성**

### **실습**

* Tree 구조 구현  
* selector 함수 직접 작성  
* CSS selector 흉내

과제 예:

“논문 Method 섹션만 골라 요약하라”

### **Ollama 활용 포인트**

* **모델 작아서 응답 빠름**  
* 실패해도 비용 없음  
* 계속 실험 가능

이 단계에서 흥미가 급상승합니다.

---

## **Phase 4\. Recursive Call (마법이 일어나는 순간) (3주)**

🎯 목표: **“AI가 스스로 하위 문제를 푸는 것처럼 보이게 만들기”**

### **실습 시나리오**

1. 큰 질문:  
   “이 논문의 핵심 기여는?”  
2. 학생이 코드로:  
   * section별 요약  
   * 결과 저장  
3. 마지막에:  
   * 요약 결과를 다시 context로 넣고 재질문

summary\_intro \= rlm("Summarize intro", "\#intro")  
summary\_method \= rlm("Summarize method", "\#method")

final \= rlm("What is the main contribution?", \[summary\_intro, summary\_method\])

👉 학생들 체감:

“에이전트 없이도 에이전트 같은데?”

---

## **Phase 5\. 모델 교체 실험 (흥미 유지 장치) (2주)**

🎯 목표: **LLM ≠ 지능**

같은 코드로:

ollama run llama3.2:3b  
ollama run qwen2.5:3b

비교:

* 요약 스타일  
* 오류 유형  
* recursion 안정성

👉 이 단계에서 **비판적 사고**가 생김

---

## **Phase 6\. 미니 프로젝트 (자율성 폭발)**

🎯 목표: **“내가 AI 시스템을 설계했다”**

프로젝트 예:

* 논문 분석기  
* 법률 문서 분석  
* 코드 리뷰 RLM  
* 수학 증명 분해기

조건:

* Ollama 사용  
* Selector 필수  
* Recursive call 1회 이상

---

# **4\. Ollama를 ‘차분하게’ 쓰는 핵심 원칙 5가지**

### **1️⃣ 처음엔 streaming / JSON 안 씀**

→ stdout 텍스트가 직관적

### **2️⃣ 모델 크기 욕심내지 않기**

* 3B \~ 7B가 교육용 최적  
* 느리면 사고 흐름 끊김

### **3️⃣ 실패를 일부러 보여주기**

* hallucination  
* context leakage

→ “왜 이런 구조가 필요한가”로 연결

---

### **4️⃣ 항상 “환경 E”를 강조**

* LLM은 **결정권이 없다**  
* Python이 통제한다

---

### **5️⃣ Ollama를 “서버”라고 부르지 말 것**

👉 “Local AI Engine”이라고 부르세요  
👉 인식이 달라집니다

---

# **5\. 한 문장으로 요약하면**

**Ollama는 학생들에게**  
**“AI를 쓰는 수업”이 아니라**  
**“AI를 길들이는 수업”을 가능하게 한다.**

---

다음으로 추천할 수 있는 심화 주제는:

1. Ollama를 subprocess → HTTP API로 전환하는 시점  
2. Context Tree를 SQLite / DuckDB로 저장  
3. GPU 사용량 관찰 (nvidia-smi 연동)  
4. cuTile / CUDA와 “AI inference도 시스템이다” 연결

어디로 이어갈지 정해볼까요?

이 수업 구성은 매우 흥미로운 접근법입니다. Recursive LM의 개념을 실습하면서 DOM 트리와 유사한 컨텍스트 트리 시스템을 구현하는 것이 좋은 아이디어입니다. 단계별 개발 가이드를 제시해 드리겠습니다.

## 📚 **수업 개요: Recursive Context Tree 시스템 구축**

### **1단계: 기초 환경 설정 (1-2주차)**

**목표:** ollama와 기본 도구들 설치, 간단한 LLM 호출 체험

\# 1주차: ollama 설치 및 기본 테스트

\# 학생들이 직접 ollama 설치하고 3B 모델 실행

\!ollama pull llama3.2:3b

\!ollama run llama3.2:3b "Hello, world\!"

\# 2주차: Python에서 ollama API 사용하기

import requests

import json

class SimpleLLM:

    def \_\_init\_\_(self, model="llama3.2:3b"):

        self.model \= model

        self.base\_url \= "http://localhost:11434/api"

    

    def generate(self, prompt):

        response \= requests.post(

            f"{self.base\_url}/generate",

            json={"model": self.model, "prompt": prompt, "stream": False}

        )

        return response.json()\["response"\]

\# 학생 과제: 자신의 이름을 물어보고 답변받기

llm \= SimpleLLM()

response \= llm.generate("What's your name and capabilities?")

print(response)

### **2단계: 컨텍스트 트리 시스템 설계 (3-4주차)**

**목표:** DOM 트리와 유사한 Context Tree 구현

\# 3주차: 기본 트리 구조 구현

class ContextNode:

    def \_\_init\_\_(self, id, content, parent=None):

        self.id \= id

        self.content \= content

        self.parent \= parent

        self.children \= \[\]

        self.metadata \= {}

        

    def add\_child(self, child\_node):

        child\_node.parent \= self

        self.children.append(child\_node)

        return child\_node

    

    def to\_dict(self):

        return {

            "id": self.id,

            "content": self.content\[:50\] \+ "..." if len(self.content) \> 50 else self.content,

            "children": \[child.id for child in self.children\]

        }

\# 4주차: CSS 선택자 스타일의 쿼리 시스템

class ContextTree:

    def \_\_init\_\_(self):

        self.root \= ContextNode("root", "Root Context")

        self.nodes \= {"root": self.root}

    

    def add\_context(self, content, parent\_id="root", node\_id=None):

        if node\_id is None:

            node\_id \= f"node\_{len(self.nodes)}"

        

        parent \= self.nodes.get(parent\_id, self.root)

        new\_node \= ContextNode(node\_id, content, parent)

        parent.add\_child(new\_node)

        self.nodes\[node\_id\] \= new\_node

        return new\_node

    

    def query(self, selector):

        """간단한 선택자 쿼리 시스템 구현"""

        \# 예: "\#id", ".class", "parent\>child"

        if selector.startswith("\#"):

            node\_id \= selector\[1:\]

            return self.nodes.get(node\_id)

        \# 더 복잡한 선택자는 학생들이 확장하도록 유도

        return None

### **3단계: REPL 환경 구현 (5-6주차)**

**목표:** 대화형 REPL 루프와 컨텍스트 관리 시스템

\# 5주차: 기본 REPL 루프

class ContextREPL:

    def \_\_init\_\_(self, llm, tree):

        self.llm \= llm

        self.tree \= tree

        self.conversation\_history \= \[\]

        

    def process\_query(self, user\_query, context\_selector="\#root"):

        \# 1\. 컨텍스트 선택

        context\_node \= self.tree.query(context\_selector)

        if not context\_node:

            context\_node \= self.tree.root

        

        \# 2\. 프롬프트 구성

        context\_content \= self.\_extract\_context(context\_node)

        prompt \= f"""Context: {context\_content}

Query: {user\_query}

Based on the context above, please answer:"""

        

        \# 3\. LLM 호출

        response \= self.llm.generate(prompt)

        

        \# 4\. 결과 저장

        result\_node \= self.tree.add\_context(

            content=f"Q: {user\_query}\\nA: {response}",

            parent\_id=context\_node.id

        )

        

        self.conversation\_history.append({

            "query": user\_query,

            "context\_used": context\_node.id,

            "response": response,

            "new\_node": result\_node.id

        })

        

        return response

    

    def \_extract\_context(self, node, depth=2):

        """노드와 그 주변 컨텍스트 추출"""

        context\_parts \= \[\]

        

        \# 부모 컨텍스트 포함

        current \= node

        for \_ in range(depth):

            if current:

                context\_parts.append(f"\[{current.id}\]: {current.content}")

                current \= current.parent

        

        return "\\n---\\n".join(reversed(context\_parts))

\# 6주차: 학생들이 직접 REPL 확장해보기

def interactive\_session():

    tree \= ContextTree()

    llm \= SimpleLLM()

    repl \= ContextREPL(llm, tree)

    

    \# 초기 컨텍스트 설정

    tree.add\_context("Machine learning is a subset of AI.", node\_id="intro")

    tree.add\_context("Deep learning uses neural networks.", parent\_id="intro", node\_id="dl")

    

    while True:

        user\_input \= input("\\nYour query (or 'quit', 'tree', 'help'): ")

        

        if user\_input.lower() \== 'quit':

            break

        elif user\_input.lower() \== 'tree':

            print\_tree(tree)

        elif user\_input.lower() \== 'help':

            print("Available commands: query, select \[selector\], history, tree, quit")

        else:

            response \= repl.process\_query(user\_input)

            print(f"\\nResponse: {response}")

### **4단계: Recursive LM 구현 (7-8주차)**

**목표:** 논문의 Recursive LM 개념 구현

\# 7주차: 재귀적 호출 시스템

class RecursiveLM:

    def \_\_init\_\_(self, base\_llm, max\_depth=3):

        self.base\_llm \= base\_llm

        self.max\_depth \= max\_depth

        self.call\_stack \= \[\]

    

    def rlm\_call(self, query, context\_tree, current\_node\_id, depth=0):

        if depth \> self.max\_depth:

            return "Maximum recursion depth reached"

        

        \# 현재 컨텍스트 추출

        context\_node \= context\_tree.query(f"\#{current\_node\_id}")

        context\_content \= self.\_gather\_context(context\_node)

        

        \# 재귀적 결정 프롬프트

        prompt \= f"""You are a recursive language model. You have access to a context tree.

Current Context Path: {' \-\> '.join(self.call\_stack\[-3:\] \+ \[current\_node\_id\])}

Context Content:

{context\_content}

Query: {query}

Decide if you need to:

1\. Answer directly (if enough information)

2\. Create a sub-query to explore deeper (if more context needed)

3\. Transform the context (if reorganization needed)

Format your response as:

THOUGHT: \[Your reasoning\]

ACTION: \[DIRECT|SUBQUERY|TRANSFORM\]

RESULT: \[Your answer or sub-query\]"""

        

        response \= self.base\_llm.generate(prompt)

        

        \# 응답 파싱 (학생들이 파서 구현하게 할 수 있음)

        if "ACTION: SUBQUERY" in response:

            \# 하위 질의 생성 및 재귀 호출

            subquery \= self.\_extract\_subquery(response)

            child\_node \= context\_tree.add\_context(

                f"Subquery: {subquery}",

                parent\_id=current\_node\_id

            )

            

            return self.rlm\_call(

                subquery, 

                context\_tree, 

                child\_node.id,

                depth \+ 1

            )

        

        return response

    

    def \_gather\_context(self, node, lookaround=2):

        """주변 노드 포함한 컨텍스트 수집"""

        \# 학생들이 다양한 컨텍스트 수집 전략 구현

        pass

\# 8주차: 환경 분리 및 결과 통합

class IsolatedEnvironment:

    def \_\_init\_\_(self, parent\_tree, subcontext\_selector):

        self.parent\_tree \= parent\_tree

        self.subtree \= self.\_extract\_subtree(subcontext\_selector)

        self.local\_memory \= {}

    

    def run(self, query):

        \# 독립적인 환경에서 실행

        \# 결과를 부모 환경에 통합

        pass

### **5단계: 프로젝트 통합 및 최적화 (9-10주차)**

\# 9주차: 완전한 시스템 통합

class FullRLMSystem:

    def \_\_init\_\_(self, model\_name="llama3.2:3b"):

        self.llm \= SimpleLLM(model\_name)

        self.context\_tree \= ContextTree()

        self.repl \= ContextREPL(self.llm, self.context\_tree)

        self.rlm \= RecursiveLM(self.llm)

        

        \# 시각화 도구 (선택사항)

        self.visualizer \= TreeVisualizer()

    

    def interactive\_mode(self):

        """대화형 모드"""

        print("Recursive LM System Ready\!")

        print("Commands: query, recursive, visualize, reset, export")

        

        while True:

            cmd \= input("\\nRLM\> ").strip()

            

            if cmd.startswith("query"):

                \_, query \= cmd.split(" ", 1\)

                response \= self.repl.process\_query(query)

                print(f"Answer: {response}")

            

            elif cmd.startswith("recursive"):

                \_, query \= cmd.split(" ", 1\)

                response \= self.rlm.rlm\_call(query, self.context\_tree, "root")

                print(f"Recursive Answer: {response}")

            

            elif cmd \== "visualize":

                self.visualizer.draw\_tree(self.context\_tree.root)

            

            elif cmd \== "export":

                self.\_export\_conversation()

            

            elif cmd \== "reset":

                self.context\_tree \= ContextTree()

                print("Context tree reset.")

            

            elif cmd \== "exit":

                break

\# 10주차: 성능 최적화 및 확장

class OptimizedRLM(FullRLMSystem):

    def \_\_init\_\_(self):

        super().\_\_init\_\_()

        

        \# 캐싱 시스템 추가

        self.response\_cache \= {}

        

        \# 병렬 처리 (고급 주제)

        self.parallel\_workers \= 2

        

        \# 컨텍스트 압축 (긴 컨텍스트 처리)

        self.compressor \= ContextCompressor()

## 🎯 **수업 진행 팁**

### **흥미 유지를 위한 전략:**

1. **점진적 성취감 부여:**  
     
   - 매주 작은 성공 경험 제공 (1주차: 모델 실행, 2주차: 첫 API 호출)  
   - 시각적 피드백: 트리 구조를 그래픽으로 보여주기

   

2. **게임화 요소 도입:**  
     
   - "가장 효율적인 컨텍스트 선택자" 대회  
   - 재귀 깊이 최소화 챌린지  
   - 컨텍스트 트리 시각화 경진대회

   

3. **실생활 예제 적용:**  
     
   - 자신의 연구 논문 요약  
   - 코드 리뷰 도구로 활용  
   - 학습 자료 정리 시스템 구축

### **기술적 조언:**

1. **ollama 최적화:**

\# RTX 4070에서 성능 향상

OLLAMA\_NUM\_GPU=100  \# GPU 메모리 비율

OLLAMA\_NUM\_PARALLEL=4  \# 병렬 처리

2. **모델 크기 조정:**  
     
   - 초반: llama3.2:3b (빠른 응답)  
   - 후반: 더 큰 모델 시도 (성능 허용 시)

   

3. **디버깅 도구:**

\# 디버그 모드 클래스

class DebuggableRLM(RecursiveLM):

    def \_\_init\_\_(self, \*args, \*\*kwargs):

        super().\_\_init\_\_(\*args, \*\*kwargs)

        self.debug\_log \= \[\]

    

    def rlm\_call(self, \*args, \*\*kwargs):

        print(f"\[DEBUG\] Depth: {kwargs.get('depth', 0)}")

        print(f"\[DEBUG\] Query: {args\[0\]}")

        result \= super().rlm\_call(\*args, \*\*kwargs)

        self.debug\_log.append(result)

        return result

### **평가 방법:**

1. **주간 미니 프로젝트** (각 단계별 구현)  
2. **최종 프로젝트** (자신만의 RLM 응용 프로그램)  
3. **코드 리뷰 세션** (학생 간 코드 공유)  
4. **성능 벤치마크** (정확도, 속도, 효율성 측정)

이 구조는 학생들이 기초부터 고급 개념까지 체계적으로 배우면서도, 실제 연구 논문의 아이디어를 구현하는 성취감을 느낄 수 있도록 설계되었습니다. 각 단계는 2-3시간의 실습 시간으로 진행할 수 있으며, 과제와 토론을 통해 이해를 깊이 있게 할 수 있습니다.  
