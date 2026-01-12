# **Interactive Recursive Build Agent (IRBA) 🐚**

**"Context-Aware Recursive Shell for Local Web Building"**

IRBA는 로컬 LLM(Ollama)과 Python REPL을 결합하여, **대화형 쉘(Shell)** 환경에서 복잡한 소프트웨어 구축 작업을 수행하는 에이전트 시스템입니다.

**In-Context Learning**과 **Recursive Task Decomposition(재귀적 작업 분할)** 철학을 바탕으로, 사용자의 목표(Goal)를 하위 작업으로 쪼개고(Divide), 각 작업을 수행하는 하위 에이전트(Sub-Agent)를 생성(Spawn)하여 최종 결과물을 만들어냅니다(Conquer).

주요 응용 분야는 \*\*"발표 자료(Markdown)를 반응형 웹 페이지(HTML/JS)로 변환하는 빌드 시스템"\*\*입니다.

## **✨ Key Features**

### **1\. 🧠 Local Intelligence (Ollama Driven)**

* 클라우드 API 의존 없이 \*\*로컬 GPU(RTX 5070 등)\*\*를 활용합니다.  
* llama3.2:3b, qwen2.5:3b 등 경량화된 고성능 모델을 스위칭하며 사용 가능합니다.

### **2\. 🐚 Interactive REPL Shell**

* 리눅스 쉘과 유사한 대화형 인터페이스를 제공합니다.  
* /add, /ls, /search 등의 명령어로 \*\*Context(작업 기억)\*\*를 동적으로 관리합니다.  
* LLM이 작성한 코드를 즉시 실행하고 결과를 피드백 받습니다.

### **3\. 🌲 Recursive "Divide & Conquer"**

* 에이전트가 문제를 해결하기 어렵다고 판단하면, 스스로 \*\*하위 에이전트(Sub-Agent)\*\*를 호출합니다.  
* 예: "웹사이트를 만들어" \-\> \[Agent A: HTML 구조 설계\] \+ \[Agent B: CSS 스타일링\] \+ \[Agent C: JS 로직\]

### **4\. 🔍 Local Document Search (RAG Lite)**

* 프로젝트 내의 **Markdown** 및 **HTML** 문서를 의미 기반(Semantic) 또는 키워드 기반으로 검색합니다.  
* 방대한 문서에서 필요한 부분만 Context에 로드하여 LLM의 Window 한계를 극복합니다.

## **🛠 Architecture**

시스템은 **CARS (Context-Aware Recursive Shell)** 아키텍처를 따릅니다.

graph TD  
    User\[User Command\] \--\> Shell\[IRBA Shell (REPL)\]  
    Shell \--\> Context\[Context Manager (Memory)\]  
    Shell \--\> Tools\[Tool Box\]  
      
    subgraph "Agent Runtime"  
        Brain\[Local LLM (Ollama)\]  
        Exec\[Python Executor\]  
    end  
      
    Shell \<--\> Brain  
    Brain \--\>|Generate Code| Exec  
    Exec \--\>|Result| Brain  
      
    Brain \--\>|Delegate Task| SubAgent\[Child Agent\]  
    SubAgent \--\>|Return Result| Brain  
      
    Tools \--\>|Search| Docs\[MD/HTML Files\]  
    Tools \--\>|Write| FileSystem\[Project Root\]


## 🚀 시작하기

### 1. 요구 사항 설치
```bash
pip install -r requirements.txt
```

### 2. Ollama 모델 준비
본 프로젝트는 기본적으로 `gemma3:4b` 모델을 사용합니다.
```bash
ollama pull gemma3:4b
```

### 3. 에이전트 실행
```bash
python agent.py --file your_document.md
```

### **1\. Prerequisites**

* **Python 3.10+**  
* **Ollama** 설치 및 서비스 실행 중일 것  
* **NVIDIA GPU** (권장, CUDA 설정 완료 시)

### **2\. Installation**

\# 1\. Clone the repository  
git clone \[https://github.com/your-username/interactive-recursive-build-agent.git\](https://github.com/your-username/interactive-recursive-build-agent.git)  
cd interactive-recursive-build-agent

\# 2\. Create Virtual Environment  
python \-m venv venv  
source venv/bin/activate  \# Windows: venv\\Scripts\\activate

\# 3\. Install Dependencies  
pip install \-r requirements.txt

### **3\. Model Setup (Ollama)**

이 프로젝트는 아래 모델들에 최적화되어 있습니다. 터미널에서 미리 다운로드해주세요.

\# General Instruction & Reasoning  
ollama pull llama3.2:3b

\# Coding Specialist  
ollama pull qwen2.5:3b

## **💻 Usage**

### **Start the Shell**

프로젝트 루트에서 에이전트를 실행합니다.

python main.py

### **Shell Commands**

| Command | Description |
| :---- | :---- |
| /model \<name\> | 사용할 Ollama 모델 변경 (예: /model qwen2.5:3b) |
| /add \<path\> | 파일/폴더를 Context(기억)에 추가 (Glob 지원) |
| /search \<query\> | 로컬 MD/HTML 문서 검색 후 Context에 추가 |
| /ls | 현재 Context에 로드된 파일 목록 확인 |
| /clear | Context 초기화 |
| /run \<goal\> | **\[메인 기능\]** 목표를 설정하고 Recursive Build 시작 |
| /exit | 종료 |

## **🏗 Scenario: Presentation Web Builder**

**목표:** docs/presentation.md 파일을 읽어서, 슬라이드쇼가 가능한 index.html 웹 페이지 만들기.

**Step 1: 쉘 실행 및 Context 로드**

(irba) root@build:\~$ /add docs/presentation.md  
\[System\] Added 'docs/presentation.md' to context.

**Step 2: 참조할 디자인/템플릿 검색 (Optional)**

(irba) root@build:\~$ /search "slide template html"  
\[Search\] Found 'templates/simple\_slide.html'. Added to context.

**Step 3: 빌드 명령 실행 (Recursive Process)**

(irba) root@build:\~$ /run "presentation.md 내용을 바탕으로 Reveal.js 스타일의 웹 프레젠테이션 index.html을 만들어줘."

🤖 \[Root Agent\]: 목표 분석 중...   
   \-\> 작업이 복잡하여 하위 에이전트에게 위임합니다.  
     
   🐣 \[Sub-Agent 1 (Parser)\]: Markdown 파싱 및 섹션 분리 담당  
      ... (Python 코드 실행: md 파일 읽기 및 JSON 구조화) ...  
      ✅ 완료.

   🐣 \[Sub-Agent 2 (Coder)\]: HTML/CSS 생성 담당  
      ... (Python 코드 실행: 구조화된 데이터를 HTML 템플릿에 주입) ...  
      ✅ 완료. 'index.html' 생성됨.

🤖 \[Root Agent\]: 모든 하위 작업 완료. 결과물을 검증합니다.  
✅ 최종 작업 완료. 브라우저에서 index.html을 확인하세요.

## **🧩 Project Structure**

.  
├── main.py              \# Entry point (Shell Loop)  
├── core/  
│   ├── agent.py         \# Recursive Agent Class  
│   ├── llm.py           \# Ollama Interface  
│   ├── context.py       \# File & Memory Manager  
│   └── executor.py      \# Python Code Sandbox  
├── tools/  
│   ├── search.py        \# Semantic Search (ChromaDB/BM25)  
│   └── file\_ops.py      \# File System Operations  
└── requirements.txt     \# Dependencies

## **🛣 Roadmap**

* \[ \] **State Persistence:** 에이전트의 작업 상태를 .irba 파일로 저장/복구 기능.  
* \[ \] **Web Search Tool:** 로컬 문서뿐만 아니라 웹 검색(DuckDuckGo) 기능 연동.  
* \[ \] **Sandbox Security:** Docker 기반의 코드 실행 환경 격리.