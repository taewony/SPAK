# **Interactive Recursive LLM Shell (IRLS) 🐚**

**"A Context-Revision–Driven Shell for Programming Local Language Models"**

IRLS는 로컬 LLM(Ollama)과 Python REPL을 결합한 **대화형 LLM 쉘 환경**이다. 이 시스템은 LLM을 자율적인 에이전트로 취급하지 않고, **프로그래밍 가능한 계산 장치(execution engine)**로 다룬다. 사용자는 REPL에서 Context를 선택·분할·변환하며, Recursive 실행을 통해 복잡한 build 작업을 단계적으로 수행한다.

IRLS는 In-Context Learning(ICL)과 Recursive Language Model(RLM)의 철학을 바탕으로, Context를 파일 시스템 기반의 외부 메모리로 관리하고, multi-turn 실행을 통해 Context를 점진적으로 재구성한다. 이 과정에서 LLM의 추론 상태는 파일로 저장된 Context Revision으로 외부화되며, IRLS는 이를 다시 불러와 후속 작업을 이어간다.

---

## 🧠 Design Philosophy

IRLS는 전통적인 Agent Framework가 아니다. 대신 다음 원칙을 따른다.

1. **LLM은 상태를 가지지 않는 실행 엔진이다**
2. **Context가 실제 상태(state)다**
3. **In-Context Learning은 일시적인 weight delta와 유사하다**
4. **Recursive 실행은 monolithic prompt를 대체한다**
5. **파일 시스템은 LLM의 외부화된 latent memory다**
6. **Human은 scheduler이자 supervisor다**

이 관점에서 IRLS는 **Latent State Emulator for Language Models**로 볼 수 있다.

---

## ✨ Key Features

### 1. 🧠 Local LLM Execution (Ollama)

* 클라우드 API 의존 없이 로컬 GPU(RTX 4070 / L40S 등)에서 실행
* llama3.2:3b, qwen2.5:3b 등 경량 모델을 즉시 교체 가능
* 빠른 피드백 루프를 통해 REPL 기반 실험에 최적화

### 2. 🐚 Interactive REPL Shell

* Linux shell과 유사한 명령 기반 인터페이스
* Context를 파일·섹션·노드 단위로 선택하고 조작
* 각 실행 결과는 Context Revision으로 저장 가능

### 3. 🌲 Recursive Context Decomposition

* 문제를 한 번에 해결하지 않고 Context를 분할
* 각 하위 Context는 **독립된 Recursive 실행 단위**로 처리
* 결과는 상위 Context에 merge되거나 새로운 Revision으로 저장

### 4. 💾 Revisioned Context Memory

* 모든 Context 변경은 파일 시스템에 revision으로 기록
* 이전 상태로 rollback하거나 branch 가능
* multi-turn build trace를 명시적으로 관리

---

## 🛠 Architecture

IRLS는 다음 구성 요소로 이루어진다.

* **REPL Environment**: 사용자 입력과 실행 흐름을 제어
* **Context Tree**: 현재 작업에 사용되는 구조화된 Context
* **Revision Store (FS)**: Context의 영속 저장소
* **Recursive Execution Kernel**: LLM 호출 및 하위 실행 생성

파일 시스템은 단순한 출력 대상이 아니라, **LLM의 외부 상태 저장소**로 사용된다.

---

## 🚀 Getting Started

### 1. Prerequisites

* Python 3.10+
* Ollama 설치 및 실행 중
* NVIDIA GPU (권장)

### 2. Installation

```bash
# Clone repository
git clone https://github.com/your-username/irls.git
cd irls

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Model Setup (Ollama)

```bash
ollama pull llama3.2:3b
ollama pull qwen2.5:3b
```

---

## 💻 Usage

### Start the Shell

```bash
python build-agent.py
```

### REPL Command Specification

| Command | Description |
|------|-------------|
| `use <path...>` | 파일/디렉토리를 Context로 로드 |
| `tree` | 현재 Context Tree 구조 출력 |
| `select <selector>` | Context Tree에서 노드 선택 |
| `view` | 선택된 Context 내용 확인 |
| `ask "<prompt>"` | 선택된 Context를 입력으로 LLM 실행 |
| `recurse <selector> "<prompt>"` | 하위 Context로 Recursive 실행 |
| `save <revision>` | 현재 Context를 revision으로 저장 |
| `revisions` | 저장된 revision 목록 출력 |
| `checkout <revision>` | 이전 revision으로 이동 |
| `model <name>` | 사용할 Ollama 모델 변경 |
| `clear` | 현재 Context 초기화 |
| `exit` | 쉘 종료 |

---

## 🏗 Example Session

```text
$ python build-agent.py
(irls)> use docs/paper.md
(irls)> select section.method
(irls)> ask "Summarize the algorithm"
(irls)> save v0.1

(irls)> recurse section.method "Rewrite as pseudocode"
(irls)> save v0.2

(irls)> ask "What assumptions are implicit?"
(irls)> save v0.3
```

각 단계의 결과는 Context Revision으로 저장되며, 이후 build 작업의 입력으로 재사용된다.

---

## 🧩 Project Structure

```
.
├── build-agent.py      # Entry point (REPL Shell)
├── core/
│   ├── kernel.py       # Recursive execution kernel
│   ├── llm.py          # Ollama interface
│   ├── context.py      # Context tree & selector
│   ├── revision.py     # Context revision manager
│   └── repl.py         # REPL command loop
├── contexts/           # Revisioned context storage
└── requirements.txt
```

---

## 🛣 Roadmap

* [ ] Context selector DSL 확장 (CSS/XPath-like)
* [ ] Context diff & merge visualization
* [ ] Multi-branch build workflow
* [ ] GPU usage & cost introspection
* [ ] Education-focused tutorial & assignments

