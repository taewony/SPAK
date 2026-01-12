# 🚀 Agent Engineering 분석: agent_repl.py

이 프로젝트의 `agent_repl.py`는 단순한 루프가 아니라, **Map-Reduce 패턴**과 **Role-Playing Prompting**을 결합한 초기 단계의 AI 에이전트 구조를 보여줍니다.

## 1. System Prompt & Role Definition
이 코드는 작업의 성격에 따라 LLM의 페르소나(Persona)를 동적으로 교체합니다.

*   **Worker Agent (Analyst):**
    *   **System Prompt:** `"너는 유능한 분석가야."` (기본값)
    *   **역할:** 개별 데이터(파일)를 읽고, 사실 위주의 정보 압축(Summarization)을 수행합니다. 주관을 배제하고 핵심 정보만 추출하는 것이 목표입니다.
*   **Manager Agent (Project Manager):**
    *   **System Prompt:** `"너는 프로젝트 매니저야."`
    *   **역할:** Worker들이 가져온 파편화된 정보를 통합(Synthesis)하여, 전체적인 맥락을 파악하고 최종 '보고서' 형태로 가공합니다.

## 2. LLM 호출 및 결과 도출 프로세스 (Map-Reduce)
데이터 처리 파이프라인은 전형적인 **Map-Reduce** 방식을 따릅니다.

1.  **Map Phase (분산 처리):**
    *   `summarize_single_file` 함수가 각 파일을 순회합니다.
    *   Context Window 제한(예: 3000자)을 고려하여 데이터를 청킹(Chunking)합니다.
    *   Analyst 에이전트(Worker)에게 개별 요약을 요청하여 `intermediate_summaries` 리스트에 축적합니다.
2.  **Reduce Phase (통합 및 결론):**
    *   `process_summary_docs` 함수에서 축적된 요약본들을 하나의 컨텍스트로 병합(Join)합니다.
    *   Manager 에이전트에게 통합된 컨텍스트를 전달하며 "종합 보고서" 작성을 요청합니다.
    *   최종적으로 사용자는 개별 파일의 단순 나열이 아닌, 구조화된 인사이트(`answer`)를 얻게 됩니다.

이 구조는 **Recursive Agent**로 발전하기 위한 기초 단계로, 단일 LLM 호출로는 불가능한 대규모 문서 처리를 가능하게 만드는 핵심 엔지니어링 기법입니다.


## 3. 사용 가이드
이 코드를 실행하면 (agent) >>> 프롬프트가 뜹니다.

> summary docs 를 입력합니다.

폴더 내의 파일들을 읽고 GPU가 일을 시작합니다 (nvidia-smi로 확인 가능).

작업이 끝나면 화면에 결과가 출력되고, 내부적으로 answer 변수에 텍스트가 담깁니다.

> save answer 를 입력합니다.

answer 변수에 있던 내용이 실제 파일(answer.md)로 저장됩니다.

이제 이 구조는 단순 스크립트가 아니라, 상태(State)를 가진 초보적인 에이전트 쉘이 되었습니다. 필요하다면 print(answer) 같은 명령어를 추가해 변수 내용을 언제든 다시 확인할 수도 있습니다.

### 3. Recursive LM (Map-Reduce) 아키텍처로 확장하기

단일 파일 처리는 성공했으니, 이제 **"여러 개의 문서를 읽어 전체를 관통하는 인사이트를 뽑아내는"** Recursive(재귀적) 구조로 코드를 확장해 봅시다.

이 방식은 사용자의 관심사인 **"Recursive Markdown Agent"** 연구의 기초가 됩니다. 3GB VRAM 제약을 고려하여 **순차 처리(Sequential Processing)** 방식으로 설계해야 합니다.

#### 제안하는 아키텍처 (Map-Reduce 패턴)

1. **Map 단계 (개별 요약):** 폴더 내의 모든 `.md` 파일을 하나씩 로드 -> 요약 -> 메모리 해제 -> 다음 파일. (Context를 계속 비워줘야 함)
2. **Reduce 단계 (종합):** 위에서 나온 "요약본들"을 모아서 -> 최종 리포트 작성.

#### 확장된 Python 코드 (`recursive_agent.py`)

```python
import ollama
import os
import glob
import time

MODEL_NAME = "gemma3:4b"
TARGET_FOLDER = "./docs"  # 마크다운 파일들이 있는 폴더

def call_llm(prompt, system_role="너는 유능한 분석가야."):
    """Ollama API를 호출하는 래퍼 함수"""
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {'role': 'system', 'content': system_role},
            {'role': 'user', 'content': prompt}
        ],
        stream=False # 긴 작업이므로 스트림 끄고 완료 후 반환 (선택 사항)
    )
    return response['message']['content']

def summarize_single_file(file_path):
    """[Map] 개별 파일 요약"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 3GB VRAM 보호를 위해 입력 길이 제한 (예: 3000자)
    if len(content) > 3000:
        content = content[:3000] + "\n...(내용 잘림)..."
        
    prompt = f"다음 문서를 읽고 핵심 내용을 3줄로 요약해줘:\n\n{content}"
    summary = call_llm(prompt)
    print(f"  -> {os.path.basename(file_path)} 요약 완료.")
    return summary

def recursive_summarization():
    # 1. 대상 파일 찾기
    md_files = glob.glob(os.path.join(TARGET_FOLDER, "*.md"))
    if not md_files:
        print("처리할 Markdown 파일이 없습니다.")
        return

    print(f"총 {len(md_files)}개의 파일을 순차적으로 분석합니다...")
    
    # 2. Map 단계: 각 파일 순차 요약 (중간 State 생성)
    intermediate_summaries = []
    
    for i, file_path in enumerate(md_files):
        print(f"[{i+1}/{len(md_files)}] 처리 중: {file_path}")
        try:
            summary = summarize_single_file(file_path)
            intermediate_summaries.append(f"파일명: {os.path.basename(file_path)}\n요약: {summary}")
            
            # 팁: 1060 GPU 열 식힐 겸, VRAM 정리 겸 짧은 대기 (선택)
            time.sleep(1) 
            
        except Exception as e:
            print(f"  -> 오류 발생: {e}")

    # 3. Reduce 단계: 요약본들을 모아서 최종 결론 도출
    print("\n--- 전체 문맥 통합(Reduce) 시작 ---")
    
    combined_context = "\n\n".join(intermediate_summaries)
    
    final_prompt = (
        f"다음은 여러 마크다운 문서들의 요약본 모음이다.\n"
        f"이 프로젝트 전체의 핵심 주제와 흐름을 파악해서 '종합 보고서'를 작성해줘.\n\n"
        f"{combined_context}"
    )
    
    final_report = call_llm(final_prompt, system_role="너는 프로젝트 매니저야.")
    
    print("\n>>> [최종 종합 보고서] <<<")
    print(final_report)
    
    # 결과 저장
    with open("final_report.md", "w", encoding="utf-8") as f:
        f.write(final_report)

if __name__ == "__main__":
    recursive_summarization()

```

## 4. 이 코드의 AI Engineering 포인트

1. **State 관리:** `intermediate_summaries` 리스트가 바로 현재의 **Context State**입니다. 파일을 읽을 때마다 이 상태가 업데이트됩니다.
2. **VRAM 방어:** 파일을 통째로 합치지 않고, **요약본(압축된 정보)**만 합쳐서 최종 단계로 넘깁니다. 이는 1060 3GB 환경에서 긴 맥락을 처리할 수 있는 유일한 방법입니다.
3. **확장성:** 만약 파일이 100개라 요약본을 합쳐도 너무 길다면? `Reduce` 단계를 한 번 더 거치면 됩니다. (Map -> Reduce 1차 -> Reduce 2차 -> Final)

이 구조를 바탕으로 `./docs` 폴더에 마크다운 파일 2~3개를 넣고 실험해보시면, 1060 카드에서도 훌륭하게 **Recursive Agent**가 동작하는 것을 확인하실 수 있을 겁니다.