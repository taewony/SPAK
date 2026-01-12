---
title: "Map-Reduce Document Summarizer"
status: completed
tags: ["pattern", "map-reduce", "summarization"]
difficulty: intermediate
code_ref: "agent_repl.py"
model: "gemma3:4b"
---

# 📝 Map-Reduce Document Summarizer

## 1. Architecture Overview
단일 Context Window(예: 8k, 128k)에 담을 수 없는 대량의 문서를 처리하기 위한 분산 처리 패턴.

### Phase 1: Map (Worker Agent)
* **Role**: `System Prompt: "너는 유능한 분석가야."`
* **Input**: 개별 Markdown 파일 (Chunk).
* **Process**: 파일을 읽고 핵심 내용 3줄 요약.
* **Output**: `intermediate_summaries` 리스트에 저장.

### Phase 2: Reduce (Manager Agent)
* **Role**: `System Prompt: "너는 프로젝트 매니저야."`
* **Input**: Phase 1에서 생성된 요약본들의 집합.
* **Process**: 전체 맥락을 통합하여 하나의 보고서 작성.
* **Output**: 최종 `answer.md`.

## 2. Code Logic (`agent_repl.py`)

```python
# Map Step
for file in md_files:
    summary = summarize_single_file(file) # Analyst Persona
    intermediate_summaries.append(summary)

# Reduce Step
final_prompt = "다음은 요약본이다. 종합 보고서를 작성해:\n" + "".join(intermediate_summaries)
result = call_llm(final_prompt, system_role="너는 프로젝트 매니저야.") # Manager Persona
```

## 3. Improvements & Next Steps
* **Context Overflow 방지**: Reduce 단계에서도 입력이 너무 길어질 경우, Reduce를 계층적(Hierarchical)으로 수행해야 함.
* **Parallel Execution**: 현재는 `for` 루프로 순차 실행하지만, `asyncio`를 사용하여 병렬 호출 가능 (GPU 메모리가 허용하는 한).

```