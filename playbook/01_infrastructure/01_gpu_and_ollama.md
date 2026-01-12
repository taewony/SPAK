---
title: "GPU Inference Mechanics & Ollama Setup"
status: completed
tags: ["infrastructure", "gpu", "ollama"]
difficulty: beginner
hardware_req: "NVIDIA GPU (min 4GB VRAM)"
tools: ["nvidia-smi", "ollama"]
---

# 🖥️ GPU Inference & Ollama Setup

## 1. Goal
LLM 추론(Inference) 과정에서 하드웨어 자원이 어떻게 사용되는지 이해하고, 로컬 환경(Windows)에서 Ollama를 최적화하여 실행한다.

## 2. Key Concepts Learned

### Prefill (Processing)
* **정의**: 사용자의 입력(Prompt)을 토큰화하고 KV Cache를 생성하는 단계.
* **특징**: 병렬 처리가 가능하여 GPU 연산 능력(Compute)에 크게 의존함. 짧은 시간에 급격한 GPU 부하 발생.

### Decode (Generating)
* **정의**: 한 번에 하나의 토큰을 생성하는 단계.
* **특징**: 이전 상태(KV Cache)를 메모리에서 불러와야 하므로 **메모리 대역폭(Memory Bandwidth)**이 병목이 됨.

## 3. Observation Log
* **Command**: `nvidia-smi -l 1` (1초마다 갱신)
* **Observation**:
    * Ollama 모델 로드 시 VRAM 사용량 급증.
    * 긴 텍스트 요약 시작 시(Prefill) GPU Compute Usage가 순간적으로 튀어오름.
    * 답변 생성 중(Decode)에는 VRAM 사용량은 유지되나 Compute Usage는 낮게 유지됨.

## 4. Action Items
- [x] Install Ollama on Windows
- [x] Pull `gemma:2b` or `llama3` model
- [x] Run `nvidia-smi` monitor
- [x] Verify VRAM usage limits (Safe buffer 설정)
