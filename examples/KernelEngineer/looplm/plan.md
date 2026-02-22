# LoopLM Phase 3 & 4 Development Plan (v2)

Phase 2에서 **커널 무결성(0-step bug 해결)**과 **지능형 역동성(Adaptive Halting)**을 확인했습니다. 이제 모델이 단순 암기를 넘어 실제 "연산 알고리즘"을 깨우치게 하고, Blackwell 하드웨어의 성능을 극한으로 끌어올리는 단계로 진입합니다. 특히 RTX 5070 환경에서의 실험 효율을 위해 **통합 실험 및 Trace 수집 체계**를 구축합니다.

---

## 🚀 Phase 3: Algorithmic Generalization (지능의 고도화)

**목표**: 덧셈 데이터셋에 대한 과적합을 방지하고, 보지 못한 긴 자릿수 문제도 해결하는 "추론의 일반화" 달성.

### **1. 데이터 스케일업 및 다양화 (`addition_prepare.py`)**
*   **샘플 확장**: 5만 개 → **20만 개 이상** (숫자 쌍 암기 차단).
*   **자릿수 분포**: 2~4자리 학습, **5~12자리 OOD(Out-of-Distribution) 테스트셋** 별도 구축.
*   **데이터 증강**: 피연산자 순서 변경 (`a+b` ↔ `b+a`), 등호 위치 변형 등.

### **2. 모델 용량 및 정규화 최적화 (Ablation A1-A5)**
*   현재 모델(`n_embd=384`)의 과적합을 해결하기 위해 용량 최적화 실험 수행.
    *   **A1/A2**: `n_embd` 축소 (256, 128) 및 `n_head` 조정.
    *   **A3/A4**: Dropout 강화 (0.3~0.4) 및 Label Smoothing (0.1) 도입.
    *   **A5**: Weight Decay 및 Learning Rate Scheduler 정밀 튜닝.

### **3. Wait-to-Think 전략 (`model_loop.py`)**
*   **입력 인코딩 단계**: `=` 이전 토큰은 낮은 `halt_threshold`로 빠르게 처리.
*   **출력 생성 단계**: `=` 이후부터는 높은 `halt_threshold` 또는 고정 루프(24회+)로 "사고 시간" 집중 부여.

---

## ⚡ Phase 4: Blackwell Persistent Optimization (성능의 극대화)

**목표**: 공유 가중치(Tied Weights)의 특성을 활용하여 RTX 5070에서 압도적인 추론 속도 달성 및 아키텍처 확정.

### **1. Persistent Weight Caching Kernel (`looplm_kernels.py`)**
*   Attention + MLP 전체를 **단일 CUDA 커널**로 통합.
*   공유 가중치를 GPU **L2 캐시/SRAM에 Pinning**하여 메모리 대역폭 병목 제거.
*   예상 효과: 추론 속도 2~3배 향상.

### **2. X0 Injection Ablation Study**
*   `h = Block(h + x0)` vs `h = Block(h)` 비교 실험.
*   "입력의 닻(Anchor)"이 추론 안정성과 OOD 일반화에 미치는 영향 분석 후 최종 수식 확정.

### **3. Learned Halt Gate**
*   Threshold 기반 정지를 넘어, 소형 신경망(`HaltHead`)이 스스로 "사고 종료 시점"을 결정하도록 학습.

---

## 🛠 Systematic Engineering Infrastructure

**목표**: 실험 자동화 및 지식 자산 축적 (RTX 5070 이동 횟수 최소화).

### **1. 통합 실험 스크립트 (`run_experiments.py`)**
*   학습(train) -> 평가(eval) -> OOD 테스트 -> Trace 데이터 추출을 일괄 수행.
*   모든 실험 결과는 `experiments/` 폴더에 `trace.json` 및 `metrics.csv`로 자동 저장.

### **2. Root Cause Analysis (`root.md`)**
*   실행 중 발생하는 모든 오류(OOM, Numerical Instability, Indexing Bug 등)는 즉시 RCA를 수행하여 `root.md`에 기록.
*   분석 결과는 다음 실험 설계에 즉각 반영하여 시행착오 비용 절감.

---

## 📅 향후 일정 요약
*   **Next Step**: `run_experiments.py` 작성 및 Phase 3-A1/A2 실험 착수.
*   **Success Metric**: 12자리 덧셈 Zero-shot 정답률 30% 이상, 추론 속도 2x 향상.

---
**Plan 업데이트 완료.** 이제 `root.md`는 우리의 나침반이 되고, `run_experiments.py`는 우리의 발이 될 것입니다. 🚀

✦ Phase 3를 위해 plan.md를 업데이트하고, RTX 5070에서의 실험 효율을 극대화하기 위한 통합 실험 체계(Systematic Trace
  Collection)를 구축했습니다.


  핵심 변경 사항
   1. 데이터 스케일업 (`addition_prepare.py`):
       * 학습 데이터를 5만 → 20만 개로 확장하여 암기를 차단했습니다.
       * OOD(Out-of-Distribution) 테스트를 위해 5~12자리 덧셈 데이터셋을 별도로 생성했습니다.
   2. Wait-to-Think 로직 구현:
       * `model_loop.py` & `looplm_kernels.py`: = 토큰 감지 시 thinking_threshold를 높여 더 깊게 사고하도록 하는 기능을
         커널 수준에서 구현했습니다. 이제 각 토큰별로 다른 정지 임계값을 가질 수 있습니다.
   3. 통합 실험 스크립트 (`run_experiments.py`):
       * Baseline 및 A1~A3(용량 조정), T1(사고 깊이 확장) 실험을 한 번에 수행합니다.
       * 모든 실험 결과는 experiments/summary.json에 집합적으로 기록되어 분석에 활용됩니다.
   4. OOD 평가 스크립트 (`eval_loop.py`):
       * 학습되지 않은 긴 자릿수 문제에 대한 정확도와 평균 사고 단계를 정밀 측정합니다.

  실행 방법
  이제 아래 명령어로 모든 실험을 한 번에 시작할 수 있습니다:
   python looplm/run_experiments.py


  실행 중 발생하는 오류와 분석 결과는 지속적으로 root.md에 지식 자산으로 축적될 것입니다. 이제 Blackwell 하드웨어의
  성능을 확인하러 가셔도 좋습니다! 🚀