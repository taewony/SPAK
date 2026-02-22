# LoopLM Phase 3 & 4: 통합 최적화 계획 (v2)

Phase 2에서 우리는 커널 무결성을 확보하고 Adaptive Halting을 성공적으로 구현했습니다.  
그러나 `addition` 실험 결과, 훈련 손실은 감소하지만 검증 손실이 증가하는 **과적합(overfitting)** 패턴이 관찰되었습니다.  
이는 모델이 데이터를 암기하기 시작했다는 신호이며, 진정한 **알고리즘 일반화(Algorithmic Generalization)** 를 위해서는 아키텍처와 훈련 전략의 정교화가 필요합니다.

아래 계획은 기존 `plan.md`의 큰 틀(Phase 3: 지능 고도화, Phase 4: 성능 극대화)을 유지하면서,  
최근 분석에서 도출된 **모델 용량 조정, 정규화 강화, 아키텍처 변형 실험** 등을 통합한 확장 버전입니다.

---

## 🧪 Phase 3: Algorithmic Generalization (지능의 일반화)

**목표**: 단순 암기를 넘어 **자릿수 일반화(length generalization)** 를 달성하고, 보지 못한 긴 덧셈 문제에서도 높은 정확도를 확보한다.

### 3.1 데이터 다양성 확보 (`addition_prepare.py`)
- **샘플 수**: 5만 → **20만 개 이상**으로 대폭 확장.  
  - 2~4자리 수 조합을 다양하게 생성하여 모든 carry 패턴을 학습.
- **자릿수 분포**: 훈련 시 최대 4자리까지 포함하되, 검증 시에는 5~12자리 문제를 별도로 준비하여 OOD(Out-of-Distribution) 성능 측정.
- **데이터 증강**: 각 배치 내에서 피연산자의 순서를 무작위로 바꾸거나(`a+b` ↔ `b+a`), 등호 위치를 변형하는 등 간단한 변환을 추가하여 일반화 유도.

### 3.2 모델 용량 및 정규화 실험 (`model_loop.py`)
과적합의 주된 원인은 현재 모델(`n_embd=384`, `n_head=6`, `num_loops=12`)이 Addition 데이터에 비해 과도한 용량을 가지고 있기 때문입니다.  
다음 실험을 통해 최적의 용량을 탐색합니다.

| 실험 | 변경 사항 | 기대 효과 |
|------|-----------|-----------|
| **A1** | `n_embd=256`, `n_head=8` (또는 4) | 표현력 축소로 과적합 완화 |
| **A2** | `n_embd=128`, `n_head=4` | 최소 용량에서의 성능 확인 |
| **A3** | `dropout=0.3` 또는 `0.4` | 정규화 강화 |
| **A4** | Label smoothing (`0.1`)을 loss에 추가 | 지나친 확신(overconfidence) 방지 |
| **A5** | Weight decay 조정 (1e-1 → 1e-2) | 최적 조합 탐색 |

각 실험은 동일한 데이터/학습률로 2000 스텝 진행 후, 검증 손실과 OOD 정확도를 비교

### 3.3 Wait-to-Think 전략 (추론 단계별 루프 제어)
덧셈 문제에서 입력 구간(`123+456`)과 출력 생성 구간(`=` 이후)의 역할을 분리합니다.
- **입력 인코딩 단계**: 불필요한 반복을 줄이기 위해 `halt_threshold`를 낮게 설정(예: 0.9) → 빠르게 수렴.
- **출력 생성 단계**: `=` 토큰 이후부터는 `halt_threshold`를 높이거나(예: 0.99) 고정 루프 수(예: 24회)를 적용하여 충분한 사고 시간 부여.
- 이를 위해 `forward`에 `token_type_ids` 또는 단순히 위치 기반 분기 로직 추가.

### 3.4 OOD 테스트 및 분석
- 훈련: 최대 4자리 수
- 테스트: 5, 6, 8, 10, 12자리 수 문제
- 측정 지표:
  - **정확도(Exact Match)**
  - **평균 루프 횟수** (토큰별 Thinking Trace)
  - **루프 횟수와 자릿수의 상관관계** (길이가 길수록 더 많은 사고가 필요한지)

---

## ⚡ Phase 4: Blackwell Persistent Optimization (성능 극대화)

**목표**: RTX 5070 (또는 최신 Blackwell 아키텍처)에서 LoopLM의 추론 속도를 극한으로 끌어올리고, 아키텍처의 최종 설계를 확정한다.

### 4.1 Persistent Weight Caching Kernel (`looplm_kernels.py`)
현재는 루프마다 PyTorch를 통해 블록을 호출하지만, Phase 4에서는 **Attention + MLP 전체를 하나의 CUDA 커널로 통합**합니다.
- **공유 가중치(LM Head와 Embedding 포함)** 를 GPU의 L2 캐시/SRAM에 고정(Pinning)하여 반복적인 메모리 로드를 제거.
- **커널 인터페이스**: `h_state_padded`, `active_mask`, `steps_taken`을 입력받아 한 번의 커널 호출로 전체 루프 처리.
- 예상 효과: 루프 오버헤드 제거로 추론 속도 **2~3배 향상**.

### 4.2 X0 Injection Ablation Study
현재 설계: `h_input = h + x0 + step_enc` (매 루프마다 초기 임베딩 재주입)  
대안 설계: `h_input = h + step_enc` (첫 루프 이후 x0 제거)  
- **비교 실험**:
  - 두 방식으로 각각 학습 후, **수렴 속도, 최종 정확도, OOD 일반화 성능** 비교.
  - 또한 `h`의 노름 변화를 추적하여 "입력의 닻" 효과가 동역학에 미치는 영향 분석.
- **결과에 따라 최종 수식 결정** (Phase 4 말에 아키텍처 확정).

### 4.3 Learned Halt Gate
현재 `max_prob < threshold` 방식은 고정된 임계값에 의존합니다.  
더 정교한 정지 결정을 위해 작은 **Halt Head** (MLP + sigmoid)를 추가합니다.
- 구조: `halt_prob = sigmoid( W·[h; step_enc] )`
- 학습: 각 스텝마다 이진 분류 손실(정지 여부)을 추가하여, 모델이 스스로 정지 시점을 학습하도록 함.
- 이렇게 하면 고정 threshold 없이도 adaptive computation이 가능하며, 일반화 성능 향상에 기여할 수 있음.

### 4.4 최종 성능 벤치마크
- **지연 시간(Latency)**: 토큰당 평균 처리 시간 (ms)
- **처리량(Throughput)**: 초당 생성 토큰 수
- **에너지 효율**: FLOPs 당 정확도

위 지표를 측정하여 기존 Transformer 및 다른 adaptive 모델과 비교.
---
## 📅 실행 일정 (마일스톤)
| 단계 | 내용 | 예상 기간 | 산출물 |
|------|------|----------|--------|
| **3.1** | 데이터 확장 및 증강 파이프라인 구축 |  | `addition_prepare.py` 업데이트 |
| **3.2** | 모델 용량/정규화 실험 (A1~A5) |  | 실험 결과 테이블, 최적 설정 도출 |
| **3.3** | Wait-to-Think 전략 구현 |  | `model_loop.py` 업데이트 |
| **3.4** | OOD 테스트 및 분석 |  | 분석 리포트 |
| **4.1** | Persistent Weight Caching Kernel 개발 |  | 통합 커널, 벤치마크 |
| **4.2** | X0 Injection Ablation |  | 최종 수식 결정 |
| **4.3** | Learned Halt Gate 구현 |  | Halt Head 통합 모델 |
| **4.4** | 최종 벤치마크 및 문서화 |  | 최종 리포트, 모델 체크포인트 |

---
## 🧠 성공 기준 (Success Metrics)
- **OOD 정확도**: 8자리 덧셈에서 **70% 이상** (Zero-shot)
- **일반화**: 4자리 훈련 모델이 12자리에서도 30% 이상 정답률 (기존 0% 대비)
- **추론 속도**: Phase 3 대비 Phase 4에서 **2배 이상** 향상
- **루프 효율**: 문제 난이도(자릿수)에 비례하여 루프 횟수 증가 (Thinking Trace 분석)
 
각 단계의 실험 결과는 `root.md`에 지속적으로 기록하여 지식 자산으로 축적합니다.
**Let's make LoopLM think deeper and faster!** 🚀