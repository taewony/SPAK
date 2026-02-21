# LoopLM Development Plan: Systematic Latent Reasoning (v3.0)

**Status**: Implementation Blueprint (SPAK-Driven)
**Core Engine**: cuTile + Hybrid Autograd
**Hardware Target**: NVIDIA Blackwell (RTX 5070)

---

## 1. Executive Strategy: The Compound Loop

우리는 단순히 코드를 짜는 것이 아니라, **"진화하는 설계 명세(DSL)"**를 중심으로 지식을 축적합니다.
- **Anchor**: `LoopLM_System_v1.dsl`이 모든 의사결정의 구심점(IR)이 됨.
- **Snowball**: FMHA의 TMA 최적화와 NanoGPT의 수치 안정성 규칙을 즉시 재사용.
- **Early Validation**: 프로젝트 극초기에 **Hierarchical Parity**와 **Autograd Connectivity**를 검증.

---

## 2. Phase 0: Baseline & Hierarchical Parity (The Anchor)

새로운 아키텍처이므로 "정답"인 표준 모델과의 대조군을 먼저 확립합니다.

1.  **Gold Standard (GPT-12L)**:
    - `model.py` 기반 12레이어 표준 모델 학습 (Val Loss 1.47 타겟).
    - 이 모델의 가중치를 `LoopLM`의 공유 가중치 초기값으로 활용.
2.  **Level 1 (Block Parity)**:
    - 단일 반복 블록(Attention + MLP)이 PyTorch 원본과 `Max Diff: 0.0`이 나오는지 검증.
3.  **Level 2 (Temporal Parity)**:
    - $L=12$ 반복 시, `Standard GPT(12L)`와 `LoopLM(1L x 12it)`의 출력 분포 유사성 확인.

---

## 3. Phase 1: Shared Block & Persistent Kernel (The Atom)

Blackwell 하드웨어의 성능을 극한으로 끌어올리는 커널을 설계합니다.

1.  **Persistent Weight Loader**:
    - 공유 가중치를 TMA를 통해 L2 캐시에 고정(Pinning). 루프 내 HBM 읽기 제거.
2.  **X0 Residue Injection**:
    - `h = h + LayerNorm(x0)` 규칙 구현. 루프 외부에서 로드된 `x0`를 레지스터에서 재사용.
3.  **Numerical Stability Floor**:
    - 모든 어텐션 연산에 `-1e20` 및 `flush_to_zero=True` 강제 적용.

---

## 4. Phase 2: Adaptive Depth & Halting (The Intelligence)

모델이 "언제 생각을 멈출지" 결정하는 지능을 구현합니다.

1.  **Entropy-based Halt Gate**:
    - Logits의 엔트로피를 측정하여 임계값 도달 시 연산 중단.
2.  **Masked Loop Execution**:
    - `ct.where(active_mask, ...)`를 통한 토큰별 조건부 업데이트 구현.
3.  **Thinking Trace Analysis**:
    - 루프 단계별 엔트로피 감소 곡선을 시각화하여 "추론의 질" 정량화.

---

## 5. Phase 3: Hybrid Training System (The Convergence)

NanoGPT의 교훈을 바탕으로 안정적인 학습 환경을 구축합니다.

1.  **Hybrid Autograd Mode**:
    - **Train**: PyTorch Native 연산으로 그래디언트 흐름 보장 (BPTT).
    - **Inference**: 최적화된 `cuTile` 커널로 시간적 반복 가속.
2.  **Curriculum Learning**:
    - 루프 횟수를 `2 -> 4 -> 8 -> 12`로 점진적으로 늘려 학습 안정성 확보.
3.  **Multi-step Supervision**:
    - 마지막 루프뿐만 아니라 중간 루프의 출력에도 Loss를 부여하여 조기 수렴 유도.

---

## 6. Phase 4: OOD Generalization Experiment

`loopLM`의 진정한 가치인 "알고리즘적 일반화"를 증명합니다.

1.  **Task**: 4자리수 덧셈 학습 -> 12자리수 덧셈 테스트.
2.  **Metric**: 
    - 자릿수가 늘어남에 따라 모델이 스스로 반복 횟수를 늘리는지 확인.
    - `Expected Steps ∝ Digit Count` 상관관계 분석.

---

## 7. Engineering Milestones (Checklist)

- [ ] **M1**: `test_parity.py`에서 1-step 비트 일치 달성.
- [ ] **M2**: 가중치 이식(Weight Transplant)을 통한 셰익스피어 문체 재현.
- [ ] **M3**: `train_looplm.py`에서 BPTT를 통한 손실값 하락 확인.
- [ ] **M4**: Blackwell 전용 Persistent Kernel의 2.0x 이상 가속 성능 실증.
- [ ] **M5**: 덧셈 실험에서 OOD 일반화 성공.

---

## 💎 Design Philosophy: "Thought per Joule"

우리는 단순히 더 큰 모델을 만드는 것이 아니라, **와트당 추론의 밀도**를 높입니다. `LoopLM`은 하드웨어가 "더 오래 생각하게" 함으로써, 고정된 파라미터 내에서 지능을 폭발시키는 시스템 엔지니어링의 정점이 될 것입니다.

---
✦ RTX 5070 노드에서 loopLM 프로젝트의 Step 0(Baseline)부터 Step 1(Space-Time 등가성)까지 검증

  Step 0: Standard Baseline (GPT-12L) 학습
  가장 먼저 비교 기준이 될 12레이어 표준 모델을 학습합니다. 이 과정에서 val loss가 1.5 수준으로 수렴하는지 확인합니다.

   # 1. looplm 폴더로 이동
   cd looplm

   # 2. 12레이어 표준 모델 학습 시작 (약 10~20분 소요 예상)
   python train_baseline_12l.py config/train_12l_baseline.py
   * 체크포인트: looplm/out_baseline_12l/ckpt.pt 생성 확인.
   * 로그: baseline_12l_trace.json에 기록된 수렴 곡선 확인.

  ---

  Step 1: LoopLM 아키텍처 무결성 검증
  방금 만든 12레이어 모델과 우리 LoopGPT 아키텍처가 수학적으로 일치하는지(1-step 기준) 확인합니다.

   # 아키텍처 및 등가성 테스트 실행
   python test_loop_parity.py
   * 기대 결과: Structural Integrity [PASS] 출력 및 1-step 오차 확인.

  ---

  Step 2: LoopLM (1L x 12it) 학습
  1개 레이어 가중치를 12번 반복하는 loopLM을 학습. 파라미터는 1/12이지만 연산량은 Step 0과 동일

   # LoopLM 학습 시작
   python train_loop.py
   * 체크포인트: looplm/out_looplm/ckpt.pt 생성 확인.
   * 로그: looplm_trace.json 생성 확인.

  ---

  Step 3: 결과 분석 및 비교 (Conceptual Node로 복사)
  학습이 완료되면 생성된 두 개의 트레이스 파일을 저에게 공유해 주세요.

   1. looplm/out_baseline_12l/baseline_12l_trace.json
   2. looplm/out_looplm/looplm_trace.json

  ---

  실행 시 주의사항
   * 데이터 위치: nanoGPT/data/shakespeare_char 폴더에 train.bin과 meta.pkl이 이미 준비되어 있어야 합니다.
   
✦ loopLM 프로젝트에 Phase 2 전용 커널들을 성공적으로 통합했습니다. (looplm/model_loop.py에 Adaptive Halting 로직을 성공적으로 통합했습니다. 이제 모델은 각 토큰별로 확신도(Confidence)를 측정하여 추론을 멈출 수 있습니다.)

  이제 모델은 단순히 Python 수준에서 마스킹을 처리하는 것이 아니라, `looplm_halt_update_kernel`을 통해 GPU에서
  원자적으로(Atomically) 상태 업데이트와 조기 종료를 결정합니다.

  최종 구성 요소:
   1. `looplm/looplm_kernels.py`:
       * looplm_attention_kernel: X0 주입 및 수치 안정성 강화.
       * looplm_halt_update_kernel: 엔트로피 기반 조건부 업데이트 (ct.where 활용).
   2. `looplm/model_loop.py`:
       * 추론 모드(eval)에서 halt_threshold가 주어지면 자동으로 cuTile 커널을 호출하여 연산을 가속합니다.

   cd looplm
   # 커널 기반 Thinking Trace 분석 실행
   python analyze_thinking.py

  검증 포인트 (Expected Outcome):
   1. 가변적 깊이: 쉬운 문자(예: 공백, 마침표 뒤의 첫 글자)는 적은 단계(2~4 steps)에서 멈추고, 문맥적으로 어려운 위치의
      문자는 더 많은 단계(8~12 steps)를 사용하는지 확인합니다.
   2. 효율성 측정: 모든 토큰이 12번 돌 때 대비 연산량이 얼마나 절감되었는지(Theoretical Efficiency Gain) 수치로
      확인합니다.