# LoopLM vs. Standard GPT: OOD Intelligence Experiment Design

본 문서는 동일한 데이터셋과 연산 예산 하에서 `Standard GPT`와 `LoopLM`의 알고리즘적 일반화(Grokking) 능력을 비교 분석하기 위한 실험 설계를 정의한다.

## 1. 실험 목표 (Objectives)
- **H1 (Generalization)**: LoopLM이 고정 깊이의 GPT보다 자릿수 확장(OOD)에 대한 저항력이 강한가?
- **H2 (Adaptive Compute)**: LoopLM의 사고 단계(Avg Steps)가 문제의 난이도(자릿수/올림 횟수)와 양의 상관관계를 갖는가?
- **H3 (Efficiency)**: 동일한 정확도를 달성하기 위해 필요한 물리적 파라미터 수의 효율성 비교.

## 2. 실험군 정의 (Experimental Groups)

| 그룹 ID | 모델 유형 | 설정 | 파라미터 수 | 최대 사고 깊이 |
| :--- | :--- | :--- | :--- | :--- |
| **CTRL-12L** | Standard GPT | 12 Layers (Static) | ~85M | 12 (Fixed) |
| **LOOP-12S** | LoopLM | 1 Layer x 12 Loops | ~7M | 12 (Adaptive) |
| **LOOP-32D** | LoopLM | 1 Layer x 32 Loops | ~7M | 32 (Adaptive) |

## 3. 평가 메트릭 (Metrics)

## 3. 평가 메트릭 (Metrics for Intelligence)

### 3.1. Compute-Efficiency Score (CES)
- **정의**: $(Accuracy_{OOD} / Total\_FLOPs)$. 
- **주장**: LoopLM은 표준 GPT와 동일한 최대 연산 기회를 가졌을 때, 실제로는 평균적으로 더 적은 연산을 사용하여 더 높은 정확도를 달성함을 증명.

### 3.2. Step-Complexity Elasticity
- **정의**: 자릿수 증가($\Delta Digits$)에 따른 사고 단계 증가($\Delta Steps$)의 민감도.
- **Normal vs Reverse**: 
    - **Normal**: 모든 자릿수에서 Max Steps 소모 (불안으로 인한 연산 포화).
    - **Reverse**: 자릿수/난이도에 비례하여 Steps가 선형적으로 증가 (지능적 연산 배분).

## 4. 최종 리포트 핵심 Claim: "The Reverse-Adaptive Synergy"

실험 결과가 다음과 같이 나올 경우, 우리는 **"지능형 알고리즘 에이전트"**의 탄생을 선언한다.

1.  **동등 연산 비교**: GPT-12L(85M)의 12층 연산량과 LoopLM(7M)의 평균 12회 루프 연산량이 동일할 때, LoopLM의 OOD 성적이 압도적으로 높음.
2.  **연산 방향의 기적**: Reverse 데이터셋을 사용할 때만 `Avg Steps`가 난이도에 따라 탄력적으로 변함 (Wait-to-Think의 실질적 발현).
3.  **파라미터 경제성**: 12배 적은 파라미터로도 시간적 반복을 통해 고차원 논리 구조를 형성함.

Master Report의 신뢰성을 위해 두 가지 서로 다른 논리 구조의 데이터셋을 생성하여 사용한다. 모든 데이터셋은 `Grokking` 유도를 위해 **Bridge Data** 전략을 공통적으로 채택한다.

### 4.1. 공통 생성 규칙 (Bridge Curriculum)
- **Train Samples**: 200,000개
- **OOD Samples**: 5,000개 (5~12자리)
- **Bridge Strategy**: 훈련 데이터의 95%는 1~4자리로 구성하되, **5%의 5~6자리 데이터**를 명시적으로 혼합하여 자릿수 확장의 연결고리를 제공함.

### 4.2. 데이터셋 유형 및 포맷
| 유형 ID | 생성 스크립트 | 데이터 포맷 (예시) | 논리적 특징 |
| :--- | :--- | :--- | :--- |
| **Normal** | `addition_prepare.py` | `123+456=579` | **Non-Causal**: Transformer의 추론 방향과 연산 방향이 충돌함 (난이도 높음) |
| **Reverse** | `addition_reverse_prepare.py`| `123+456=975` | **Causal**: 일의 자리부터 즉시 생성. Transformer의 특성에 최적화됨 (Grokking 유력) |

## 5. 실험 및 리포트 생성 워크플로우 (Data Flow)

1.  **Step 1**: `addition_reverse_prepare.py`를 실행하여 20만 개의 '지능 최적화형' 데이터를 생성한다.
2.  **Step 2**: `run_experiments.py`를 통해 `R1~R4` (Reverse 모델) 실험을 수행한다.
3.  **Step 3**: 기존에 수행된 `baseline` (Normal 모델) 결과와 `R1~R4` 결과를 추출한다.
4.  **Step 4**: `eval_loop.py`의 자릿수별 버킷 데이터를 취합하여 최종 **Master Intelligence Report**를 작성한다.

실험 종료 후 다음과 같은 비교표를 생성한다.

```markdown
### [Master Report] Standard GPT vs LoopLM Intelligence Comparison

| Metrics | GPT-12L (Baseline) | LoopLM (Dynamic) | Difference |
| :--- | :---: | :---: | :---: |
| 4-digit Acc (Train) | 99.9% | 99.5% | -0.4% |
| 8-digit Acc (OOD)   | 5.0%  | 85.0% | +80.0% |
| 12-digit Acc (OOD)  | 0.0%  | 45.0% | +45.0% |
| Avg Steps (12-dig)  | 12.0  | 31.5  | +19.5 (Deep) |
| Efficiency (Acc/Params) | 1.0x | 12.1x | 12배 효율 |
```

## 6. 결론 도출 가이드
- 만약 **GPT-12L**의 OOD 성적이 처참하고, **LoopLM**이 루프를 늘려 이를 해결한다면: 
  > *"지능은 고정된 파라미터의 양보다, 상태를 정제하는 시간(Recurrence)에 더 의존한다"* 
  는 결론을 내리고 Phase 4를 종결한다.
