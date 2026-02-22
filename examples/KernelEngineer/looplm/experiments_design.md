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

### 3.1. Bucketized OOD Accuracy
- 테스트 데이터를 자릿수별(5, 6, 8, 10, 12)로 분류하여 각 버킷에서의 정답률(Exact Match)을 측정.
- **Key Insight**: 어떤 지점에서 모델의 논리가 붕괴되는지 파악.

### 3.2. Complexity-Loop Correlation
- **Baseline**: 모든 문제에 대해 12 레이어 고정 소모.
- **LoopLM**: 문제의 난이도에 따라 소모된 `Avg Steps` 기록.
- **상관관계 공식**: $Corr(Digits, Steps)$ 가 1에 가까울수록 지능적임.

### 3.3. Generalization Gap
- $Loss_{train}$ 과 $Accuracy_{OOD}$ 사이의 간격 측정 (Grokking 발현 여부 판단).

## 4. 데이터셋 세부 사양 (Data Generation Strategy)

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
