# Root Cause Analysis v5: The Zero-Bias & NaN Protection

**Date**: February 21, 2026
**Conclusion**: `PaddingMode.ZERO`가 `exp(0)=1`을 유발하여 확률 분포를 오염시키고, `-inf` 연산 시 발생하는 `NaN`이 제어 흐름을 마비시켰음.

---

## 1. [HALT BUG] The ZERO-Exp Bias
*   **Problem**: `max_prob`가 실제보다 작게 계산되어 모델이 절대 멈추지 않음.
*   **Root Cause**: `ct.load`의 `PaddingMode.ZERO`가 `vocab_size` 밖의 영역을 `0.0`으로 채움. 소프트맥스 분모(`sum_exp`) 계산 시 이들이 각각 `1.0`씩 기여하여 확률을 희석시킴.
*   **Fix**: `PaddingMode.CONSTANT`와 `constant_value=-1e20`을 사용하여 패딩 영역의 기여도를 `0`으로 만듦.

---

## 2. [STABILITY BUG] The NaN-Guard Necessity
*   **Problem**: 패딩 토큰 위치에서 계산기 비정상 종료.
*   **Root Cause**: `logits(-inf) - max_logit(-inf)` 연산 시 IEEE 754 표준에 따라 `NaN` 발생. 이 `NaN`이 `Threshold` 비교 연산에 들어가면서 모든 논리적 판단이 무력화됨.
*   **Fix**: `max_logit > -1e10` 체크를 통한 **NaN Guard** 도입. 패딩 행에 대해서는 안전한 더미 값을 사용하여 연산을 수행하되, 결과는 마스크로 걸러냄.

---

## 3. Systematic Verification
*   이제 모델은 수학적으로 순수한 소프트맥스 확률을 기반으로 조기 종료를 결정합니다.
*   하드웨어 경계(OOB)와 데이터 경계(Vocab Padding) 모두에서 수치적 안전성이 확보되었습니다.
