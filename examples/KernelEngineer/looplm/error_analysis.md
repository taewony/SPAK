지금 주신 `looplm_kernels.py`를 기준으로 **수렴 실패 + 비정상 iterative latent reasoning**의 원인을 구조적으로 점검하겠습니다.

# 🔥 가장 의심되는 부분부터 말하겠습니다

## ❗❗ Halt Kernel의 max_prob 계산이 잘못되었습니다.

현재 코드:

```python
max_logit = ct.max(logits, axis=1, keepdims=True)
exp_logits = ct.exp(logits - max_logit)
sum_exp = ct.sum(exp_logits, axis=1, keepdims=True)
max_prob = 1.0 / sum_exp
```

이건 수학적으로:

[
\text{max_prob} = \frac{1}{\sum_j e^{z_j - z_{max}}}
]

이 값은:

[
\frac{e^{z_{max}}}{\sum_j e^{z_j}}
]

이 아닙니다.

---

## 🎯 정확한 softmax max prob는

[
\max_i \left( \frac{e^{z_i}}{\sum_j e^{z_j}} \right)
]

지금 구현은:

[
\frac{1}{\sum_j e^{z_j - z_{max}}}
]

이는 실제 max prob보다 **항상 작습니다.**

---

## 🔥 결과

halt 조건:

```python
is_active_next = (max_prob < Threshold) & was_active
```

max_prob가 실제보다 작으므로:

* threshold 0.9이면
* 거의 항상 < 0.9
* 계속 active
* halt 거의 안 됨

→ iterative reasoning이 끝나지 않음
→ steps 과도하게 증가
→ 수렴 실패

---

# ✅ 올바른 계산 방법

수정:

```python
max_logit = ct.max(logits, axis=1, keepdims=True)
exp_logits = ct.exp(logits - max_logit)
sum_exp = ct.sum(exp_logits, axis=1, keepdims=True)
max_prob = ct.max(exp_logits / sum_exp, axis=1, keepdims=True)
```

또는 더 단순:

```python
softmax = exp_logits / sum_exp
max_prob = ct.max(softmax, axis=1, keepdims=True)
```

---

# 🔥 이게 왜 치명적인가?

LoopLM의 halt는:

> confidence 기반 adaptive depth

입니다.

confidence 계산이 틀리면:

* depth가 비정상
* 일부 토큰은 never halt
* 일부는 즉시 halt
* gradient distribution 붕괴

→ latent dynamics 이상

---

# 🔴 두 번째 문제: padding_mode=ZERO in logits load

```python
logits = ct.load(... padding_mode=ct.PaddingMode.ZERO)
```

하지만 Python side에서:

```python
logits_padded.fill_(-inf)
```

로 채우고 있음.

padding_mode ZERO는:

* OOB 접근 시 0 채움
* 0 logit → exp(0)=1

즉:

padding row에서 sum_exp 왜곡 가능.

이건 매우 위험.

padding_mode는 NONE이 더 안전.

---

# 🔴 세 번째 문제: mask 업데이트 논리

```python
was_active = mask > 0.5
is_active_next = (max_prob < Threshold) & was_active
```

이건 논리상 맞습니다.

하지만 주의:

```python
steps_updated = steps + ct.astype(was_active, ct.int32)
```

이건:

* halt된 step에서도 1 증가
* 즉 halt iteration도 step에 포함

이건 설계 의도에 따라 다름.

---

# 🔴 네 번째 문제: H_current overwrite timing

```python
h_updated = ct.where(was_active, h_next, h_curr)
```

좋은데, 만약 max_prob 계산 오류로:

* 항상 active
* h_curr가 계속 h_next로 overwrite

→ 상태가 폭주 가능

---

# 🔴 다섯 번째: Attention kernel 안정성

attention kernel은 대체로 문제 없어 보입니다.

다만:

```python
m_ij = max(m_i, ct.max(qk, axis=-1, keepdims=True) * qk_scale_log2)
```

이 부분은:

FlashAttention 안정화 변형인데,

`* qk_scale_log2` 위치가 의도와 다르면
softmax 안정성 깨질 수 있음.

하지만 현재 문제는 halt kernel 쪽이 훨씬 유력합니다.

---

# 🔬 왜 훈련이 수렴하지 않는가?

training 모드에서는 kernel bypass 합니다.

하지만:

* inference 중 thinking trace 이상
* latent dynamics 비정상

은 halt kernel 때문일 가능성 매우 높음.

특히:

> confidence 계산 오류 → halt 논리 붕괴

---

# 🎯 핵심 진단 요약

| 문제                       | 영향          |
| ------------------------ | ----------- |
| ❌ max_prob 계산 오류         | halt 거의 안 됨 |
| ⚠ padding_mode ZERO      | 확률 왜곡 가능    |
| ⚠ halt iteration step 증가 | depth 과대 계산 |
| ⚠ attention 안정화 검증 필요    | 잠재적 수치 불안정  |

---

# 🔥 가장 중요한 수정 1개

max_prob 계산을 반드시 고치십시오.

현재 구현은 수학적으로 잘못되었습니다.

이게 iterative reasoning 붕괴의 90% 확률 원인입니다.

---

# 🚀 추가 진단 제안

1. max_prob 평균 출력
2. halt threshold 대비 실제 max_prob 분포 확인
3. steps histogram 확인
4. threshold 0.5로 낮춰 실험


