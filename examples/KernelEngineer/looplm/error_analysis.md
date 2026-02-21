
# 4️⃣ ⚠ 진짜 문제 가능성: padding_mode=ZERO

```python
logits = ct.load(... padding_mode=ct.PaddingMode.ZERO)
```

이것이 매우 위험합니다.

왜냐하면:

Python side에서:

```
logits_padded.fill_(-inf)
```

로 padding을 채우고 있습니다.

그런데 padding_mode=ZERO는:

* OOB 접근 시 0 반환
* 0 logit → exp(0) = 1

이게 sum_exp에 들어가면:

[
\sum e^{z_j - z_{max}}
]

이 padding 개수만큼 +1 증가합니다.

결과:

[
max_prob = \frac{1}{\text{실제 sum} + \text{padding count}}
]

→ max_prob가 실제보다 작아짐
→ halt 거의 안 됨
→ 계속 active
→ iterative reasoning 비정상

이게 지금 가장 강한 의심 포인트입니다.

---

# 5️⃣ ⚠ 또 하나의 구조적 문제: padding row 처리

현재 kernel은:

```python
was_active = mask > 0.5
```

padding row는 active_mask=0.

하지만 logits는 ZERO padding.

padding row에 대해:

* mask = 0
* was_active = False
* steps 증가 안 함

여기까지는 안전.

하지만 real token tile 안에서:

* TILE_SIZE_V > V
* 오른쪽 padding 영역 ZERO

이게 확률 계산에 들어갑니다.

이건 심각합니다.