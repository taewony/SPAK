제공된 `AttentionFMHA.py`는 NVIDIA cuTile 라이브러리를 사용하여 **Fused Multi-Head Attention (FMHA)** 를 GPU에서 고효율로 실행하는 커널 코드입니다. 핵심은 **타일링(Tiling)**과 **Online Softmax** 기법을 통해, 대규모 Attention 계산을 GPU 메모리 계층(L1/L2 캐시, 공유 메모리, 레지스터)에 최적화하여 연산하는 데 있습니다.

주요 최적화 기법은 다음과 같습니다:

* **Fusion**: QK 연산, Softmax, PV 연산을 하나의 커널로 융합하여 중간 결과를 전역 메모리에 쓰고 읽는 비용을 제거합니다.  
* **Online Softmax**: 시퀀스 전체(`N`)를 한 번에 Softmax하지 않고, 키/밸류 타일(`TILE_N`)을 순회하며 누적된 통계량(`m_i`, `l_i`)을 업데이트합니다.  
* **Tiling**: 큰 행렬 연산을 `TILE_M`(쿼리), `TILE_N`(키/밸류), `TILE_D`(헤드 차원) 크기의 작은 타일로 분해하여 레지스터/공유 메모리 사용을 최적화합니다.

아래는 요청하신 **세 가지 Attention 구현 (단순, Fused, Fused Multi-Head)** 을 `einsum` 수식과 DSL 코드로 역공학하여 작성한 비교입니다.

---

### **1\. 단순 Attention Mechanism (Naive)**

가장 기본적인 형태로, 수학적 정의와 논리적 흐름을 나타냅니다.

**수학식 (Einsum)**: `Output[b, m, :] = softmax( (Q[b, m, :] @ K[b, :, n]^T) / sqrt(d_k) ) @ V[b, n, :]` (또는 `"bmd,bnd->bmn; bmn,bnd->bmd"`로 분리 표현)

**DSL 코드**:

\# Procedure: naive\_attention\_kernel

\# Input: Q\[B, M, D\], K\[B, N, D\], V\[B, N, D\]

\# Output: O\[B, M, D\]

\# Einsum: O\[b,m,d\] \= Σ\_n Softmax( Σ\_d Q\[b,m,d\]\*K\[b,n,d\] / √d\_k )\[n\] \* V\[b,n,d\]

procedure naive\_attention\_kernel:

    b \= blockIdx.x  \# 배치 인덱스

    m \= blockIdx.y  \# 쿼리 위치 인덱스

    \# 1\. QK 연산: \[D\] 차원 내적

    acc\_qk \= zeros(N) \# \[N\]

    for n in 0..N-1:

        for d in 0..D-1:

            acc\_qk\[n\] \+= Q\[b, m, d\] \* K\[b, n, d\]

        acc\_qk\[n\] \= acc\_qk\[n\] / sqrt(D)

    \# 2\. Global Softmax: N 전체에 대한 정규화 필요

    m\_max \= max(acc\_qk\[:\])   \# N 전체 최대값 탐색

    exp\_sum \= 0

    for n in 0..N-1:

        acc\_qk\[n\] \= exp(acc\_qk\[n\] \- m\_max)

        exp\_sum \+= acc\_qk\[n\]

    for n in 0..N-1:

        attn\[n\] \= acc\_qk\[n\] / exp\_sum \# Softmax 완료

    \# 3\. PV 연산: Attention 가중치 적용

    for d in 0..D-1:

        acc\_out\[d\] \= 0

        for n in 0..N-1:

            acc\_out\[d\] \+= attn\[n\] \* V\[b, n, d\]

        O\[b, m, d\] \= acc\_out\[d\]

**💡 특징**: `QK`, `Softmax`, `PV` 단계가 명확히 분리되고, Softmax를 위해 **전체 N 차원에 대한 최대값(`m_max`)과 합(`exp_sum`)을 먼저 계산**해야 합니다. 이는 추가적인 전역 메모리 접근이 필요합니다.

---

### **2\. Online Softmax가 적용된 Fused Attention**

N 차원을 타일로 나누어 순회하며, Softmax 통계량을 점진적으로 업데이트하여 중간 저장을 피합니다.

**수학식 (Einsum)**: `Output[b, m, :] = OnlineSoftmax( (Q[b, m, :] @ K[b, :, n_tile]^T) / sqrt(d_k) ) @ V[b, n_tile, :]`

**DSL 코드**:

\# Procedure: fused\_attention\_online\_softmax\_kernel

\# Input: Q\[B, M, D\], K\[B, N, D\], V\[B, N, D\]

\# Output: O\[B, M, D\]

\# Einsum: O\[b,m,d\] \= Σ\_n\_tile OnlineSoftmax\_Tile( Q\[b,m,d\]\*K\[b,n\_tile,d\] / √d\_k ) \* V\[b,n\_tile,d\]

procedure fused\_attention\_online\_softmax\_kernel:

    b \= blockIdx.x  \# 배치

    m\_tile \= blockIdx.y \# 쿼리 타일 (TILE\_M)

    \# Online Softmax 상태 변수 초기화

    m\_i \= \-inf  \# 현재까지 처리한 타일 중 최대값

    l\_i \= 0.0   \# 현재까지의 정규화 합

    acc \= zeros(D) \# 출력 누적기

    \# 키/밸류 타일 순회 (N 차원을 TILE\_N 크기로 나눔)

    for j in 0..(N/TILE\_N)-1:

        \# 1\. QK 타일 연산

        q\_tile \= load(Q\[b, m\_tile, 0:D\]) \# \[TILE\_M, D\]

        k\_tile \= load(K\[b, j\*TILE\_N : (j+1)\*TILE\_N, 0:D\]) \# \[TILE\_N, D\]

        qk\_tile \= matmul(q\_tile, k\_tile.T) / sqrt(D) \# \[TILE\_M, TILE\_N\]

        \# 2\. Online Softmax 업데이트 (현재 타일에 대해서만)

        m\_ij \= max(m\_i, max(qk\_tile, dim=-1)) \# \[TILE\_M, 1\]

        p\_tile \= exp(qk\_tile \- m\_ij)          \# \[TILE\_M, TILE\_N\]

        l\_ij \= sum(p\_tile, dim=-1)            \# \[TILE\_M, 1\]

        \# 3\. 이전 누적값(acc)과 현재 통계량을 조정

        alpha \= exp(m\_i \- m\_ij)

        l\_i \= l\_i \* alpha \+ l\_ij

        acc \= acc \* alpha \# 출력 누적값 조정

        \# 4\. PV 타일 연산 및 누적

        v\_tile \= load(V\[b, j\*TILE\_N : (j+1)\*TILE\_N, 0:D\]) \# \[TILE\_N, D\]

        acc \= acc \+ matmul(p\_tile, v\_tile) \# \[TILE\_M, D\]

        m\_i \= m\_ij \# 상태 업데이트

    \# 5\. 최종 정규화 및 저장

    O\[b, m\_tile, 0:D\] \= acc / l\_i

**💡 특징**: **`m_i`, `l_i`라는 상태 변수**를 유지하며 N 차원을 타일(`TILE_N`) 단위로 순회합니다. 각 타일 처리 후 누적 출력(`acc`)과 상태를 업데이트하여, **시퀀스 전체에 대한 중간 결과를 전역 메모리에 저장하지 않습니다**.

---

### **3\. Fused Multi-Head Attention (FMHA)**

Fused Attention을 **다중 헤드(Batch, Head)** 차원으로 확장하고, GQA(Grouped Query Attention)를 지원합니다.

**수학식 (Einsum)**: `Output[b, h, m, :] = OnlineSoftmax( (Q[b, h, m, :] @ K[b, h//G, :, :]^T) / sqrt(d_k) ) @ V[b, h//G, :, :]` (여기서 `G`는 `QUERY_GROUP_SIZE`)

**DSL 코드**:

\# Procedure: fused\_multihead\_attention\_kernel

\# Input: Q\[B, H, M, D\], K\[B, H/KV\_Heads, N, D\], V\[B, H/KV\_Heads, N, D\]

\# Output: O\[B, H, M, D\]

\# Einsum: O\[b,h,m,d\] \= Σ\_n\_tile OnlineSoftmax\_Tile( Q\[b,h,m,d\]\*K\[b,g,n\_tile,d\] / √d\_k ) \* V\[b,g,n\_tile,d\]

\# where g \= h // QUERY\_GROUP\_SIZE

procedure fused\_multihead\_attention\_kernel:

    \# 3D 그리드 매핑: (쿼리타일, 배치\*헤드, 1\)

    bid\_x \= blockIdx.x \# 처리할 쿼리 시퀀스 타일 (M 차원)

    bid\_y \= blockIdx.y \# 배치와 헤드를 결합한 인덱스

    \# 배치(b)와 헤드(h) 인덱스 디코딩

    batch\_idx \= bid\_y // H

    head\_idx \= bid\_y % H

    kv\_head\_idx \= head\_idx // QUERY\_GROUP\_SIZE \# GQA를 위한 KV 헤드 인덱스

    \# Online Softmax 상태 변수 초기화 (타일 단위)

    m\_i \= full((TILE\_M, 1), \-inf)

    l\_i \= full((TILE\_M, 1), 0.0)

    acc \= full((TILE\_M, D), 0.0)

    \# 현재 블록이 담당하는 쿼리 타일 로드 \[TILE\_M, D\]

    q\_tile \= load(Q\[batch\_idx, head\_idx, bid\_x\*TILE\_M : , :\])

    \# 키/밸류 타일 순회 (N 차원)

    for j in 0..(N/TILE\_N)-1:

        \# 1\. QK 타일 연산: 쿼리 헤드에 대응하는 KV 헤드 사용

        k\_tile \= load(K\[batch\_idx, kv\_head\_idx, j\*TILE\_N : , :\]) \# \[TILE\_N, D\]

        qk\_tile \= matmul(q\_tile, k\_tile.T) \* qk\_scale \# \[TILE\_M, TILE\_N\]

        \# 2\. (옵션) 캐주얼 마스킹 적용 (causal=True 시)

        if causal:

            qk\_tile \= apply\_causal\_mask(qk\_tile, bid\_x, j)

        \# 3\. Online Softmax 업데이트 (타일별)

        m\_ij \= max(m\_i, max(qk\_tile, dim=-1, keepdims=True))

        p\_tile \= exp2(qk\_tile \- m\_ij) \# cuTile은 exp2 사용

        l\_ij \= sum(p\_tile, dim=-1, keepdims=True)

        alpha \= exp2(m\_i \- m\_ij)

        l\_i \= l\_i \* alpha \+ l\_ij

        acc \= acc \* alpha

        \# 4\. PV 타일 연산 및 누적

        v\_tile \= load(V\[batch\_idx, kv\_head\_idx, j\*TILE\_N : , :\]) \# \[TILE\_N, D\]

        acc \= acc \+ matmul(p\_tile.astype(V.dtype), v\_tile)

        m\_i \= m\_ij

    \# 5\. 최종 정규화 및 저장

    O\[batch\_idx, head\_idx, bid\_x\*TILE\_M : , :\] \= acc / l\_i

**💡 특징**: **배치(`b`)와 헤드(`h`) 차원을 2D 그리드(`blockIdx.y`)로 결합**하여 병렬 처리합니다. `kv_head_idx` 계산을 통해 **GQA**를 지원하며, **캐주얼 마스킹** 로직이 추가됩니다.

---

### **📊 3가지 구현 방식의 핵심 차이점 비교**

| 특성 | 단순 Attention | Fused Attention (Online Softmax) | Fused Multi-Head Attention (FMHA) |
| :---- | :---- | :---- | :---- |
| **연산 흐름** | 단계별 분리(`QK`→`Softmax`→`PV`) | **타일 기반 Fusion** (`QK`→`Softmax 업데이트`→`PV` in loop) | **Multi-Head 병렬화 추가** |
| **소프트맥스** | Global (전체 N 대상) | **Online (타일별 누적)** | Online \+ **다중 헤드별 독립 처리** |
| **메모리 접근** | 중간 결과(`qk`, `attn`) 전역 메모리 저장 | 중간 결과 **레지스터/공유 메모리 유지**, 전역 메모리 접근 **최소화** | **헤드 간 병렬 처리**로 메모리 대역폭 활용도 향상 |
| **병렬화 축** | 주로 `B`, `M` | `B`, `M` \+ **`N`을 타일로 분할** | **`B*H`를 결합한 2D 그리드** \+ `M`, `N` 타일링 |
| **주요 최적화** | \- | **Fusion, Online Softmax** | **Multi-Head 병렬화, GQA, Causal Masking** |
| **cuTile 함수 매핑** | \- | `ct.load`, `ct.mma`, `ct.max`, `ct.sum`, `ct.exp2` | 위 함수들 \+ **그리드 매핑 (`bid()`)**, **조건부 마스킹** |

이 분석을 바탕으로, 제공된 `AttentionFMHA.py` 코드는 **3번째 Fused Multi-Head Attention (FMHA)** 을 구현한 것임을 알 수 있습니다. 이는 GPU의 메모리 계층과 대규모 병렬 처리를 극대화하기 위한 설계입니다.