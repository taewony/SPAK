# Phase 6.5: SwiGLU Integration & The Limits of Implicit Reasoning

## 1. 개요

스크래치패드(Explicit Reasoning)로 전면 전환하기 직전, 최신 LLM의 표준인 **SwiGLU(Swish-Gated Linear Unit)**를 도입하여 단일 블록의 논리적 표현력(Logical Expressivity)을 극대화합니다.

## 2. 코드 수정 계획 (Action Items)

### Step 1: model.py 업데이트

GPTConfig에 use_swiglu 플래그를 추가하고, SwiGLU 클래스를 구현합니다.

# model.py 내 추가/수정 사항
```code
class SwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 파라미터 수를 기존 MLP(4*embd)와 비슷하게 맞추기 위해 8/3 비율 사용
        # LLaMA 논문 표준 비율 적용
        hidden_dim = int(8 * config.n_embd / 3)
        
        self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.w2 = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.w3 = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # Swish(xW1) * xW2 -> W3
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        # Config에 따라 MLP 또는 SwiGLU 선택
        if getattr(config, 'use_swiglu', False):
            self.mlp = SwiGLU(config)
        else:
            self.mlp = MLP(config)
```

### Step 2: train_loop.py 파라미터 추가

use_swiglu = True 설정 변수를 추가하여 GPTConfig에 전달되도록 수정.

## 3. 검증 실험 (Exp8_SwiGLU_Dynamic)

파라미터 크기와 학습 조건을 이전에 가장 성공적이었던 LoopLM-30 (Deep)과 동일하게 맞추되, MLP만 SwiGLU로 변경하여 15,000 스텝을 학습시킵니다.

가설: SwiGLU의 조건부 게이팅 능력 덕분에, 8자리에서 2.1%로 막혔던 정확도가 10% 이상으로 상승하거나, 학습 수렴 속도가 훨씬 빨라질 것이다.