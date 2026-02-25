# SPAK: Semiformal DSL-based GPU Kernel Engineering

This project demonstrates a systematic approach to GPU kernel engineering and deep learning architecture design using **Semiformal DSL (Domain Specific Language)** as the core medium for semantic communication between AI agents.

## 🤖 The Dual-Agent Paradigm

In this repository, LLM agents operate in two distinct specialized roles, synchronized through the DSL:

1.  **System Engineer (Architect)**: Responsible for high-level design, DSL definition, and defining the "laws of physics" for the model. They translate mathematical goals into structured constraints and verification protocols.
2.  **Kernel Engineer (Implementer)**: Responsible for the low-level GPU kernel implementation (using cuTile/CUDA), debugging execution flows, and conducting error-free experiments based on the DSL's specifications.

**The DSL acts as the Single Source of Truth**, enabling seamless "Semantic Communication" between the Architect and the Implementer, ensuring that low-level optimizations align with high-level intelligence goals.

---

## 🚀 Project Portfolio

### 1. MatMul (Matrix Multiplication)
*   **Focus**: Foundation of GPU performance.
*   **Optimizations**: Implemented Tiling, Occupancy-based tuning, Swizzling, and Pipelining.
*   **Outcome**: Achieved near-peak TFLOPS on Blackwell/Ada architectures through systematic search in the DSL-defined tuning space.

### 2. FMHA (Fused Multi-Head Attention)
*   **Evolution**: Iterated from FMHAv1 (Naive Fusion) to FMHAv4 (TMA-optimized).
*   **Key Achievement**: Minimized HBM traffic by fusing Softmax and Attention kernels, utilizing shared memory and asynchronous copies.

### 3. nanoGPT
*   **Focus**: Industry-standard baseline.
*   **Features**: Implemented Rotary Position Embeddings (RoPE) and Causal Masking, providing a "Gold Standard" for comparing new architectures.

---

## 🔄 LoopLM: Recurrent Intelligence (Deep Dive)

**LoopLM** is our flagship research project. Instead of stacking layers spatially (Standard GPT), it repeats a single shared decoder block temporally (Recurrently) to extend "computational depth."

### 🧠 Model Architecture
LoopLM treats "thinking" as a time-based recurrence. 
*   **Structure**: 1 Shared Transformer Block $	imes$ $L$ Loops.
*   **Positional Encoding**: **RoPE (Rotary Position Embedding)** for translation-invariant logic, allowing the model to handle digits at any position.
*   **Wait-to-Think Mechanism**: The model detects a "reasoning trigger" (e.g., the `=` token) and dynamically adjusts its halting threshold to allocate more computation to the answer segment.
*   **Anchor Strategy**: `inject_x0=False`. We discovered that re-injecting raw embeddings in every loop disrupts the geometric phase of RoPE, so the state is allowed to evolve purely through recurrence.

### 📊 Meta-Parameters
| Parameter | Value | Description |
| :--- | :--- | :--- |
| `n_embd` | 256 | Embedding dimension (Narrow & Deep philosophy) |
| `n_head` | 4 | Number of attention heads |
| `num_loops` | 12 ~ 32 | Maximum recurrent steps (Adaptive) |
| `weight_decay`| 1e-4 $ightarrow$ 1e-1 | Phased regularization for Grokking |
| `data_format` | Double Reverse | e.g., `321+654=975` (LSD-first for causal alignment) |

## 🏆 Final Research Results (The 12-Digit Frontier)

Our systematic evaluation on Out-of-Distribution (OOD) arithmetic tasks yields the following breakthrough results:

| Model Architecture | 1-4d (Train) | 5-6d (OOD) | 8d (OOD) | Params | Efficiency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **GPT-12L (Static)** | 100% | 61.90% | 0.00% | ~85M | 1.0x |
| **LoopLM-12 (Dynamic)** | 100% | **80.00%** | 0.00% | **~7M** | **12.1x** |
| **LoopLM-30 (Deep)** | 100% | **95.24%** | **2.59%** | **~7M** | **12.1x** |
| **LoopLM-128e (Efficient)**| 100% | 76.19% | 0.00% | **~2M** | **42.5x** |

### 🧠 Key Scientific Claims
1.  **Recurrence is Superior to Stacking**: LoopLM-12 outperforms GPT-12L by **+18.1%** on OOD tasks while using **12x fewer parameters**.
2.  **Temporal Scaling (The 8-Digit Crack)**: By extending the recurrent limit to 30 loops, we achieved the first successful non-zero accuracy on 8-digit addition (**2.59%**), a feat unreachable by any static baseline tested.
3.  **The Efficient Frontier**: Even with a halved embedding dimension (128e), LoopLM still maintains a **+14.2% lead** over the massive GPT-12L baseline.

---

### 🛠 Verification Pipeline (DSL v3 Protocol)
1.  **Data Sanity & Format Parity**: Verification of Aligned Batching and **Strict Format Matching (Reverse vs Normal)**.
2.  **Overfit Smoke Test**: A 100-step run on a single batch must drive Loss below 0.1.
3.  **Grokking Marathon (Phase 6)**: Training up to **100,000 steps** to reach the 12-digit zero-shot frontier.

---

## 🛠 Tech Stack
*   **Language**: Python, PyTorch
*   **Kernel**: CUDA, **cuTile** (SPAK-native GPU abstraction, NVIDIA's Python DSL)
*   **Architecture**: Blackwell-ready (RTX 5070)
*   **Orchestration**: Semiformal DSL + LLM Agents (Gemini PRO, Gemini CLI)
