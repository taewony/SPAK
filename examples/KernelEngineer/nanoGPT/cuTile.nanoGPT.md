## Transforming nanoGPT SNF to cuTile GPU Kernels Using TileGym Primitives

Your SNF code for nanoGPT provides a beautifully structured, high-level specification of the GPT‑2 architecture. The next step is to systematically map each component to the **cuTile primitives** exposed by TileGym (via `ops.py`) and then generate efficient GPU kernels. This process mirrors the methodology we developed for FMHA: **semantic lifting, mapping to hardware‑aware operators, and iterative refinement**.

Below is a step‑by‑step guide to perform this transformation, leveraging the TileGym operation interface and the insights gained from your FMHA compound engineering.

---

### 1. Decompose the SNF into Atomic Operations

The SNF describes nanoGPT as a composition of well‑defined operators. We first extract the **atomic compute and memory primitives** that need GPU implementations.

| SNF Component | Description | Required Ops |
|---------------|-------------|--------------|
| `Gather(θ_wte, token)` + `Gather(θ_wpe, Range(T))` | Token & position embedding lookup | **Indexed load** (gather) from parameter tensors, **elementwise addition** |
| `LayerNorm` (multiple uses) | Normalization along channel dimension | **LayerNorm** (mean, variance, scale+shift) |
| `CausalSelfAttention` | Q/K/V projections, QK^T, softmax, PV, output projection | **Linear projections** (matrix multiplications), **FMHA** (fused attention), **causal masking** |
| `MLP` | Two linear layers with GELU activation | **Linear** (matmul), **GELU** (elementwise activation) |
| `TransformerBlock` | Residual connections around attention and MLP | **Elementwise add** (residual), **LayerNorm** (pre‑norm) |
| Output projection + weight tying | Final linear layer using token embedding weights | **Linear** with shared weights |

---

### 2. Map to TileGym Operations (`ops.py`)

The `ops.py` file provides a unified interface to cuTile (and other backend) implementations. Many of the required operations already have optimized cuTile kernels behind the scenes.

| SNF Operator | TileGym Function (from `ops.py`) | Notes |
|--------------|-----------------------------------|-------|
| **LayerNorm** | `layer_norm_legacy` or `persistent_layer_norm` | Both accept input, weight, bias, eps. The persistent version uses TMA and is tuned for Blackwell. |
| **FMHA (Causal Self‑Attention)** | `fmha(q, k, v, scaling, is_causal=True)` | This is the exact FMHA kernel we already studied! It handles Q/K/V projections? Wait – in `ops.py`, `fmha` expects Q/K/V as inputs, meaning the projections must be done separately. In GPT‑2, attention first projects x to Q/K/V via linear layers. So we need **linear projections** first. |
| **Linear Projection** | Not directly in `ops.py`; but TileGym provides **matrix multiplication** primitives (e.g., `ct.mma` in cuTile). For performance, we should use **fused operations** where possible. The `get_fused_swiglu_module` is for SwiGLU, not GELU. For GPT‑2, we need a **fused GELU linear** or at least a fast `matmul` followed by elementwise GELU. |
| **GELU Activation** | No direct GELU in `ops.py`. cuTile may have `ct.gelu` or we can implement via `erf` approximation. However, for performance, a fused kernel (linear + GELU) would be ideal – similar to the fused SwiGLU concept. |
| **Embedding Gather** | Not in `ops.py`; cuTile provides `ct.gather` or we can implement using `ct.load` with indices. TileGym likely expects embeddings to be handled at a higher level. |
| **Elementwise Add** | Simple addition can be done with `ct.add` or just `+` in cuTile; no special op needed. |
| **Weight Tying** | This is a model‑level constraint, not a kernel primitive. It means the output projection uses the same weight matrix as the token embedding. In the kernel, we just load that weight again. |

**Key observation**: The `fmha` op in TileGym **does not include the Q/K/V projections**; it expects Q, K, V already projected. Therefore, a full GPT‑2 attention layer requires:
- Three separate linear projections (or one fused QKV projection) → **matrix multiplications**.
- Then `fmha` on the projected tensors.
- Possibly an output projection (another linear).

---

### 3. Identify Missing Primitives and Design New Ones

From the mapping, we see that **linear projections** and **GELU** are not directly provided as fused ops in `ops.py`. However, TileGym’s philosophy is to provide composable primitives. For nanoGPT, we can:

- Use **cuTile’s matrix multiplication** (`ct.mma`) to implement linear layers.
- Combine with **elementwise GELU** (either a separate kernel or fuse it with the preceding matmul).

**Option A: Separate Kernels**
```
proj = ct.mma(x, W_proj)   # [B, T, C] @ [C, C] = [B, T, C]
act = ct.gelu(proj)        # elementwise GELU
```
This is simple but may have higher memory traffic.

**Option B: Fused Kernel (recommended)**
Write a custom cuTile kernel that performs:
- Load x tile and weight tile.
- Compute matmul tile (using `ct.mma`).
- Apply GELU to the result tile.
- Store output.

This fused pattern is analogous to the **fused SwiGLU** module mentioned in `ops.py` (`PartiallyFusedSwiGLUMLP`). For GPT‑2, we can create a `FusedGELUMLP` class that follows the same pattern, using cuTile’s `mma` and a custom GELU implementation (approximated via `erf` or polynomial). The TileGym codebase likely has a GELU kernel hidden somewhere; we can adapt it.

---

### 4. Generate cuTile Kernel Code from SNF

With the mapping and missing pieces identified, we can now use an LLM (like Gemini) to generate the actual cuTile kernels. The process mirrors the FMHA case:

1. **Create a Kernel Template for the Transformer Block**  
   - Input: `h[B, T, C]` (residual stream)  
   - Parameters: `θ_blocks[l]` (weights for attention and MLP)  
   - Output: updated `h`

2. **For each sub‑operation, emit cuTile code** using the patterns we developed for FMHA:
   - **LayerNorm**: Use `persistent_layer_norm` (TMA‑enabled) or implement a tiled version similar to FMHA’s online softmax but for mean/std.
   - **Attention**:
     - Project Q, K, V using fused matmul kernels (or separate `ct.mma` calls).
     - Call `fmha` (which itself is a cuTile kernel, already optimized).
   - **MLP**:
     - Use a fused GELU linear kernel (to be written) or separate matmul + GELU.
   - **Residual Add**: Simple elementwise addition.

3. **Tile Sizes and Occupancy**  
   - Leverage the knowledge from FMHA: on RTX 5070, `tile_m=64, tile_n=64` and `occupancy=2` were optimal for attention. For MLP, which is channel‑bound, different tile sizes might be better. We can either reuse the same tile parameters or allow separate tuning.

4. **Integrate with the Engineering Loop**  
   - As in FMHA, we can write an autotuner that sweeps over tile sizes, latency hints, and other knobs to find the best configuration for each component.

---

### 5. Example: Mapping the SNF to cuTile Code Snippets

Here’s a conceptual mapping for one transformer block:

```python
# SNF: n1 = LayerNorm(h_in, θ.ln1)
n1 = persistent_layer_norm(h_in, weight=θ_blocks[l].ln1.weight, bias=θ_blocks[l].ln1.bias, eps=1e-5)

# SNF: q, k, v = Linear(x).split()
# We'll use a fused QKV projection kernel (to be implemented) or separate matmuls.
qkv = fused_linear_gelu(n1, θ_blocks[l].attn.W_qkv)   # hypothetical fused kernel
q, k, v = split(qkv)                                   # view operation

# SNF: att = Softmax(q @ k.T / sqrt(D_h) + Mask_Causal)
# Use TileGym's fmha op (which internally uses cuTile's tile_fmha)
att_out = fmha(q, k, v, scaling=1.0/math.sqrt(D_h), is_causal=True)

# SNF: h_mid = h_in + Linear(att @ v)
h_mid = h_in + att_out   # Linear here is part of fmha output (already projected back)

# SNF: n2 = LayerNorm(h_mid, θ.ln2)
n2 = persistent_layer_norm(h_mid, weight=θ_blocks[l].ln2.weight, bias=θ_blocks[l].ln2.bias, eps=1e-5)

# SNF: h = GELU(Linear(n2, 4*C))
# Use fused GELU linear kernel
mlp_out = fused_gelu_linear(n2, θ_blocks[l].mlp.W1, θ_blocks[l].mlp.W2)   # hypothetical

# SNF: h_out = h_mid + Linear(h, C)
h_out = h_mid + mlp_out
```

The actual cuTile kernels for `fused_linear_gelu` and `fused_gelu_linear` would be new developments, but they follow the same template as the `PartiallyFusedSwiGLUMLP` in TileGym.

---

### 6. Role of the LLM in Code Generation

With the SNF as a precise specification, an LLM can be prompted to generate the cuTile kernel code by:

- Providing the SNF and the TileGym operation interfaces.
- Including the hardware context (RTX 5070, Blackwell, TMA).
- Specifying the desired tile sizes and occupancy (from prior knowledge).
- Using few‑shot examples (like the FMHA kernel) to guide the output.

Example prompt fragment:

```
You are an expert cuTile kernel writer. Given the following SNF specification for a GPT‑2 transformer block, generate a cuTile kernel that implements the block efficiently on RTX 5070 (Blackwell). Use the TileGym operation patterns where appropriate, and assume tile_m=64, tile_n=64, occupancy=2. Include TMA latency hints for memory loads.

SNF:
...
```

The LLM can then produce a complete Python file with the kernel definition, using `ct.mma`, `ct.load` with latency, and the necessary control flow.

---

### 7. Compound Engineering for nanoGPT

The FMHA experience taught us that **knowledge compounds**. The same principles apply here:

- **Start with a naive SNF** (v1) that only specifies correctness.
- **Iteratively refine** by adding design axes: tile sizes, latency hints, fusion strategies.
- **Autotune** to discover optimal configurations on target hardware.
- **Capture insights** (e.g., “on RTX 5070, GELU linear fusion yields 1.3× speedup”) as facts in the knowledge base.
- **Transfer knowledge** from FMHA: e.g., the optimal tile sizes for attention may also apply to the Q/K/V projections, and the TMA latency values might be reused.

By following this methodology, you can build a **self‑improving system** for GPT‑2 kernel generation, just as you did for FMHA.

---

### 8. Next Steps

1. **Implement the missing fused kernels** (GELU linear) – either by extending TileGym or writing custom cuTile kernels.
2. **Create an SNF‑to‑kernel compiler** that uses an LLM to translate the SNF into cuTile code, with the option to plug in the autotuner.
3. **Run the engineering loop** on RTX 5070 to find optimal configurations for each component.
4. **Document the discovered rules** in the DSL’s knowledge base, ready for reuse in future models.

This approach transforms nanoGPT from a static PyTorch model into a **dynamically optimizable, hardware‑aware system**, exactly as you achieved with FMHA.

Would you like me to elaborate on any specific part, such as the design of the fused GELU linear kernel or the integration with the autotuner?