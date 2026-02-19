# NanoGPT Production Engineering Plan

## 🎯 Objective
Scale the SPAK "Compound Engineering" methodology to a production-grade GPT-2 architecture (768 dim, LayerNorm, GELU), leveraging the modular `TileGym` operations library.

## 🏗 Engineering Strategy

### 1. Modular Backend (TileGym)
- **Attention**: Utilize `tilegym.ops.fmha` with inherited Blackwell-optimal configs (64x64 tiles, K:2/V:5 latency).
- **Normalization**: Utilize `tilegym.ops.layer_norm_legacy` or `persistent_layer_norm` for TMA-aware normalization.
- **Path Handling**: Support dynamic loading of `TileGym` source located within the model directory.

### 2. Architectural Invariants
- **Weight Tying**: Enforce `wte.weight == lm_head.weight` to match GPT-2/nanoGPT semantics.
- **Residual Scaling**: Apply `1/sqrt(2*L)` scaling to residual projections.
- **Stability**: Ensure `-1e20` safety floor in all attention mechanisms.

---

## 🚀 Roadmap & Status

### Phase 1: Modular Implementation (COMPLETED)
- [x] **GPT Architecture**: Ported `model.py` to `nanogpt_cutile.py` with `TileGym` dispatchers.
- [x] **Fallback Layer**: Implemented robust PyTorch fallbacks for non-GPU or missing-source environments.
- [x] **Mixed Precision**: Integrated `torch.amp` for stable `float16/32` training.

### Phase 2: Distributed Training (ACTIVE)
- [ ] **Data Prep**: Generate `train.bin` for `shakespeare_char`.
- [ ] **Execution**: Run `train_nanogpt_cutile.py` on RTX 5070 node.
- [ ] **Verification**: Confirm loss matches PyTorch baseline (~1.47 on Shakespeare).

---

## 📝 Engineering Notes

### Compounding Insights
- **From MicroGPT**: The `-1e20` safety floor is now standard in our attention class.
- **From FMHAv4**: 1.11x speedup is the target for the attention blocks.
- **Modular Import**: We use a dynamic path check to import `TileGym` if it's placed inside the `nanoGPT` folder.
