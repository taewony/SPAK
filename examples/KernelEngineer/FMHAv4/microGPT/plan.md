# MicroGPT Compound Engineering Plan

## 🎯 Objective
Transform Karpathy's `microgpt.py` (atomic scalar autograd) into a high-performance, tensor-based GPU implementation using `cuTile`, while inheriting hardware-optimized design laws from the FMHAv4 project.

## 🏗 Engineering Strategy

### 1. Semantic Lifting & Inheritance
- **Attention**: Inherit the "Blackwell-Optimal" configuration (64x64 tiles, TMA latency hints K:2/V:5) from FMHAv4.
- **Normalization**: Integrate the persistent `RMSNorm` kernel from NVIDIA `TileGym`, applying the "Rows > 2*SMS" persistence heuristic.
- **MLP**: Vectorize scalar loops into `ct.mma` (Tiled MatMul) operations.

### 2. Dual-Loop Cognitive Goals
- **Outer Loop (Architect)**: Evolve `microgpt_system_v1.dsl` to hold the "Transferred Knowledge" from FMHA to a full model context.
- **Inner Loop (Engineer)**: Verify convergence on `names.txt` and measure "Apple-to-Apple" speedup against the scalar baseline.

---

## 🚀 Roadmap & Status

### Phase 1: Infrastructure (COMPLETED)
- [x] **DSL Initialization**: Created `microgpt_system_v1.dsl` with baseline metrics and design axes.
- [x] **Kernel Implementation**: Developed `microgpt_cutile.py` with GQA-ready attention and persistent RMSNorm.
- [x] **Training Harness**: Developed `train_microgpt_cutile.py` matching Karpathy's hyperparams and data loading.

### Phase 2: Distributed Execution (ACTIVE)
- [ ] **Baseline Sweep**: Run original `microgpt.py` to establish exact CPU ms/step.
- [ ] **Tiled Sweep**: Run `train_microgpt_cutile.py` on the RTX 5070 node.
- [ ] **Trace Synchronization**: Ingest `microgpt_train_trace.json`.

### Phase 3: Knowledge Crystallization
- [ ] **Fidelity Proof**: Confirm 100-step loss curve parity.
- [ ] **Compound Update**: Update DSL with verified MicroGPT performance facts.

---

## 📝 Engineering Notes

### Compounding Insights
- **From FMHAv4**: We are reusing the exact `microgpt_attention_kernel` logic, which proved 1.11x faster than PyTorch Native.
- **From RMSNorm**: We discovered that disabling TMA in `ct.store` for normalization can yield a 30% performance boost—this is encoded as a rule in our DSL.

### Targets
- **Speedup**: >100x vs. scalar Python loops.
- **Convergence**: Loss < 2.3 at step 100.
