# MicroGPT Compound Engineering Plan

## 🎯 Objective
Transform Karpathy's `microgpt.py` (atomic scalar autograd) into a high-performance, tensor-based GPU implementation using `cuTile`, while inheriting hardware-optimized design laws from the FMHAv4 project.

## 🏗 Engineering Strategy

### 1. Semantic Lifting & Inheritance
... (omitted) ...

### 2. Sanity Verification (Pre-Sync)
- **Logical Parity**: Every new cuTile kernel/block must pass `microgpt_sanity_check.py` against a PyTorch reference.
- **Initialization Check**: Verify `.to(device)` and `.half()` parameter registration on the Conceptual Node.

### 3. Dual-Loop Cognitive Goals
- **Outer Loop (Architect)**: Evolve `microgpt_system_v1.dsl` to hold the "Transferred Knowledge" from FMHA to a full model context.
- **Inner Loop (Engineer)**: Verify convergence on `names.txt` and measure "Apple-to-Apple" speedup against the scalar baseline.

---

## 🚀 Roadmap & Status

### Phase 1: Infrastructure (COMPLETED)
- [x] **DSL Initialization**: Created `microgpt_system_v1.dsl` with baseline metrics and design axes.
- [x] **Kernel Implementation**: Developed `microgpt_cutile.py` with GQA-ready attention and persistent RMSNorm.
- [x] **Training Harness**: Developed `train_microgpt_cutile.py` matching Karpathy's hyperparams and data loading.

### Phase 2: Distributed Execution (ACTIVE)
- [x] **Initialization Check**: Passed on Conceptual Node (Windows PC).
- [ ] **Baseline Sweep**: Run original `microgpt.py` to establish CPU ms/step.
- [ ] **Parity Check**: Run `microgpt_sanity_check.py` on RTX 5070 Node.
- [ ] **Tiled Training**: Run `train_microgpt_cutile.py` on the RTX 5070 Node.
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
