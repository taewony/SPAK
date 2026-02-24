================================================================================
📊 FINAL INTELLIGENCE COMPARISON REPORT
================================================================================
Model Architecture   | OOD Acc    | Avg Steps  | Reasoning Type
--------------------------------------------------------------------------------
GPT-12L (Static)     |     1.50% |      12.00 | Fixed (Static Depth)
LoopLM-12 (Dynamic)  |     1.50% |      12.00 | Adaptive (Dynamic Depth)
LoopLM-30 (Deep)     |     1.50% |      30.00 | Adaptive (Dynamic Depth)
LoopLM-128e (Efficient) |     1.50% |      24.00 | Adaptive (Dynamic Depth)
LoopLM-Grok (High-Reg) |     1.50% |      12.00 | Adaptive (Dynamic Depth)
GPT-1L (Control)     |     1.00% |       1.00 | Fixed (Static Depth)
LoopLM-100k (Marathon) |     1.50% |      12.00 | Adaptive (Dynamic Depth)

📈 Digit-wise Accuracy (Generalization Curve)
Model                | 1-4d     | 5-6d     | 8d       | 10d      | 12d     
--------------------------------------------------------------------------------
GPT-12L (Static)     |     9.1% |     3.3% |     0.0% |     0.0% |     0.0%
LoopLM-12 (Dynamic)  |     9.1% |     3.3% |     0.0% |     0.0% |     0.0%
LoopLM-30 (Deep)     |     9.1% |     3.3% |     0.0% |     0.0% |     0.0%
LoopLM-128e (Efficient) |     9.1% |     3.3% |     0.0% |     0.0% |     0.0%
LoopLM-Grok (High-Reg) |     9.1% |     3.3% |     0.0% |     0.0% |     0.0%
GPT-1L (Control)     |     9.1% |     0.0% |     0.0% |     0.0% |     0.0%
LoopLM-100k (Marathon) |     9.1% |     3.3% |     0.0% |     0.0% |     0.0%