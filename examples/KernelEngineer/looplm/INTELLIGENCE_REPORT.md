================================================================================
📊 FINAL INTELLIGENCE COMPARISON REPORT
================================================================================
Model Architecture   | OOD Acc    | Avg Steps  | Reasoning Type
--------------------------------------------------------------------------------
GPT-12L (Static)     |    21.00% |      12.00 | Fixed (Static Depth)
LoopLM-12 (Dynamic)  |    25.50% |      12.00 | Adaptive (Dynamic Depth)
LoopLM-30 (Deep)     |    31.50% |      30.00 | Adaptive (Dynamic Depth)
LoopLM-128e (Efficient) |    24.50% |      24.00 | Adaptive (Dynamic Depth)
LoopLM-Grok (High-Reg) |    21.00% |      12.00 | Adaptive (Dynamic Depth)
GPT-1L (Control)     |     6.00% |       1.00 | Fixed (Static Depth)
LoopLM-100k (Marathon) |    21.50% |      12.00 | Adaptive (Dynamic Depth)

📈 Digit-wise Accuracy (Generalization Curve)
Model                | 1-4d     | 5-6d     | 8d       | 10d      | 12d     
--------------------------------------------------------------------------------
GPT-12L (Static)     |   100.0% |    35.5% |     0.0% |     0.0% |     0.0%
LoopLM-12 (Dynamic)  |   100.0% |    64.5% |     0.0% |     0.0% |     0.0%
LoopLM-30 (Deep)     |   100.0% |   100.0% |     2.1% |     0.0% |     0.0%
LoopLM-128e (Efficient) |   100.0% |    58.1% |     0.0% |     0.0% |     0.0%
LoopLM-Grok (High-Reg) |   100.0% |    35.5% |     0.0% |     0.0% |     0.0%
GPT-1L (Control)     |    54.5% |     0.0% |     0.0% |     0.0% |     0.0%
LoopLM-100k (Marathon) |   100.0% |    38.7% |     0.0% |     0.0% |     0.0%