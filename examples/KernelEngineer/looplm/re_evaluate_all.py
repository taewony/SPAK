import os
import torch
import json
from eval_loop import evaluate_ood

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    exp_base = os.path.join(script_dir, "experiments")
    
    # 분석 대상 실험군 정의 (Phase 4 ~ Phase 7)
    target_exps = [
        {"name": "Exp1_Baseline_RoPE_Fixed", "label": "GPT-12L (Static)"},
        {"name": "Exp2_LoopLM_RoPE_Fixed", "label": "LoopLM-12 (Dynamic)"},
        {"name": "Exp3_LoopLM_Ultimate_Thinking", "label": "LoopLM-30 (Deep)"},
        {"name": "Exp4_LoopLM_Narrow_Deep_Time", "label": "LoopLM-128e (Efficient)"},
        {"name": "Exp5_LoopLM_Forced_Grokking", "label": "LoopLM-Grok (High-Reg)"},
        {"name": "Exp6_Baseline_Small", "label": "GPT-1L (Control)"},
        {"name": "Exp7_Grokking_Marathon", "label": "LoopLM-100k (Marathon)"},
        {"name": "Exp8_SwiGLU_Dynamic", "label": "LoopLM-SwiGLU (Advanced)"}
    ]
    
    results = {}
    
    print("="*80)
    print("🧠 LoopLM Intelligence Generalization & Reasoning Depth Evaluation")
    print("="*80)

    for exp in target_exps:
        ckpt_path = os.path.join(exp_base, exp["name"], "ckpt.pt")
        if not os.path.exists(ckpt_path):
            print(f"⚠️  Skipping {exp['name']}: Checkpoint not found.")
            continue
            
        print(f"\n🔍 Analyzing {exp['label']}...")
        # OOD 평가 수행 (샘플 수를 200개로 늘려 신뢰도 확보)
        res = evaluate_ood(ckpt_path, num_samples=200)
        if res:
            results[exp['label']] = res

    # --- 최종 지능 대조 리포트 생성 ---
    report_content = []
    report_content.append("="*80)
    report_content.append("📊 FINAL INTELLIGENCE COMPARISON REPORT")
    report_content.append("="*80)
    report_content.append(f"{'Model Architecture':<20} | {'OOD Acc':<10} | {'Avg Steps':<10} | {'Reasoning Type'}")
    report_content.append("-" * 80)

    for label, res in results.items():
        acc = res['accuracy'] * 100
        steps = res['avg_steps']
        r_type = "Fixed (Static Depth)" if label.startswith("GPT") else "Adaptive (Dynamic Depth)"
        report_content.append(f"{label:<20} | {acc:>8.2f}% | {steps:>10.2f} | {r_type}")

    report_content.append("\n📈 Digit-wise Accuracy (Generalization Curve)")
    report_content.append(f"{'Model':<20} | {'1-4d':<8} | {'5-6d':<8} | {'8d':<8} | {'10d':<8} | {'12d':<8}")
    report_content.append("-" * 80)
    
    for label, res in results.items():
        b = res['buckets']
        def get_acc(size):
            data = b.get(size, [0, 0, 0, 0])
            return (data[0]/data[1]*100) if data[1] > 0 else 0.0
        report_content.append(f"{label:<20} | {get_acc(1):>7.1f}% | {get_acc(6):>7.1f}% | {get_acc(8):>7.1f}% | {get_acc(10):>7.1f}% | {get_acc(12):>7.1f}%")

    # 파일 저장 및 출력
    final_report = "\n".join(report_content)
    print("\n" + final_report)
    
    report_path = os.path.join(script_dir, "INTELLIGENCE_REPORT.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(final_report)
    print(f"\n✅ Report saved to: {report_path}")

    print("\n💡 Conclusion:")
    if "LoopLM-30 (Deep)" in results:
        loop_acc = results["LoopLM-30 (Deep)"]["accuracy"]
        base_acc = results.get("GPT-12L (Static)", {"accuracy": 0})["accuracy"]
        if loop_acc > base_acc:
            print(f"-> LoopLM confirms Algorithmic Generalization! Superior by { (loop_acc-base_acc)*100:.1f}% in OOD.")
        else:
            print("-> Models still in Memorization phase. More 'Grokking' iterations required.")

if __name__ == "__main__":
    main()
