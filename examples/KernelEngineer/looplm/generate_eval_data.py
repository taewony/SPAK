import os
import json
import torch
import sys

# Ensure looplm is in path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

from eval_loop import evaluate_ood

def main():
    print("🚀 [RTX 5070] Starting Paper Evaluation Data Generation...")
    
    # 논문에 들어갈 핵심 엔트리 5개 정의
    # LoopLM-12 (Test-Time 24)는 Exp2의 가중치를 가져와서 루프만 24회로 늘려 평가합니다.
    paper_entries = [
        {"name": "GPT-12L (Static)", "path": "experiments/Exp1_Baseline_RoPE_Fixed/ckpt.pt", "max_loops": None},
        {"name": "LoopLM-12 (Dynamic)", "path": "experiments/Exp2_LoopLM_RoPE_Fixed/ckpt.pt", "max_loops": 12},
        {"name": "LoopLM-128e (Efficient)", "path": "experiments/Exp4_LoopLM_Narrow_Deep_Time/ckpt.pt", "max_loops": 24},
        {"name": "LoopLM-30 (Deep Thinking)", "path": "experiments/Exp3_LoopLM_Ultimate_Thinking/ckpt.pt", "max_loops": 30},
        {"name": "LoopLM-12 (Test-Time 24)", "path": "experiments/Exp2_LoopLM_RoPE_Fixed/ckpt.pt", "max_loops": 24}
    ]

    results = {}
    num_samples = 500

    for entry in paper_entries:
        name = entry["name"]
        rel_path = entry["path"]
        max_loops = entry["max_loops"]
        
        # Robust path resolution
        ckpt_path = os.path.abspath(os.path.join(script_dir, rel_path))
        
        print(f"\n>>>> Evaluating: {name}")
        print(f"     Path: {ckpt_path}")
        print(f"     Max Loops Force: {max_loops}")

        if not os.path.exists(ckpt_path):
            print(f"     ❌ FAILED: Checkpoint not found at {ckpt_path}")
            continue
            
        try:
            # Run evaluation with the specified max_loops override
            metrics = evaluate_ood(ckpt_path, num_samples=num_samples, max_loops=max_loops)
            
            if metrics is None:
                print(f"     ❌ FAILED: evaluate_ood returned None")
                continue

            buckets = metrics["buckets"]
            def safe_acc(correct, total):
                return (correct / total * 100.0) if total > 0 else 0.0

            c_56 = buckets[5][0] + buckets[6][0]
            t_56 = buckets[5][1] + buckets[6][1]
            
            extracted = {
                "avg_steps": round(metrics["avg_steps"], 2),
                "accuracy_1_4d": round(safe_acc(buckets[1][0], buckets[1][1]), 2),
                "accuracy_5_6d": round(safe_acc(c_56, t_56), 2),
                "accuracy_8d": round(safe_acc(buckets[8][0], buckets[8][1]), 2),
                "accuracy_10d": round(safe_acc(buckets[10][0], buckets[10][1]), 2),
                "accuracy_12d": round(safe_acc(buckets[12][0], buckets[12][1]), 2)
            }
            results[name] = extracted
            print(f"     ✅ SUCCESS: 1-4d({extracted['accuracy_1_4d']}%), 5-6d({extracted['accuracy_5_6d']}%)")
            
        except Exception as e:
            print(f"     ❌ ERROR during evaluation: {e}")

    # Final Save
    out_file = os.path.join(script_dir, "paper_evaluation_data.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\n🎉 ALL DONE! {len(results)} entries saved to: {out_file}")
    if "LoopLM-12 (Test-Time 24)" not in results:
        print("⚠️  WARNING: 'LoopLM-12 (Test-Time 24)' was NOT generated. Check path/errors above.")

if __name__ == "__main__":
    main()
