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
    
    # 논문에 들어갈 핵심 엔트리 5개 정의 (Path, max_loops)
    paper_entries = {
        "GPT-12L (Static)": {
            "path": "experiments/Exp1_Baseline_RoPE_Fixed/ckpt.pt",
            "max_loops": None
        },
        "LoopLM-12 (Dynamic)": {
            "path": "experiments/Exp2_LoopLM_RoPE_Fixed/ckpt.pt",
            "max_loops": 12
        },
        "LoopLM-128e (Efficient)": {
            "path": "experiments/Exp4_LoopLM_Narrow_Deep_Time/ckpt.pt",
            "max_loops": 24
        },
        "LoopLM-30 (Deep Thinking)": {
            "path": "experiments/Exp3_LoopLM_Ultimate_Thinking/ckpt.pt",
            "max_loops": 30
        },
        "LoopLM-12 (Test-Time 24)": {
            "path": "experiments/Exp2_LoopLM_RoPE_Fixed/ckpt.pt", # Exp2 체크포인트 재사용!
            "max_loops": 24 # 추론 시에만 루프를 2배로 강제 (Test-Time Compute)
        }
    }

    results = {}
    num_samples = 500 # 논문용이므로 샘플 수를 넉넉히 잡아 통계적 유의성 확보

    for name, config in paper_entries.items():
        ckpt_path = os.path.join(script_dir, config["path"])
        max_loops = config["max_loops"]
        
        print(f"\nEvaluating: {name} (max_loops={max_loops})")
        if not os.path.exists(ckpt_path):
            print(f"❌ Checkpoint not found: {ckpt_path}")
            continue
            
        try:
            # Run evaluation
            metrics = evaluate_ood(ckpt_path, num_samples=num_samples, max_loops=max_loops)
            
            # Extract only the necessary buckets for the paper
            buckets = metrics["buckets"]
            
            def safe_acc(correct, total):
                return (correct / total * 100.0) if total > 0 else 0.0

            # Combine 5 and 6 digit buckets for "5-6d"
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
            print(f"✅ Success: 1-4d({extracted['accuracy_1_4d']}%), 5-6d({extracted['accuracy_5_6d']}%)")
            
        except Exception as e:
            print(f"❌ Failed to evaluate {name}: {e}")

    # Save to standard JSON file
    out_file = os.path.join(script_dir, "paper_evaluation_data.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\n🎉 Data generation complete! File saved to: {out_file}")
    print("➡️ TRANSFER 'paper_evaluation_data.json' TO YOUR WINDOWS PC.")

if __name__ == "__main__":
    main()
