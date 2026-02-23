import os
import json
import time
import sys

# Ensure current dir is in path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

from eval_loop import evaluate_ood

def re_evaluate():
    experiments_dir = os.path.join(script_dir, "experiments")
    if not os.path.exists(experiments_dir):
        print(f"Error: {experiments_dir} not found.")
        return

    results = []
    subdirs = [d for d in os.listdir(experiments_dir) if os.path.isdir(os.path.join(experiments_dir, d))]
    
    print(f"Found {len(subdirs)} experiment folders. Starting re-evaluation...")

    for name in subdirs:
        ckpt_path = os.path.join(experiments_dir, name, "ckpt.pt")
        if not os.path.exists(ckpt_path):
            continue

        print(f"\n>>> Re-evaluating: {name}")
        # Run evaluation (1000 samples for better statistical significance)
        eval_res = evaluate_ood(ckpt_path, num_samples=1000)
        
        if eval_res:
            res = {
                "experiment": name,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "config": f"Re-evaluated from {ckpt_path}",
                "ood_metrics": eval_res
            }
            results.append(res)

    summary_path = os.path.join(experiments_dir, "summary_diagnostic.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=4)
    
    latest_path = os.path.join(experiments_dir, "summary_latest.json")
    with open(latest_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\n✨ Re-evaluation Complete! Results saved to {summary_path}")
    print("Run 'python generate_master_report.py' to see the new report.")

if __name__ == "__main__":
    re_evaluate()
