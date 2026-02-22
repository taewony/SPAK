import subprocess
import os
import json
import time

def run_command(cmd):
    print(f"Running: {cmd}")
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end='')
    process.wait()
    return process.returncode

def main():
    experiments = [
        {"name": "baseline", "args": "n_embd=384 num_loops=12 dropout=0.2"},
        {"name": "A1_low_cap", "args": "n_embd=256 n_head=4 num_loops=12"},
        {"name": "A2_very_low_cap", "args": "n_embd=128 n_head=4 num_loops=12"},
        {"name": "A3_high_dropout", "args": "n_embd=384 num_loops=12 dropout=0.4"},
        {"name": "T1_deep_thinking", "args": "n_embd=256 n_head=4 num_loops=24"}, # Increased loops
    ]

    results = []
    script_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    summary_filename = f"summary_{timestamp}.json"
    summary_path = os.path.join(script_dir, "experiments", summary_filename)
    
    # Ensure experiments dir exists
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)

    for exp in experiments:
        name = exp["name"]
        args_str = exp["args"]
        formatted_args = " ".join([f"--{a}" for a in args_str.split()])
        out_dir_rel = f"experiments/{name}"
        
        # ... (Training and Evaluation logic) ...
        
        # 3. Save combined result with trace links
        res = {
            "experiment": name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": formatted_args,
            "ood_metrics": eval_res,
            "paths": {
                "ckpt": os.path.join(script_dir, out_dir_rel, "ckpt.pt"),
                "trace": os.path.join(script_dir, out_dir_rel, "looplm_trace.json")
            }
        }
        results.append(res)
        
        print(f"[{name}] Experiment completed and metrics indexed.")
        
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=4)
        
        # Also maintain a 'latest' symlink-like copy
        latest_path = os.path.join(script_dir, "experiments", "summary_latest.json")
        with open(latest_path, "w") as f:
            json.dump(results, f, indent=4)

    print("\n" + "#"*60)
    print("🏁 ALL EXPERIMENTS COMPLETE")
    print("#"*60)
    
    # Simple summary table print
    print("\nSummary Table:")
    print(f"{'Experiment':<20} | {'Accuracy':<10} | {'Avg Steps':<10}")
    print("-" * 46)
    for r in results:
        acc = r['ood_metrics']['accuracy'] * 100 if r['ood_metrics'] else 0
        steps = r['ood_metrics']['avg_steps'] if r['ood_metrics'] else 0
        print(f"{r['experiment']:<20} | {acc:>8.2f}% | {steps:>10.2f}")

if __name__ == "__main__":
    main()
