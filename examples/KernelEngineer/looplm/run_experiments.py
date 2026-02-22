import subprocess
import os
import json
import time
import sys

# Ensure looplm is in path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

try:
    from eval_loop import evaluate_ood
except ImportError:
    print("Warning: Could not import evaluate_ood from eval_loop.py")
    def evaluate_ood(*args, **kwargs): return None

def run_command(cmd):
    print(f"Running: {cmd}")
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end='')
    process.wait()
    return process.returncode

def main():
    SMOKE_TEST = False 

    # --- ⚔️ THE FINAL FAIR BATTLE: Standard GPT vs LoopLM ---
    # Matching Max Compute Budget = 12 Layers / 12 Loops
    experiments = [
        # 1. Standard GPT - Normal Direction
        {"name": "B1_Static_Normal", "args": "dataset=addition n_layer=12 n_embd=256 n_head=4 max_iters=10000"},
        # 2. Standard GPT - Reverse Direction
        {"name": "B2_Static_Reverse", "args": "dataset=addition_reverse n_layer=12 n_embd=256 n_head=4 max_iters=10000"},
        # 3. LoopLM - Normal Direction
        {"name": "L1_Dynamic_Normal", "args": "dataset=addition num_loops=12 n_embd=256 n_head=4 max_iters=10000"},
        # 4. LoopLM - Reverse Direction
        {"name": "L2_Dynamic_Reverse", "args": "dataset=addition_reverse num_loops=12 n_embd=256 n_head=4 max_iters=10000"},
    ]

    results = []
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(script_dir, "experiments", f"summary_{timestamp}.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    
    for exp in experiments:
        name = exp["name"]
        args_str = exp["args"]
        formatted_args = " ".join([f"--{a}" for a in args_str.split()])
        out_dir_rel = f"experiments/{name}"
        
        # Determine script type
        train_script = "train_baseline_12l.py" if "n_layer=12" in args_str else "train_loop.py"
        train_path = os.path.join(script_dir, train_script)
        
        print("\n" + "="*60)
        print(f"🚀 STARTING FINAL MATCH: {name}")
        print(f"   Using Script: {train_script}")
        print("="*60)
        
        # 1. Train
        ret = run_command(f"python {train_path} {formatted_args} --out_dir={out_dir_rel}")
        if ret != 0: continue
            
        # 2. Evaluate
        ckpt_path = os.path.join(script_dir, out_dir_rel, "ckpt.pt")
        eval_res = evaluate_ood(ckpt_path, num_samples=200)
        
        # 3. Save
        res = {"experiment": name, "config": formatted_args, "ood_metrics": eval_res}
        results.append(res)
        with open(summary_path, "w") as f: json.dump(results, f, indent=4)
        latest_path = os.path.join(script_dir, "experiments", "summary_latest.json")
        with open(latest_path, "w") as f: json.dump(results, f, indent=4)

    print("\n🏁 FINAL FAIR BATTLE COMPLETE. RUN generate_master_report.py NOW!")

if __name__ == "__main__":
    main()
