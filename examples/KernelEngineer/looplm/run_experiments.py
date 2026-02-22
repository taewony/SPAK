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
    
    for exp in experiments:
        name = exp["name"]
        args = exp["args"]
        out_dir = f"looplm/experiments/{name}"
        os.makedirs(out_dir, exist_ok=True)
        
        print(f"
=== Starting Experiment: {name} ===")
        
        # 1. Train
        train_cmd = f"python looplm/train_loop.py {args} out_dir={out_dir} max_iters=2000"
        ret = run_command(train_cmd)
        if ret != 0:
            print(f"Experiment {name} failed at training.")
            continue
            
        # 2. Evaluate OOD
        ckpt_path = f"{out_dir}/ckpt.pt"
        eval_cmd = f"python looplm/eval_loop.py {ckpt_path}"
        # We'll run it and capture the output or import the function
        # For simplicity, let's import it here
        from eval_loop import evaluate_ood
        eval_res = evaluate_ood(ckpt_path, num_samples=200)
        
        # 3. Save combined result
        res = {
            "experiment": name,
            "config": args,
            "ood_metrics": eval_res
        }
        results.append(res)
        
        with open(f"looplm/experiments/summary.json", "w") as f:
            json.dump(results, f, indent=4)

    print("
=== All Experiments Complete ===")
    print(json.dumps(results, indent=4))

if __name__ == "__main__":
    main()
