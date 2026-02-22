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
        args_str = exp["args"]
        # Convert "key=val" to "--key=val"
        formatted_args = " ".join([f"--{a}" for a in args_str.split()])
        
        out_dir = f"looplm/experiments/{name}"
        os.makedirs(out_dir, exist_ok=True)
        
        print("\n" + "="*60)
        print(f"🚀 STARTING EXPERIMENT: {name}")
        print(f"   Config: {formatted_args}")
        print(f"   Output: {out_dir}")
        print("="*60)
        
        # 1. Train
        print(f"[{name}] Step 1: Training for 2000 iterations...")
        train_cmd = f"python looplm/train_loop.py {formatted_args} --out_dir={out_dir} --max_iters=2000"
        ret = run_command(train_cmd)
        if ret != 0:
            print(f"❌ [{name}] Experiment failed during training phase.")
            continue
            
        # 2. Evaluate OOD
        print(f"\n[{name}] Step 2: Evaluating OOD performance (Generalization)...")
        ckpt_path = f"{out_dir}/ckpt.pt"
        
        from eval_loop import evaluate_ood
        eval_res = evaluate_ood(ckpt_path, num_samples=200)
        
        if eval_res:
            print(f"✅ [{name}] Results: Accuracy {eval_res['accuracy']*100:.2f}%, Avg Steps: {eval_res['avg_steps']:.2f}")
        
        # 3. Save combined result
        res = {
            "experiment": name,
            "config": formatted_args,
            "ood_metrics": eval_res
        }
        results.append(res)
        
        print(f"[{name}] Experiment completed and metrics saved.")
        
        with open(f"looplm/experiments/summary.json", "w") as f:
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
