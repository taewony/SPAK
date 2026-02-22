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
    # ==========================================================
    # 🔍 SMOKE_TEST: True면 1회만 학습하고 2개 샘플만 평가하여 
    # 로직(파일 저장, 경로 등)이 정상인지 1분 내에 검증합니다.
    # ==========================================================
    SMOKE_TEST = False
    # ==========================================================

    basic_experiments = [
        {"name": "baseline", "args": "n_embd=384 num_loops=12 dropout=0.2"},
        {"name": "A1_low_cap", "args": "n_embd=256 n_head=4 num_loops=12"},
        {"name": "A2_very_low_cap", "args": "n_embd=128 n_head=4 num_loops=12"},
        {"name": "A3_high_dropout", "args": "n_embd=384 num_loops=12 dropout=0.4"},
        {"name": "T1_deep_thinking", "args": "n_embd=256 n_head=4 num_loops=24"},
    ]

    advanced_experiments_p3 = [
        # G1: Long training to find Grokking point
        {"name": "G1_grokking_long", "args": "n_embd=256 n_head=4 num_loops=16 dropout=0.3 max_iters=10000"},
        # W2: Stiff Thinking (Wait for 99.9% confidence)
        {"name": "W2_stiff_thinking", "args": "n_embd=256 n_head=4 num_loops=24 dropout=0.2 max_iters=5000"},
        # T2: Deep & Narrow (Force logic reuse with 32 loops)
        {"name": "T2_deep_narrow", "args": "n_embd=192 n_head=3 num_loops=32 dropout=0.2 max_iters=5000"},
        # A4: Robust Regularization (High Dropout to kill memorization)
        {"name": "A4_robust_reg", "args": "n_embd=256 n_head=4 num_loops=16 dropout=0.5 max_iters=7000"},
    ]

    # --- ⚡ Phase 4: Blackwell Persistent & Architecture Ablation ---
    experiments = [
        # P4_X0_Baseline: Standard injection (Current best)
        {"name": "P4_X0_Baseline", "args": "n_embd=256 n_head=4 num_loops=16 inject_x0=True max_iters=5000"},
        # P4_Pure_Dynamics: Remove X0 injection (Pure recurrent state)
        {"name": "P4_Pure_Dynamics", "args": "n_embd=256 n_head=4 num_loops=16 inject_x0=False max_iters=5000"},
        # P4_Deep_Grok: Combining best of Phase 3 with Phase 4
        {"name": "P4_Deep_Grok", "args": "n_embd=256 n_head=4 num_loops=24 inject_x0=True max_iters=10000 dropout=0.3"},
    ]
    # ---------------------------------------------------------------

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
        
        # Override max_iters if present in exp["args"]
        current_max_iters = 5 if SMOKE_TEST else 2000
        if "max_iters=" in args_str:
            for part in args_str.split():
                if "max_iters=" in part:
                    current_max_iters = int(part.split('=')[1])
                    if SMOKE_TEST: current_max_iters = 5

        ood_samples = 2 if SMOKE_TEST else 200
        out_dir_rel = f"experiments/{name}"
        
        # Determine the correct path to train_loop.py
        train_script = os.path.join(script_dir, "train_loop.py")
        
        print("\n" + "="*60)
        print(f"🚀 STARTING ADVANCED EXPERIMENT: {name}")
        print(f"   Config: {formatted_args}")
        print(f"   Output: {out_dir_rel}")
        print("="*60)
        
        # 1. Train
        print(f"[{name}] Step 1: Training for {current_max_iters} iterations...")
        train_cmd = f"python {train_script} {formatted_args} --out_dir={out_dir_rel} --max_iters={current_max_iters}"
        ret = run_command(train_cmd)

        if ret != 0:
            print(f"❌ [{name}] Experiment failed during training phase.")
            continue
            
        # 2. Evaluate OOD
        print(f"\n[{name}] Step 2: Evaluating OOD performance (Generalization)...")
        ckpt_path = os.path.join(script_dir, out_dir_rel, "ckpt.pt")
        
        eval_res = evaluate_ood(ckpt_path, num_samples=ood_samples)
        
        if eval_res:
            print(f"✅ [{name}] Results: Accuracy {eval_res['accuracy']*100:.2f}%, Avg Steps: {eval_res['avg_steps']:.2f}")
        else:
            eval_res = None
        
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
