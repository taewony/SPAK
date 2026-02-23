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

    # --- ⚔️ THE FINAL MATCH LIST ---
    # Unified Max Iters = 15,000 for all experiments to ensure absolute fairness.
    experiments = [
        # --- ⚖️ Fair Battle Group (Depth 12) ---
        {"name": "B1_Static_Normal", "args": "dataset=addition n_layer=12 n_embd=256 n_head=4 max_iters=15000"},
        {"name": "B2_Static_Reverse", "args": "dataset=addition_reverse n_layer=12 n_embd=256 n_head=4 max_iters=15000"},
        {"name": "L1_Dynamic_Normal", "args": "dataset=addition num_loops=12 n_embd=256 n_head=4 max_iters=15000"},
        {"name": "L2_Dynamic_Reverse", "args": "dataset=addition_reverse num_loops=12 n_embd=256 n_head=4 max_iters=15000"},
        
        # --- 🔄 Reverse Advanced Group (R1-R4) ---
        {"name": "R1_Reverse_Baseline", "args": "dataset=addition_reverse n_embd=256 n_head=4 num_loops=16 max_iters=15000"},
        {"name": "R2_Reverse_Grok", "args": "dataset=addition_reverse n_embd=256 n_head=4 num_loops=24 max_iters=15000 dropout=0.2"},
        {"name": "R3_Reverse_Efficient", "args": "dataset=addition_reverse n_embd=128 n_head=4 num_loops=32 max_iters=15000"},
        {"name": "R4_Reverse_Deep_Thinking", "args": "dataset=addition_reverse n_embd=256 n_head=4 num_loops=48 max_iters=15000 dropout=0.2"},
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
        
        # Override max_iters if present in exp["args"]
        current_max_iters = 5 if SMOKE_TEST else 15000
        if "max_iters=" in args_str:
            for part in args_str.split():
                if "max_iters=" in part:
                    current_max_iters = int(part.split('=')[1])
                    if SMOKE_TEST: current_max_iters = 5

        ood_samples = 2 if SMOKE_TEST else 200
        
        # Determine script type
        train_script = "train_baseline_12l.py" if "n_layer=12" in args_str else "train_loop.py"
        train_path = os.path.join(script_dir, train_script)
        
        print("\n" + "="*60)
        print(f"🚀 STARTING MATCH: {name}")
        print(f"   Using Script: {train_script}")
        print(f"   Config: {formatted_args}")
        print("="*60)
        
        # 1. Train
        ret = run_command(f"python {train_path} {formatted_args} --out_dir={out_dir_rel} --max_iters={current_max_iters}")
        if ret != 0: continue
            
        # 2. Evaluate
        ckpt_path = os.path.join(script_dir, out_dir_rel, "ckpt.pt")
        eval_res = evaluate_ood(ckpt_path, num_samples=ood_samples)
        
        # 3. Save
        res = {"experiment": name, "config": formatted_args, "ood_metrics": eval_res}
        results.append(res)
        with open(summary_path, "w") as f: json.dump(results, f, indent=4)
        latest_path = os.path.join(script_dir, "experiments", "summary_latest.json")
        with open(latest_path, "w") as f: json.dump(results, f, indent=4)

    print("\n🏁 FINAL FAIR BATTLE COMPLETE. RUN generate_master_report.py NOW!")

if __name__ == "__main__":
    main()
