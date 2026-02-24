import subprocess
import os
import sys
import time
import json

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

try:
    from eval_loop import evaluate_ood
except ImportError:
    print("Warning: Could not import evaluate_ood. Skipping auto-evaluation.")
    def evaluate_ood(*args, **kwargs): return {}

def run_command(cmd):
    print(f"\n[CMD] {cmd}")
    process = subprocess.Popen(cmd, shell=True)
    process.wait()
    return process.returncode

def main():
    # ==========================================================
    # Phase 5: Grokking, Parameter Efficiency, and Deep Thinking
    # ==========================================================
    experiments = [
        # 1. Parameter Efficiency Test (Narrow & Deep Time)
        # 파라미터는 최소화(embd=128)하고, 루프를 24회로 늘려 '사고의 시간'만으로 문제를 푸는지 테스트
        {
            "name": "Exp4_LoopLM_Narrow_Deep_Time",
            "script": "train_loop.py",
            "args": (
                "--dataset=addition_reverse "
                "--num_loops=24 --n_embd=128 --n_head=4 " # Model capacity reduced by 4x
                "--inject_x0=False "
                "--max_iters=15000 --batch_size=128 "
                "--weight_decay=1e-4"
            )
        },
        # 2. Forced Grokking Test (High Regularization)
        # 암기를 방지하기 위해 강한 규제 적용. Loss가 늦게 떨어지더라도 OOD가 튀어오르는지 확인
        {
            "name": "Exp5_LoopLM_Forced_Grokking",
            "script": "train_loop.py",
            "args": (
                "--dataset=addition_reverse "
                "--num_loops=12 --n_embd=256 --n_head=4 "
                "--inject_x0=False "
                "--max_iters=20000 --batch_size=128 " # 더 오래 학습
                "--weight_decay=1e-1 --dropout=0.2"   # <-- [핵심] 암기 방지, 규칙 유도
            )
        },
        # 3. Standard GPT Small (Exp4와의 공정한 비교군)
        # Exp4와 파라미터 수가 비슷한 표준 GPT를 학습시켜, 파라미터 낭비를 증명
        {
            "name": "Exp6_Baseline_Small",
            "script": "train_baseline_12l.py",
            "args": (
                "--dataset=addition_reverse "
                "--n_layer=1 --n_embd=256 --n_head=4 " # 1 Layer Only
                "--max_iters=15000 --batch_size=128 "
                "--weight_decay=1e-4"
            )
        }
    ]

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results = []

    print(f"🚀 Starting Phase 5: Grokking & Efficiency ({len(experiments)} Experiments)")

    for i, exp in enumerate(experiments):
        name = exp["name"]
        script_name = exp["script"]
        args_str = exp["args"]
        out_dir = f"experiments/{name}"

        print(f"\n{'='*60}")
        print(f"▶️  Running [{i+1}/{len(experiments)}]: {name}")
        print(f"    Config: {args_str}")
        print(f"{'='*60}")

        full_cmd = f"python {script_name} {args_str} --out_dir={out_dir}"
        
        # 1. Train
        start_time = time.time()
        ret_code = run_command(full_cmd)
        duration = time.time() - start_time

        if ret_code != 0:
            print(f"❌ {name} failed.")
            continue

        # 2. Auto-Evaluate OOD
        print(f"🔍 Automatically evaluating OOD for {name}...")
        ckpt_path = os.path.join(script_dir, out_dir, "ckpt.pt")
        eval_metrics = evaluate_ood(ckpt_path, num_samples=200) # 200개 샘플 평가

        results.append({
            "name": name,
            "duration_sec": round(duration, 2),
            "ood_metrics": eval_metrics
        })

    # Save Results
    summary_path = os.path.join(script_dir, "experiments", f"summary_phase5_{timestamp}.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=4)
    
    # Update latest pointer for Master Report
    latest_path = os.path.join(script_dir, "experiments", "summary_latest.json")
    with open(latest_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\n✅ Phase 5 Complete! Summary saved to {summary_path}")

if __name__ == "__main__":
    main()