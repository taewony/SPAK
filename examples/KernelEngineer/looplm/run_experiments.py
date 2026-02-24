import subprocess
import os
import sys
import time
import json

# Ensure looplm is in path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

# Try importing evaluation function
try:
    # 가정: eval_loop.py가 같은 디렉토리에 있다고 가정
    from eval_loop import evaluate_ood
except ImportError:
    print("Warning: Could not import evaluate_ood. Skipping auto-evaluation.")
    def evaluate_ood(*args, **kwargs): return {}

def run_command(cmd):
    print(f"\n[CMD] {cmd}")
    # 실시간 로그 출력을 위해 Popen 사용
    process = subprocess.Popen(cmd, shell=True)
    process.wait()
    return process.returncode

def main():
    # ==========================================================
    # 🔍 SMOKE_TEST: True면 100 step만 돌려서 에러 없는지만 확인
    # ==========================================================
    SMOKE_TEST = False 
    # ==========================================================

    # 실험 목록: [이름, 스크립트파일, 인자들]
    experiments = [
        # ... (이전 실험들 생략 가능, Exp7 위주로 구성) ...
        {
            "name": "Exp7_Grokking_Marathon",
            "script": "train_loop.py",
            "args": (
                "--dataset=addition_reverse "
                "--num_loops=12 --n_embd=256 --n_head=4 "
                "--inject_x0=False "
                "--max_iters=100000 "
                "--batch_size=128 "
                "--learning_rate=5e-4 "
                "--weight_decay=0.2 "
                "--dropout=0.2"
            )
        }
    ]

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results = []

    print(f"🚀 Starting {len(experiments)} Experiments for RoPE & Batching Validation...")

    for i, exp in enumerate(experiments):
        name = exp["name"]
        script_name = exp["script"]
        args_str = exp["args"]
        
        # Smoke Test Override
        if SMOKE_TEST:
            args_str += " --max_iters=100 --eval_interval=50"
            out_dir = f"experiments/smoke_{name}"
        else:
            out_dir = f"experiments/{name}"

        print(f"\n{'='*60}")
        print(f"▶️  Running [{i+1}/{len(experiments)}]: {name}")
        print(f"    Script: {script_name}")
        print(f"    Output: {out_dir}")
        print(f"{'='*60}")

        # 실행 커맨드 조립
        full_cmd = f"python {script_name} {args_str} --out_dir={out_dir}"
        
        # 1. 학습 실행
        start_time = time.time()
        ret_code = run_command(full_cmd)
        duration = time.time() - start_time

        if ret_code != 0:
            print(f"❌ Experiment {name} failed with return code {ret_code}")
            continue

        # 2. 결과 기록 (로그 파일 파싱 대신 간단히 성공 여부만)
        results.append({
            "name": name,
            "status": "Success",
            "duration_sec": round(duration, 2),
            "out_dir": out_dir
        })

    # 최종 요약 저장
    summary_path = os.path.join(script_dir, "experiments", f"summary_{timestamp}.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\n✅ All experiments finished. Check results in {summary_path}")

if __name__ == "__main__":
    main()