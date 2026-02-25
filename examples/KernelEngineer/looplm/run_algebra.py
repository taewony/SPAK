import subprocess
import os
import time
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

def run_command(cmd):
    print(f"\n[{time.strftime('%H:%M:%S')}] 🚀 실행 중: {cmd}")
    process = subprocess.Popen(cmd, shell=True)
    process.wait()
    return process.returncode

def main():
    print("="*60)
    print("🧠 Phase 8: Algebraic Equation Grokking (50k Iters)")
    print("="*60)

    experiments = [
        {
            "name": "Exp8_Algebra_GPT",
            "script": "train_baseline_12l.py",
            "args": (
                "--dataset=algebra_reverse "
                "--n_layer=12 --n_embd=256 --n_head=4 "
                "--max_iters=50000 --batch_size=128 "
                "--weight_decay=1e-3 --dropout=0.0"
            )
        },
        {
            "name": "Exp8_Algebra_Loop30",
            "script": "train_loop.py",
            "args": (
                "--dataset=algebra_reverse "
                "--num_loops=30 --n_embd=256 --n_head=4 "
                "--max_iters=50000 --batch_size=128 "
                "--weight_decay=1e-3 --dropout=0.0"
            )
        }
    ]

    for i, exp in enumerate(experiments):
        out_dir = f"experiments/{exp['name']}"
        full_cmd = f"python {exp['script']} {exp['args']} --out_dir={out_dir}"
        
        start_time = time.time()
        ret_code = run_command(full_cmd)
        duration = time.time() - start_time
        
        if ret_code == 0:
            print(f"✅ {exp['name']} 완료! (소요 시간: {duration/3600:.2f}시간)")
        else:
            print(f"❌ {exp['name']} 실패 (Return Code: {ret_code})")
            break # 첫 번째가 실패하면 멈춤

    print("\n🏁 모든 Phase 8 훈련이 종료되었습니다.")

if __name__ == "__main__":
    main()