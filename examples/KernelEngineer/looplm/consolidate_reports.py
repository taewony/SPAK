import os
import subprocess
import shutil
from datetime import datetime

def consolidate():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    exp_dir = os.path.join(script_dir, "experiments")
    archive_dir = os.path.join(exp_dir, "archive")
    
    print("="*60)
    print("🧹 Phase 7: Result Consolidation & Archiving")
    print("="*60)

    # 1. Archive old json files
    if not os.path.exists(archive_dir):
        os.makedirs(archive_dir)
    
    for f in os.listdir(exp_dir):
        if f.startswith("summary") and f.endswith(".json"):
            shutil.move(os.path.join(exp_dir, f), os.path.join(archive_dir, f))
            print(f"Archived: {f}")

    # 2. Run Authoritative Evaluation (re_evaluate_all.py)
    # This script re-calculates metrics from ckpt.pt using fixed logic
    print("
🚀 Running Fresh Evaluation for all Valid Experiments...")
    re_eval_script = os.path.join(script_dir, "re_evaluate_all.py")
    
    try:
        subprocess.run(["python", re_eval_script], check=True)
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return

    # 3. Finalize Report Name with Timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    src_report = os.path.join(script_dir, "INTELLIGENCE_REPORT.md")
    dst_report = os.path.join(script_dir, f"MASTER_REPORT_v{timestamp}.md")
    
    if os.path.exists(src_report):
        shutil.copy(src_report, dst_report)
        # Also keep a copy as the latest standard
        shutil.copy(src_report, os.path.join(script_dir, "FINAL_MASTER_REPORT.md"))
        print(f"✅ SUCCESS!")
        print(f"   - Latest Report: looplm/FINAL_MASTER_REPORT.md")
        print(f"   - Versioned Backup: looplm/MASTER_REPORT_v{timestamp}.md")
    else:
        print("❌ Error: INTELLIGENCE_REPORT.md was not generated.")

if __name__ == "__main__":
    consolidate()
