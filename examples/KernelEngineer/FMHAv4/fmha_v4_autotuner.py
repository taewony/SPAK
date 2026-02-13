import subprocess
import json
import re

def run_bench(tm, tn, klat, vlat):
    cmd = f"python fmha_v4_test.py --tile_m {tm} --tile_n {tn} --klat {klat} --vlat {vlat}"
    try:
        result = subprocess.check_output(cmd, shell=True).decode()
        match = re.search(r"TFLOPS=([\d.]+)", result)
        if match:
            return float(match.group(1))
    except:
        return 0.0
    return 0.0

def main():
    # Engineering Loop Search Space
    tile_ms = [64, 128]
    tile_ns = [64, 128]
    klats = [2, 3]
    vlats = [4, 5]
    
    results = []
    
    print(f"{'Config':<20} | {'TFLOPS':<10}")
    print("-" * 35)
    
    best_tflops = 0
    best_cfg = None

    for tm in tile_ms:
        for tn in tile_ns:
            for kl in klats:
                for vl in vlats:
                    tflops = run_bench(tm, tn, kl, vl)
                    cfg_str = f"{tm}x{tn} L:{kl}/{vl}"
                    print(f"{cfg_str:<20} | {tflops:<10.2f}")
                    
                    results.append({
                        "tm": tm, "tn": tn, "kl": kl, "vl": vl, "tflops": tflops
                    })
                    
                    if tflops > best_tflops:
                        best_tflops = tflops
                        best_cfg = (tm, tn, kl, vl)

    print("-" * 35)
    print(f"Best: {best_cfg} -> {best_tflops:.2f} TFLOPS")
    
    # Prepare Trace for Compound Engineering
    trace = {
        "type": "Insight",
        "category": "optimal",
        "content": f"Best TMA config found on RTX5070: Tile={best_cfg[0]}x{best_cfg[1]}, KLat={best_cfg[2]}, VLat={best_cfg[3]}",
        "metrics": {
            "tflops": best_tflops,
            "tile_m": best_cfg[0],
            "tile_n": best_cfg[1],
            "k_lat": best_cfg[2],
            "v_lat": best_cfg[3]
        },
        "source": "fmha_v4_autotuner.py",
        "confidence": 0.95
    }
    
    # Save to file for easy GitHub syncing
    with open("last_engineering_trace.json", "w") as f:
        json.dump(trace, f, indent=4)
    
    print(f"\n[OK] Results saved to last_engineering_trace.json")
    print(f"__SPAK_TRACE__{json.dumps(trace)}")

if __name__ == "__main__":
    main()
