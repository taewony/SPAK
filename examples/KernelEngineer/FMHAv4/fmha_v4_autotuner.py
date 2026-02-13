import subprocess
import json
import re

def run_bench(tm, tn, klat, vlat, causal):
    cmd = f"python fmha_v4_test.py --tile_m {tm} --tile_n {tn} --klat {klat} --vlat {vlat} --causal {causal}"
    try:
        output = subprocess.check_output(cmd, shell=True).decode()
        tflops_match = re.search(r"TFLOPS=([\d.]+)", output)
        device_match = re.search(r"DEVICE=(.+)", output)
        
        tflops = float(tflops_match.group(1)) if tflops_match else 0.0
        device = device_match.group(1).strip() if device_match else "Unknown"
        return tflops, device
    except:
        return 0.0, "Error"

def main():
    # Engineering Loop Search Space
    tile_ms = [64, 128]
    tile_ns = [64, 128]
    klats = [2, 3]
    vlats = [4, 5]
    causal_options = [1] # Focusing on causal as requested
    
    history = []
    
    print(f"{'Config':<25} | {'TFLOPS':<10} | {'Device'}")
    print("-" * 55)
    
    best_tflops = 0
    best_cfg = None
    target_device = "Unknown"

    for causal in causal_options:
        for tm in tile_ms:
            for tn in tile_ns:
                for kl in klats:
                    for vl in vlats:
                        tflops, device = run_bench(tm, tn, kl, vl, causal)
                        target_device = device
                        cfg_str = f"{tm}x{tn} L:{kl}/{vl} C:{causal}"
                        print(f"{cfg_str:<25} | {tflops:<10.2f} | {device}")
                        
                        history.append({
                            "tile_m": tm, 
                            "tile_n": tn, 
                            "k_lat": kl, 
                            "v_lat": vl, 
                            "causal": causal,
                            "tflops": tflops,
                            "device": device
                        })
                        
                        if tflops > best_tflops:
                            best_tflops = tflops
                            best_cfg = history[-1]

    print("-" * 55)
    if best_cfg:
        print(f"Best: {best_cfg['tile_m']}x{best_cfg['tile_n']} L:{best_cfg['k_lat']}/{best_cfg['v_lat']} -> {best_tflops:.2f} TFLOPS")
    
    # Trace for Compound Engineering (Full history for negative result analysis)
    trace = {
        "type": "EngineeringSweep",
        "device": target_device,
        "best_config": best_cfg,
        "full_history": history,
        "source": "fmha_v4_autotuner.py",
        "timestamp": "2026-02-13"
    }
    
    with open("last_engineering_trace.json", "w") as f:
        json.dump(trace, f, indent=4)
    
    print(f"\n[OK] Full sweep results saved to last_engineering_trace.json")
    # For CLI capture
    print(f"__SPAK_TRACE__{json.dumps({'type': 'Insight', 'category': 'hardware_sweep', 'device': target_device, 'best_tflops': best_tflops})}")

if __name__ == "__main__":
    main()
