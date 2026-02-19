import json
import matplotlib.pyplot as plt
import os

def load_trace(file_path):
    if not os.path.exists(file_path):
        return None
    with open(file_path, 'r') as f:
        data = json.load(f)
    convergence = [item for item in data if item.get('type') == 'Convergence']
    return {
        'steps': [item['step'] for item in convergence],
        'losses': [item['loss'] for item in convergence],
        'times': [item['step_time_ms'] for item in convergence]
    }

def plot_comparison():
    baseline = load_trace('microgpt_baseline_trace.json')
    cutile = load_trace('microgpt_train_trace.json')

    if not cutile:
        print("Error: cuTile trace not found.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # 1. Loss Comparison
    ax1.plot(cutile['steps'], cutile['losses'], label='SPAK (cuTile)', color='tab:red', alpha=0.7)
    if baseline:
        ax1.plot(baseline['steps'], baseline['losses'], label='Baseline (Scalar Python)', color='black', linestyle='--', alpha=0.5)
    
    ax1.set_title('Convergence Comparison (Loss)')
    ax1.set_ylabel('Cross Entropy Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Performance Comparison (Log scale for speedup visibility)
    ax2.plot(cutile['steps'], cutile['times'], label='SPAK (cuTile)', color='tab:blue')
    if baseline:
        ax2.plot(baseline['steps'], baseline['times'], label='Baseline (CPU)', color='gray', alpha=0.5)
    
    ax2.set_yscale('log')
    ax2.set_title('Performance Comparison (Step Time in ms - Log Scale)')
    ax2.set_ylabel('Time (ms)')
    ax2.set_xlabel('Training Step')
    ax2.legend()
    ax2.grid(True, which="both", ls="-", alpha=0.2)

    plt.tight_layout()
    plt.savefig('comparison_results.png')
    print("Graph saved to comparison_results.png")

    if baseline:
        avg_base = sum(baseline['times']) / len(baseline['times'])
        avg_cu = sum(cutile['times'][5:]) / len(cutile['times'][5:]) # ignore warmup
        print(f"Average Baseline Step: {avg_base:.2f}ms")
        print(f"Average cuTile Step: {avg_cu:.2f}ms")
        print(f"Speedup: {avg_base/avg_cu:.1f}x")

if __name__ == "__main__":
    plot_comparison()
