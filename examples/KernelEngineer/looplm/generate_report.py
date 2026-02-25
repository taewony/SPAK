import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def main():
    print("📊 [Windows PC] Generating Paper Assets...")
    
    data_file = "paper_evaluation_data.json"
    if not os.path.exists(data_file):
        print(f"❌ Error: Cannot find '{data_file}'. Please transfer it from the RTX 5070 PC.")
        return

    with open(data_file, "r") as f:
        data = json.load(f)

    # Output directory
    out_dir = "paper_assets"
    os.makedirs(out_dir, exist_ok=True)

    # Set Academic Theme
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams['font.family'] = 'serif'
    
    # ---------------------------------------------------------
    # 1. Generate Table 1: LaTeX & Markdown
    # ---------------------------------------------------------
    table_lines = []
    table_lines.append(r"\begin{table}[h]")
    table_lines.append(r"\centering")
    table_lines.append(r"\caption{Length Generalization Accuracy (\%) and Inference Steps on Addition Task}")
    table_lines.append(r"\label{tab:main_results}")
    table_lines.append(r"\resizebox{\columnwidth}{!}{%")
    table_lines.append(r"\begin{tabular}{l|ccccc|c}")
    table_lines.append(r"\toprule")
    table_lines.append(r"\textbf{Model Architecture} & \textbf{1-4d (ID)} & \textbf{5-6d (OOD)} & \textbf{8d (OOD)} & \textbf{10d (OOD)} & \textbf{12d (OOD)} & \textbf{Avg Steps} \\")
    table_lines.append(r"\midrule")
    
    for name, metrics in data.items():
        if "Test-Time" in name:
            table_lines.append(r"\midrule") # Separate Test-Time compute row
            
        line = f"{name} & {metrics['accuracy_1_4d']} & {metrics['accuracy_5_6d']} & {metrics['accuracy_8d']} & {metrics['accuracy_10d']} & {metrics['accuracy_12d']} & {metrics['avg_steps']} \\\\"
        table_lines.append(line)
        
    table_lines.append(r"\bottomrule")
    table_lines.append(r"\end{tabular}%")
    table_lines.append(r"}")
    table_lines.append(r"\end{table}")

    with open(os.path.join(out_dir, "table_main_results.tex"), "w") as f:
        f.write("\n".join(table_lines))

    # ---------------------------------------------------------
    # 2. Plot 1: Length Generalization Curve
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    x_labels = ["1-4d", "5-6d", "8d", "10d", "12d"]
    x_pos = range(len(x_labels))
    
    colors = {
        "GPT-12L (Static)": "#333333",          # Dark Gray
        "LoopLM-12 (Dynamic)": "#1f77b4",       # Blue
        "LoopLM-30 (Deep Thinking)": "#d62728", # Red
        "LoopLM-128e (Efficient)": "#2ca02c",   # Green
        "LoopLM-12 (Test-Time 24)": "#9467bd"   # Purple
    }
    
    markers = ["o", "s", "^", "D", "v"]
    
    for i, (name, metrics) in enumerate(data.items()):
        y_vals = [
            metrics["accuracy_1_4d"], metrics["accuracy_5_6d"], 
            metrics["accuracy_8d"], metrics["accuracy_10d"], metrics["accuracy_12d"]
        ]
        ls = '--' if 'Test-Time' in name else '-'
        ax.plot(x_pos, y_vals, label=name, color=colors.get(name, "black"), 
                marker=markers[i%len(markers)], linestyle=ls, linewidth=2.5, markersize=8)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Accuracy (%)", fontweight='bold')
    ax.set_xlabel("Number of Digits (Operand Length)", fontweight='bold')
    ax.set_title("OOD Length Generalization: LoopLM vs. Standard GPT", fontweight='bold', pad=15)
    ax.legend(title="Model", frameon=True, shadow=True)
    ax.set_ylim(-5, 105)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig1_generalization_curve.pdf"), dpi=300)
    plt.savefig(os.path.join(out_dir, "fig1_generalization_curve.png"), dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # 3. Plot 2: Test-Time Compute Impact (Bar Chart)
    # ---------------------------------------------------------
    # Only compare 12L Static, Loop-12, and Loop-12(Test-Time 24) on 5-6d OOD
    tt_models = ["GPT-12L (Static)", "LoopLM-12 (Dynamic)", "LoopLM-12 (Test-Time 24)"]
    if all(m in data for m in tt_models):
        fig, ax = plt.subplots(figsize=(7, 6))
        
        y_vals = [data[m]["accuracy_5_6d"] for m in tt_models]
        x_vals = ["Standard\n(12 Layers)", "LoopLM\n(12 Loops)", "LoopLM\n(Zero-Shot 24 Loops)"]
        bar_colors = ["#7f7f7f", "#1f77b4", "#9467bd"]
        
        bars = ax.bar(x_vals, y_vals, color=bar_colors, edgecolor='black', linewidth=1.5, width=0.6)
        
        ax.set_ylabel("OOD Accuracy on 5-6 Digits (%)", fontweight='bold')
        ax.set_title("The Power of Inference-Only Test-Time Compute", fontweight='bold', pad=15)
        ax.set_ylim(0, 110)
        
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval + 2, f"{yval:.1f}%", ha='center', va='bottom', fontweight='bold', fontsize=12)
            
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "fig2_test_time_compute.pdf"), dpi=300)
        plt.savefig(os.path.join(out_dir, "fig2_test_time_compute.png"), dpi=300)
        plt.close()

    print(f"🎉 Success! Paper assets (PDFs, PNGs, and TeX) generated in '{out_dir}/'")

if __name__ == "__main__":
    main()