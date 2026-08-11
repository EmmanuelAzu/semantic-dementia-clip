import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

def plot_representation_results(csv_path, output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "data", "results")

    df = pd.read_csv(csv_path)

    p_col = "Pruning_Level" if "Pruning_Level" in df.columns else "pruning_level"
    x_vals = df[p_col] * 100

    sns.set_theme(style="whitegrid", palette="deep")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: Retrieval Accuracy & Alignment
    ax1 = axes[0]
    if "i2t_top1" in df.columns:
        ax1.plot(x_vals, df["i2t_top1"], marker="o", label="Image-to-Text Top-1 Acc", linewidth=2.5, color="#1f77b4")
    elif "top1_specific_acc" in df.columns:
        ax1.plot(x_vals, df["top1_specific_acc"], marker="o", label="Specific Top-1 Acc", linewidth=2.5, color="#1f77b4")
    
    if "mrr" in df.columns:
        ax1.plot(x_vals, df["mrr"], marker="s", label="Mean Reciprocal Rank (MRR)", linewidth=2, color="#ff7f0e")

    ax1.set_title("Retrieval & Accuracy Metrics", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Pruning Intensity (%)", fontsize=11)
    ax1.set_ylabel("Score (0.0 - 1.0)", fontsize=11)
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(loc="best")
    ax1.grid(True, linestyle="--", alpha=0.7)

    # Panel 2: Vector Space Preservation & Drift
    ax2 = axes[1]
    if "cka_vision" in df.columns:
        ax2.plot(x_vals, df["cka_vision"], marker="^", label="Vision CKA", linewidth=2, color="#2ca02c")
    if "cka_text" in df.columns:
        ax2.plot(x_vals, df["cka_text"], marker="v", label="Text CKA", linewidth=2, color="#d62728")
    if "npr_vision" in df.columns:
        ax2.plot(x_vals, df["npr_vision"], marker="d", label="Vision NPR (k=5)", linewidth=2, color="#9467bd")

    ax2.set_title("Representation & Topology Drift", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Pruning Intensity (%)", fontsize=11)
    ax2.set_ylabel("Similarity / Preservation Index", fontsize=11)
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(loc="best")
    ax2.grid(True, linestyle="--", alpha=0.7)

    plt.suptitle("Joint Space Model Degradation & Representation Analysis", fontsize=15, fontweight="bold", y=0.98)

    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.basename(csv_path).replace(".csv", "_representation_metrics.png")
    out_path = os.path.join(output_dir, filename)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Representation results plot saved to: {out_path}")

if __name__ == "__main__":
    csv_file = os.path.join(PROJECT_ROOT, "data", "results", "joint_space_metrics_full.csv")
    if os.path.exists(csv_file):
        plot_representation_results(csv_file)
    else:
        print(f"[!] Target file not found: {csv_file}")