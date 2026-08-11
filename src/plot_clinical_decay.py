import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

def plot_clinical_decay(csv_path, output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "data", "results")

    df = pd.read_csv(csv_path)

    p_col = "Pruning_Level" if "Pruning_Level" in df.columns else "pruning_level"
    x_vals = df[p_col] * 100

    sns.set_theme(style="whitegrid", palette="muted")
    fig, ax = plt.subplots(figsize=(10, 6))

    metrics = [
        ("Correct", "Correct Retrieval", "o", "forestgreen"),
        ("Coordinate Error", "Coordinate Error", "s", "darkorange"),
        ("Superordinate Error", "Superordinate Error", "^", "firebrick"),
        ("Domain Error", "Domain Error", "x", "purple"),
        ("Domain Collapse", "Domain Collapse", "d", "darkred"),
    ]

    for col, label, marker, color in metrics:
        if col in df.columns:
            ax.plot(x_vals, df[col], marker=marker, label=label, linewidth=2.5, color=color)

    sim_type = "Text Hub" if "text" in csv_path.lower() else "Joint Space"
    ax.set_title(f"Simulated Semantic Dementia: {sim_type}\nHierarchical Conceptual Breakdown", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Network Pruning Intensity (% Zeroed)", fontsize=12)
    ax.set_ylabel("Error / Correct Count", fontsize=12)
    ax.legend(title="Response Type", loc="best", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.7)

    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.basename(csv_path).replace(".csv", "_clinical_decay.png")
    out_path = os.path.join(output_dir, filename)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Clinical decay plot saved to: {out_path}")

if __name__ == "__main__":
    csv_file = os.path.join(PROJECT_ROOT, "data", "results", "joint_space_metrics_full.csv")
    if os.path.exists(csv_file):
        plot_clinical_decay(csv_file)
    else:
        print(f"[!] Target file not found: {csv_file}")