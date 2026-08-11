import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


def plot_clinical_decay(csv_path, output_dir):
    print(f"[*] Loading simulation data from {csv_path}...")
    df = pd.read_csv(csv_path)

    base_name = os.path.basename(csv_path).replace('.csv', '')

    sns.set_theme(style="whitegrid", palette="muted")
    plt.figure(figsize=(12, 7))

    x_vals = df['Pruning_Level'] * 100

    plt.plot(x_vals, df['Correct'], marker='o', label='Correct Retrieval', linewidth=2.5, color='forestgreen')
    plt.plot(x_vals, df['Coordinate Error'], marker='s', label='Coordinate Error (e.g., Dog -> Cat)', linewidth=2, color='darkorange')
    plt.plot(x_vals, df['Superordinate Error'], marker='^', label='Superordinate Error (e.g., Dog -> Animal)', linewidth=2, color='firebrick')
    plt.plot(x_vals, df['Domain Error'], marker='x', label='Domain Error', linewidth=2, color='purple')
    plt.plot(x_vals, df['Domain Collapse'], marker='d', label='Domain Collapse (e.g., Living -> Non-Living)', linewidth=2, color='black', linestyle='--')

    sim_type = "Aphasia (Text Hub)" if "text" in base_name else "Visual Agnosia (Vision Hub)"
    plt.title(f'Simulated Semantic Dementia: {sim_type}\nHierarchical Conceptual Breakdown via Network Atrophy', fontsize=16, fontweight='bold', pad=15)
    
    plt.xlabel('Network Pruning Intensity (% of Weights Zeroed)', fontsize=14)
    plt.ylabel('Number of Retrievals', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    plt.legend(title="Response Type", title_fontsize='13', fontsize='11', loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)

    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f'{base_name}_curve.png')
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    
    print(f"[+] Plot successfully saved to: {out_file}")
    plt.close()

if __name__ == "__main__":
    results_dir = os.path.join(PROJECT_ROOT, "data", "results")
    
    csv_candidates = [
        os.path.join(results_dir, "amodal_sd_simulation.csv"),
        os.path.join(results_dir, "text_hub_simulation.csv")
    ]
    
    csv_file = next((c for c in csv_candidates if os.path.exists(c)), None)
    
    if csv_file:
        plot_clinical_decay(csv_file, results_dir)
    else:
        print(f"[!] Could not locate simulation CSV file in '{results_dir}'. Please execute the testing harness first.")