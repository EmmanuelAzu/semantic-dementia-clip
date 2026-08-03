import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root is on Python path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

from src.testing_harness import TestingHarness
from src.generate_tsne import generate_tsne_grid

def plot_decay_curves(df_results: pd.DataFrame, title: str, output_path: str):
    """Generates the hierarchical decay curve plot for a simulation run."""
    plt.figure(figsize=(12, 7))
    sns.set_theme(style="whitegrid")
    
    # Identify error count columns
    error_cols = ["Correct", "Coordinate Error", "Superordinate Error", "Domain Error", "Domain Collapse"]
    available_cols = [c for c in error_cols if c in df_results.columns]
    
    markers = ['o', 's', '^', 'X', 'D']
    colors = ['#2ca02c', '#ff7f0e', '#d62728', '#9467bd', '#000000']
    
    for col, marker, color in zip(available_cols, markers, colors):
        plt.plot(
            df_results['Pruning_Level'] * 100, 
            df_results[col], 
            marker=marker, 
            color=color,
            linewidth=2.5, 
            markersize=7, 
            label=col
        )
        
    plt.title(title, fontsize=15, fontweight='bold', pad=15)
    plt.xlabel("Network Pruning Intensity (% of Hub Weights Zeroed)", fontsize=12)
    plt.ylabel("Number of Retrievals", fontsize=12)
    plt.legend(title="Response Type", frameon=True, facecolor="white", loc="center left", bbox_to_anchor=(1.01, 0.5))
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Decay curve plot saved to: {output_path}")

def run_master_pipeline():
    """Executes full clinical simulation and renders all quantitative & visual plots."""
    print("============================================================")
    print(" RUNNING SEMANTIC DEMENTIA SIMULATION PIPELINE ")
    print("============================================================")
    
    metadata_path = os.path.join(PROJECT_ROOT, "data", "processed", "metadata_processed.csv")
    index_tensor_path = os.path.join(PROJECT_ROOT, "data", "processed", "image_index.pt")
    results_dir = os.path.join(PROJECT_ROOT, "data", "results")
    
    # 1. Initialize Testing Harness
    harness = TestingHarness(
        metadata_path=metadata_path,
        index_tensor_path=index_tensor_path
    )
    
    # ------------------------------------------------------------
    # RUN 1: TEXT HUB ATROPHY (Progressive Aphasia)
    # ------------------------------------------------------------
    print("\n[PHASE 1/4] Running Text Hub Atrophy Simulation...")
    df_text = harness.run_simulation(encoder_type="text", step=0.1)
    df_text.to_csv(os.path.join(results_dir, "aphasia_simulation.csv"), index=False)
    
    print("\n[PHASE 2/4] Generating Text Hub Error Decay Plot & t-SNE Grid...")
    plot_decay_curves(
        df_text, 
        title="Simulated Semantic Dementia: Progressive Fluent Aphasia (Text Hub)", 
        output_path=os.path.join(results_dir, "text_hub_decay_curve.png")
    )
    generate_tsne_grid(
        metadata_path=metadata_path,
        output_plot_path=os.path.join(results_dir, "text_hub_tsne_grid.png"),
        encoder_type="text"
    )
    
    # ------------------------------------------------------------
    # RUN 2: VISION HUB ATROPHY (Visual Agnosia)
    # ------------------------------------------------------------
    print("\n[PHASE 3/4] Running Vision Hub Atrophy Simulation...")
    df_vision = harness.run_simulation(encoder_type="vision", step=0.1)
    df_vision.to_csv(os.path.join(results_dir, "agnosia_simulation.csv"), index=False)
    
    print("\n[PHASE 4/4] Generating Vision Hub Error Decay Plot & t-SNE Grid...")
    plot_decay_curves(
        df_vision, 
        title="Simulated Semantic Dementia: Visual Agnosia (Vision Hub)", 
        output_path=os.path.join(results_dir, "vision_hub_decay_curve.png")
    )
    generate_tsne_grid(
        metadata_path=metadata_path,
        output_plot_path=os.path.join(results_dir, "vision_hub_tsne_grid.png"),
        encoder_type="vision"
    )
    
    print("\n============================================================")
    print(" [✓] PIPELINE COMPLETE! ALL EXPERIMENTS AND PLOTS GENERATED ")
    print("============================================================")
    print(f"Results saved in: {results_dir}")
    print("  - aphasia_simulation.csv & agnosia_simulation.csv")
    print("  - text_hub_decay_curve.png & vision_hub_decay_curve.png")
    print("  - text_hub_tsne_grid.png & vision_hub_tsne_grid.png")

if __name__ == "__main__":
    run_master_pipeline()