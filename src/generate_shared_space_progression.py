"""
generate_shared_space_progression.py
------------------------------------
Visualizes shared multimodal space dissolution under structured hub pruning
(Images [•] vs Text Concept Vectors [★]) across 0%, 30%, 60%, and 90% atrophy.
"""

import os
import sys
import copy
import torch
import clip
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.spatial import procrustes
from sklearn.manifold import TSNE
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.pruning_engine import CLIPPruningEngine

CATEGORY_COLORS = {
    "Mammal": "#d73027",             # Crimson Red
    "Non-Mammal": "#4575b4",         # Deep Navy Blue
    "Flora & Food": "#2ca02c",       # Forest Green
    "Land Vehicle": "#ff7f0e",       # Bright Orange
    "Non-Land Vehicle": "#17becf",   # Sky Blue
    "Home & Furniture": "#762a83"    # Deep Purple
}

def extract_joint_embeddings(model, metadata, preprocess, device):
    """Extracts unit-normalized image features AND matching specific concept text vectors."""
    model.eval()
    
    coord_col = 'coordinate' if 'coordinate' in metadata.columns else 'basic'
    spec_col = 'specific' if 'specific' in metadata.columns else 'concept'
    
    # 1. Extract Image Features
    img_tensors = []
    valid_indices = []
    for idx, row in metadata.iterrows():
        img_path = row['filepath']
        if not os.path.isabs(img_path):
            img_path = os.path.join(PROJECT_ROOT, img_path)
        if os.path.exists(img_path):
            img_tensors.append(preprocess(Image.open(img_path).convert("RGB")))
            valid_indices.append(idx)
            
    filtered_meta = metadata.iloc[valid_indices].copy().reset_index(drop=True)
    img_batch = torch.stack(img_tensors).to(device)
    
    # 2. Extract Text Vectors for every unique specific concept
    unique_concepts = filtered_meta[[spec_col, coord_col]].drop_duplicates().reset_index(drop=True)
    text_prompts = [f"a photo of a {name}" for name in unique_concepts[spec_col]]
    text_tokens = clip.tokenize(text_prompts).to(device)
    
    with torch.no_grad():
        img_feats = model.encode_image(img_batch)
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        
        text_feats = model.encode_text(text_tokens)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        
    return img_feats.cpu().numpy(), text_feats.cpu().numpy(), unique_concepts, filtered_meta

def generate_shared_progression_grid(
    metadata_path=None, 
    output_plot_path=None,
    encoder_type="both",  # Matches CLIPPruningEngine ('text', 'vision', or 'both')
    pruning_levels=[0.0, 0.3, 0.6, 0.9]
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if metadata_path is None:
        metadata_path = os.path.join(PROJECT_ROOT, "data", "processed", "metadata_processed.csv")
    if output_plot_path is None:
        output_plot_path = os.path.join(PROJECT_ROOT, "data", "results", "shared_space_progression.png")
        
    metadata = pd.read_csv(metadata_path)
    base_model, preprocess = clip.load("ViT-B/32", device=device)
    
    print(f"[*] Extracting joint embeddings under '{encoder_type.upper()}' projection hub pruning...")
    feats_by_level = {}
    
    for p in pruning_levels:
        # Deepcopy base_model to prevent compounding in-place pruning mutations across iterations
        model_copy = copy.deepcopy(base_model)
        engine = CLIPPruningEngine(model_copy)
        pruned = engine.get_pruned_model(amount=p, encoder_type=encoder_type)
            
        V, T, unique_concepts, meta = extract_joint_embeddings(pruned, metadata, preprocess, device)
        feats_by_level[p] = (V, T, unique_concepts, meta)

    # Cosine t-SNE & Procrustes Spatial Alignment
    print("[*] Computing Cosine-Metric t-SNE projections and Procrustes spatial alignment...")
    coords_by_level = {}
    
    # 1. Baseline Anchor Space (0% Atrophy)
    V0, T0, _, _ = feats_by_level[0.0]
    combined_0 = np.vstack([V0, T0])
    tsne_base = TSNE(n_components=2, perplexity=30, metric='cosine', random_state=42, init='pca', learning_rate='auto')
    base_raw = tsne_base.fit_transform(combined_0)
    
    # Zero-center AND normalize base_raw to unit Frobenius norm to align with scipy's procrustes output
    base_centered = base_raw - np.mean(base_raw, axis=0)
    base_coords = base_centered / np.linalg.norm(base_centered)
    
    coords_by_level[0.0] = (base_coords[:len(V0)], base_coords[len(V0):])
    
    # 2. Align subsequent atrophy levels to baseline
    for p in pruning_levels[1:]:
        V_p, T_p, _, _ = feats_by_level[p]
        combined_p = np.vstack([V_p, T_p])
        tsne_p = TSNE(n_components=2, perplexity=30, metric='cosine', random_state=42, init='pca', learning_rate='auto')
        raw_p = tsne_p.fit_transform(combined_p)
        
        # Procrustes alignment to 0% baseline frame
        _, aligned_p, _ = procrustes(base_coords, raw_p)
        coords_by_level[p] = (aligned_p[:len(V_p)], aligned_p[len(V_p):])

    # Dynamic shared limits computed across ALL coordinates to avoid point clipping
    all_x, all_y = [], []
    for img_c, text_c in coords_by_level.values():
        all_x.extend(img_c[:, 0])
        all_x.extend(text_c[:, 0])
        all_y.extend(img_c[:, 1])
        all_y.extend(text_c[:, 1])
        
    x_margin = (max(all_x) - min(all_x)) * 0.08
    y_margin = (max(all_y) - min(all_y)) * 0.08
    xlim = (min(all_x) - x_margin, max(all_x) + x_margin)
    ylim = (min(all_y) - y_margin, max(all_y) + y_margin)

    # Render 2x2 Plot Grid
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    
    for idx, p in enumerate(pruning_levels):
        ax = axes[idx]
        img_coords, text_coords = coords_by_level[p]
        _, _, concepts_df, meta = feats_by_level[p]
        coord_col = 'coordinate' if 'coordinate' in meta.columns else 'basic'
        
        # Plot Image Dots (•)
        for cat, color in CATEGORY_COLORS.items():
            mask = (meta[coord_col] == cat).values
            if mask.any():
                ax.scatter(
                    img_coords[mask, 0], img_coords[mask, 1], 
                    c=color, alpha=0.6, s=30, zorder=2, edgecolors='none'
                )
                
        # Plot Text Concept Vectors (★)
        for t_idx, row in concepts_df.iterrows():
            cat = row[coord_col]
            if cat in CATEGORY_COLORS:
                ax.scatter(
                    text_coords[t_idx, 0], text_coords[t_idx, 1], 
                    c=CATEGORY_COLORS[cat], marker='*', s=200, 
                    edgecolors='black', linewidth=1.0, zorder=5
                )
                
        ax.set_title(f"Shared Space | Projection Atrophy: {int(p * 100)}%", fontsize=13, fontweight='bold', pad=10)
        ax.set_xlabel("Aligned Cosine Dim 1", fontsize=10)
        ax.set_ylabel("Aligned Cosine Dim 2", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.3)
        
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    # Legend
    legend_handles = []
    for cat, color in CATEGORY_COLORS.items():
        legend_handles.append(mpatches.Patch(color=color, label=f"Category: {cat}"))
    legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', label='Image Vector (•)', markerfacecolor='gray', markersize=8))
    legend_handles.append(plt.Line2D([0], [0], marker='*', color='w', label='Text Concept Vector (★)', markerfacecolor='black', markeredgecolor='black', markersize=12))
    
    fig.legend(
        handles=legend_handles, 
        loc="center right", 
        bbox_to_anchor=(1.16, 0.5), 
        fontsize=11, 
        frameon=True, 
        facecolor="white"
    )

    plt.suptitle(
        f"Shared Multimodal Space Dissolution Under '{encoder_type.upper()}' Hub Atrophy\n(Projection Matrices Target Pruning)", 
        fontsize=15, fontweight='bold', y=0.98
    )
    plt.tight_layout(rect=[0, 0, 0.98, 0.95])
    
    os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)
    plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[+] Progression chart saved to:\n    {output_plot_path}")

if __name__ == "__main__":
    generate_shared_progression_grid(encoder_type="both")