"""
generate_tsne.py
----------------
Generates a 2x2 t-SNE grid visualization with Procrustes spatial alignment,
hierarchical color palettes, and 1.5-sigma confidence boundary ellipses.
"""

import os
import sys
import torch
import clip
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
from scipy.spatial import procrustes
from sklearn.manifold import TSNE
from PIL import Image

# Dynamically resolve project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.pruning_engine import CLIPPruningEngine

# 1. HIERARCHICAL COLOR PALETTE MAPPING
# Domain/Superordinate = Color Family; Coordinate Category = Specific Shade
CATEGORY_COLORS = {
    "Mammal": "#d73027",             # Deep Crimson (Living / Mammal)
    "Non-Mammal": "#4575b4",         # Deep Navy Blue (Living / Aquatic & Reptile)
    "Flora & Food": "#2ca02c",       # Forest Green (Living / Produce & Plants)
    "Land Vehicle": "#ff7f0e",       # Bright Orange (Non-Living / Land Vehicles)
    "Non-Land Vehicle": "#17becf",   # Cyan / Sky-Blue (Non-Living / Aircraft & Ships)
    "Home & Furniture": "#762a83"    # Purple (Non-Living / Household Objects)
}

def draw_confidence_ellipse(x, y, ax, color, n_std=1.5, alpha=0.18):
    """Draws a 2D Gaussian confidence boundary ellipse around a category cluster."""
    if len(x) < 4:
        return
        
    cov = np.cov(x, y)
    vals, vectors = np.linalg.eigh(cov)
    
    order = vals.argsort()[::-1]
    vals, vectors = vals[order], vectors[:, order]
    
    theta = np.degrees(np.arctan2(*vectors[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(np.maximum(0, vals))
    
    ellipse = Ellipse(
        xy=(np.mean(x), np.mean(y)), 
        width=width, 
        height=height,
        angle=theta, 
        facecolor=color, 
        edgecolor=color, 
        alpha=alpha, 
        linewidth=1.5,
        zorder=1
    )
    ax.add_patch(ellipse)

def extract_embeddings(model, metadata, preprocess, device, batch_size=32):
    """Extracts normalized visual embeddings for dataset images."""
    model.eval()
    batch_images = []
    valid_indices = []
    
    for idx, row in metadata.iterrows():
        img_path = row['filepath']
        if not os.path.isabs(img_path):
            img_path = os.path.join(PROJECT_ROOT, img_path)
            
        if os.path.exists(img_path):
            img = Image.open(img_path).convert("RGB")
            batch_images.append(preprocess(img))
            valid_indices.append(idx)
            
    if not batch_images:
        raise ValueError("No valid images found to extract embeddings. Check filepaths in metadata.")
        
    filtered_meta = metadata.iloc[valid_indices].copy().reset_index(drop=True)
    
    embedding_chunks = []
    with torch.no_grad():
        for i in range(0, len(batch_images), batch_size):
            chunk = torch.stack(batch_images[i:i + batch_size]).to(device)
            feats = model.encode_image(chunk)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            embedding_chunks.append(feats.cpu().numpy())
            
    all_embeddings = np.concatenate(embedding_chunks, axis=0)
    return all_embeddings, filtered_meta

def generate_tsne_grid(
    metadata_path=None, 
    output_plot_path=None, 
    encoder_type="vision", 
    pruning_levels=[0.0, 0.3, 0.6, 0.9]
):
    """
    Generates a 2x2 t-SNE grid plot featuring Procrustes spatial alignment,
    hierarchical color family mapping, and 1.5-sigma confidence boundary overlays.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if metadata_path is None:
        metadata_path = os.path.join(PROJECT_ROOT, "data", "processed", "metadata_processed.csv")
    if output_plot_path is None:
        output_plot_path = os.path.join(PROJECT_ROOT, "data", "results", "enhanced_tsne_grid.png")
        
    metadata = pd.read_csv(metadata_path)
    coord_col = 'coordinate' if 'coordinate' in metadata.columns else 'basic'
    
    # Filter metadata to keep primary coordinate classes in CATEGORY_COLORS
    filtered_meta = metadata[metadata[coord_col].isin(CATEGORY_COLORS.keys())].copy().reset_index(drop=True)
    if filtered_meta.empty:
        filtered_meta = metadata.copy()
    
    # Load base model & pruning engine
    base_model, preprocess = clip.load("ViT-B/32", device=device)
    pruning_engine = CLIPPruningEngine(base_model)
    
    print(f"[*] Extracting visual embeddings across {len(pruning_levels)} atrophy levels...")
    embeddings_by_level = {}
    
    for p in pruning_levels:
        if encoder_type == "joint":
            pruned_model = pruning_engine.get_pruned_model(amount=p, encoder_type="text")
            pruned_model = pruning_engine.get_pruned_model(amount=p, encoder_type="vision", model=pruned_model)
        else:
            pruned_model = pruning_engine.get_pruned_model(amount=p, encoder_type=encoder_type)
            
        pruned_model.eval()
        feats, meta = extract_embeddings(pruned_model, filtered_meta, preprocess, device)
        embeddings_by_level[p] = feats

    # Compute t-SNE & Apply Procrustes Spatial Alignment relative to 0% baseline
    print("[*] Computing t-SNE projections and applying Procrustes spatial alignment...")
    tsne_coords = {}
    
    tsne_base = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
    coords_0 = tsne_base.fit_transform(embeddings_by_level[0.0])
    
    # Procrustes transform aligns subsequent levels to 0% baseline coordinate orientation
    for p in pruning_levels[1:]:
        tsne_p = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
        raw_coords_p = tsne_p.fit_transform(embeddings_by_level[p])
        
        mtx_base, mtx_aligned, _ = procrustes(coords_0, raw_coords_p)
        tsne_coords[p] = mtx_aligned
        
    tsne_coords[0.0] = mtx_base

    # Render 2x2 Subplot Grid
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    
    for idx, p in enumerate(pruning_levels):
        ax = axes[idx]
        coords = tsne_coords[p]
        
        for cat_name, color in CATEGORY_COLORS.items():
            indices = meta[meta[coord_col] == cat_name].index.values
            if len(indices) == 0:
                continue
                
            x_pts = coords[indices, 0]
            y_pts = coords[indices, 1]
            
            # Layer A: 1.5-sigma Confidence Boundary Ellipse
            draw_confidence_ellipse(x_pts, y_pts, ax, color=color, n_std=1.5, alpha=0.18)
            
            # Layer B: Scatter Points
            ax.scatter(
                x_pts, y_pts, 
                c=color, 
                label=cat_name, 
                s=35, 
                alpha=0.85, 
                edgecolors='white', 
                linewidth=0.5,
                zorder=2
            )
            
        ax.set_title(f"Atrophy Level: {int(p * 100)}%", fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel("Aligned t-SNE Dim 1", fontsize=11)
        ax.set_ylabel("Aligned t-SNE Dim 2", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.4)

    # Global Legend
    legend_patches = [
        mpatches.Patch(color=color, label=cat_name) 
        for cat_name, color in CATEGORY_COLORS.items()
    ]
    fig.legend(
        handles=legend_patches, 
        title="Coordinate Category", 
        loc="center right", 
        bbox_to_anchor=(1.14, 0.5),
        fontsize=11, 
        title_fontsize=12,
        frameon=True,
        facecolor="white"
    )

    plt.suptitle(
        f"Semantic Cluster Dissolution Under {encoder_type.upper()} Hub Atrophy\n(Procrustes-Aligned t-SNE with 1.5σ Confidence Boundaries)", 
        fontsize=16, 
        fontweight='bold', 
        y=0.98
    )
    plt.tight_layout(rect=[0, 0, 0.98, 0.95])
    
    os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)
    plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[+] Enhanced t-SNE grid successfully saved to:\n    {output_plot_path}")

if __name__ == "__main__":
    generate_tsne_grid(encoder_type="joint")