import copy
import os
import sys
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
from scipy.spatial import procrustes
from sklearn.manifold import TSNE

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.pruning_engine import CLIPPruningEngine

HIERARCHICAL_COLOR_MAP = {
    "Mammal": {
        "Tabby Cat": "#990000",
        "Egyptian Cat": "#d73027",
        "German Shepherd": "#f46d43",
        "Golden Retriever": "#fdae61",
    },
    "Non-Mammal": {
        "Goldfish": "#053061",
        "Bullfrog": "#2166ac",
        "Scorpion": "#4393c3",
        "Tarantula": "#92c5de",
    },
    "Flora & Food": {
        "Lemon": "#00441b",
        "Banana": "#238b45",
        "Guacamole": "#66c2a4",
        "Espresso": "#a1d99b",
    },
    "Land Vehicle": {
        "Sports Car": "#8c510a",
        "Police Van": "#bf812d",
        "Motor Scooter": "#dfc27d",
        "Freight Car": "#f6e8c3",
    },
    "Non-Land Vehicle": {
        "Airplane": "#276419",
        "Catamaran": "#4d9221",
        "Lifeboat": "#7fbc41",
        "Gondola": "#b8e186",
    },
    "Home & Furniture": {
        "Rocking Chair": "#40004b",
        "Dining Table": "#762a83",
        "Bathtub": "#9970ab",
        "Candle": "#c2a5cf",
    },
}


def draw_confidence_ellipse(x, y, ax, color, n_std=1.5, alpha=0.15):
    if len(x) < 3:
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
        linewidth=1.2,
        zorder=1,
    )
    ax.add_patch(ellipse)


def generate_joint_hierarchical_tsne(
    evaluator,
    pruning_levels=[0.0, 0.25, 0.50, 0.75, 0.90, 0.95],
    output_path=None,
    max_tsne_samples=2000,
):
    if output_path is None:
        output_path = os.path.join(
            PROJECT_ROOT, "data", "results", "joint_hierarchical_tsne.png"
        )

    meta = evaluator.metadata.copy().reset_index(drop=True)
    spec_col = (
        "specific"
        if "specific" in meta.columns
        else ("concept" if "concept" in meta.columns else "name")
    )

    if len(meta) > max_tsne_samples:
        sample_indices = meta.sample(
            n=max_tsne_samples, random_state=42
        ).index.values
        meta_tsne = meta.loc[sample_indices].reset_index(drop=True)
    else:
        sample_indices = None
        meta_tsne = meta

    base_model = evaluator.base_model
    embeddings_by_level = {}

    for p in pruning_levels:
        model_copy = copy.deepcopy(base_model)
        engine = CLIPPruningEngine(model_copy)
        pruned_model = engine.get_pruned_model(
            amount=float(p), encoder_type="joint"
        )

        img_feats, text_feats, _, _ = evaluator._extract_joint_features(
            pruned_model
        )

        if sample_indices is not None:
            idx_tensor = torch.as_tensor(sample_indices, device=img_feats.device)
            img_feats = img_feats[idx_tensor]

        embeddings_by_level[p] = np.vstack(
            [img_feats.detach().cpu().numpy(), text_feats.detach().cpu().numpy()]
        )

    perp = min(30, max(5, (len(meta_tsne) - 1) // 3))
    tsne_base = TSNE(n_components=2, perplexity=perp, random_state=42)
    coords_0 = tsne_base.fit_transform(embeddings_by_level[0.0])
    norm_0 = np.linalg.norm(coords_0)
    coords_0 = (coords_0 - np.mean(coords_0, axis=0)) / (norm_0 if norm_0 > 0 else 1.0)

    tsne_coords = {0.0: coords_0}
    for p in pruning_levels[1:]:
        tsne_p = TSNE(n_components=2, perplexity=perp, random_state=42)
        raw_p = tsne_p.fit_transform(embeddings_by_level[p])
        _, aligned_p, _ = procrustes(coords_0, raw_p)
        tsne_coords[p] = aligned_p

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.flatten()

    for idx, p in enumerate(pruning_levels):
        ax = axes[idx]
        coords = tsne_coords[p][: len(meta_tsne)]

        for domain, sub_map in HIERARCHICAL_COLOR_MAP.items():
            for concept_cat, color in sub_map.items():
                indices = meta_tsne[
                    meta_tsne[spec_col] == concept_cat
                ].index.values
                if len(indices) == 0:
                    continue

                x_pts, y_pts = coords[indices, 0], coords[indices, 1]
                draw_confidence_ellipse(
                    x_pts, y_pts, ax, color=color, n_std=1.5
                )
                ax.scatter(
                    x_pts,
                    y_pts,
                    c=color,
                    label=f"{domain}: {concept_cat}",
                    s=20,
                    alpha=0.7,
                )

        ax.set_title(
            f"Atrophy Level: {int(p * 100)}%", fontsize=12, fontweight="bold"
        )
        ax.grid(True, linestyle="--", alpha=0.3)

    plt.suptitle(
        "Hierarchical Joint Space Clustering Dissolution (1.5σ Boundaries)",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Hierarchical Joint t-SNE plot saved to:\n    {output_path}")