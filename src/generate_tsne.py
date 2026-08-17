import os
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE


def generate_joint_hierarchical_tsne(
    evaluator, pruning_levels=[0.0, 0.05, 0.15, 0.35, 0.55, 0.70, 0.80, 0.85]
):
    """Plot t-SNE trajectories dynamically scaled to match any number of pruning levels."""
    print("[*] Generating Joint Hierarchical t-SNE Plots...")
    meta = evaluator.metadata.reset_index(drop=True)

    spec_col = next(
        (c for c in ["specific", "concept", "label"] if c in meta.columns),
        "specific",
    )
    coord_col = next(
        (c for c in ["coordinate", "category"] if c in meta.columns),
        "coordinate",
    )
    super_col = next(
        (
            c
            for c in ["superordinate", "domain", "macro"]
            if c in meta.columns
        ),
        "superordinate",
    )

    super_classes = sorted(list(meta[super_col].unique()))
    coord_classes = sorted(list(meta[coord_col].unique()))
    spec_classes = sorted(list(meta[spec_col].unique()))

    hue_map = {
        s: i / max(1, len(super_classes)) for i, s in enumerate(super_classes)
    }

    color_lut = {}
    for _, row in meta.iterrows():
        sp, co, su = row[spec_col], row[coord_col], row[super_col]
        if sp not in color_lut:
            h = hue_map[su]
            co_idx = coord_classes.index(co)
            s = 0.5 + 0.5 * (co_idx / max(1, len(coord_classes)))
            sp_idx = spec_classes.index(sp)
            v = 0.35 + 0.55 * (sp_idx / max(1, len(spec_classes)))
            color_lut[sp] = mcolors.hsv_to_rgb((h, s, v))

    # Dynamically compute rows and columns for the grid layout
    n_plots = len(pruning_levels)
    n_cols = 4
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4.5 * n_rows))
    axes = np.array(axes).flatten()

    for idx, p_level in enumerate(pruning_levels):
        pruned_model = evaluator._apply_pruning(evaluator.base_model, p_level)
        img_feats, text_feats, labels, _, _ = evaluator._extract_joint_features(
            pruned_model
        )

        all_feats = torch.cat([img_feats, text_feats], dim=0).numpy()
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, init="pca")
        tsne_coords = tsne.fit_transform(all_feats)

        n_img = len(img_feats)
        img_coords = tsne_coords[:n_img]
        text_coords = tsne_coords[n_img:]

        ax = axes[idx]
        img_colors = [color_lut[c] for c in labels]

        ax.scatter(
            img_coords[:, 0],
            img_coords[:, 1],
            c=img_colors,
            marker="o",
            alpha=0.7,
            s=25,
            label="Image Embeddings",
        )
        ax.scatter(
            text_coords[:, 0],
            text_coords[:, 1],
            c="black",
            marker="X",
            s=70,
            edgecolors="white",
            linewidth=1,
            label="Text Prompts",
        )

        ax.set_title(
            f"Pruning: {p_level * 100:.0f}%", fontsize=12, fontweight="bold"
        )
        ax.grid(True, linestyle="--", alpha=0.4)

    # Hide unused subplot axes
    for unused in range(n_plots, len(axes)):
        fig.delaxes(axes[unused])

    plt.suptitle(
        "Full Dataset Joint Feature Space Trajectory Under Hierarchical HSV Color Mapping",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    save_path = os.path.join(
        "data", "results", "hierarchical_hsv_tsne_trajectory.png"
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Hierarchical HSV t-SNE plot saved to: {save_path}")