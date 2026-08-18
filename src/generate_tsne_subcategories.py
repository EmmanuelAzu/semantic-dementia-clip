import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.manifold import TSNE

# Selected 5-stage pruning schedule
PRUNING_LEVELS_5 = [0.00, 0.15, 0.35, 0.60, 0.85]


def _get_taxonomy_mapping(metadata):
    spec_col = next(
        (c for c in ["specific", "concept", "label"] if c in metadata.columns),
        "specific",
    )
    coord_col = next(
        (
            c
            for c in ["coordinate", "basic", "category"]
            if c in metadata.columns
        ),
        "coordinate",
    )
    super_col = next(
        (
            c
            for c in ["superordinate", "domain", "macro"]
            if c in metadata.columns
        ),
        "superordinate",
    )
    return spec_col, coord_col, super_col


def generate_tsne_5levels_with_key(
    evaluator,
    pruning_levels=PRUNING_LEVELS_5,
    samples_per_class=15,  # Control point density (e.g., 15-20 per class)
    output_dir="./data/results/tsne",
):
    """Generates decluttered 5-stage t-SNE plots by stratifying/subsampling data points per specific class."""
    print(
        f"[*] Generating decluttered 5-stage t-SNE plots (Max {samples_per_class} samples/class)..."
    )
    os.makedirs(output_dir, exist_ok=True)
    meta = evaluator.metadata.reset_index(drop=True)
    spec_col, coord_col, super_col = _get_taxonomy_mapping(meta)
    concept_lookup = meta.drop_duplicates(spec_col).set_index(spec_col)

    # Color palette for Coordinate Subcategories
    subcategories = sorted(list(meta[coord_col].unique()))
    palette = sns.color_palette("tab10", n_colors=len(subcategories))
    subcat_to_color = {sub: palette[i] for i, sub in enumerate(subcategories)}

    fig, axes = plt.subplots(1, 5, figsize=(25, 5.2))

    marker_map = {
        "Correct": ("o", 45, 0.80, 0.0, "none"),
        "Coordinate Error": ("s", 55, 0.90, 1.2, "black"),
        "Superordinate Error": ("^", 65, 0.95, 1.2, "black"),
        "Domain Collapse": ("X", 75, 1.00, 1.5, "darkred"),
    }

    # Pre-select stratified random indices to keep identical subsamples across all 5 pruning stages
    np.random.seed(42)
    selected_indices = []
    for concept in meta[spec_col].unique():
        concept_idx = meta[meta[spec_col] == concept].index.values
        n_select = min(len(concept_idx), samples_per_class)
        chosen = np.random.choice(concept_idx, size=n_select, replace=False)
        selected_indices.extend(chosen)

    selected_indices = sorted(selected_indices)
    print(
        f"[*] Reduced total sample points from {len(meta)} to {len(selected_indices)} across dataset."
    )

    for idx, p_level in enumerate(pruning_levels):
        ax = axes[idx]
        pruned_model = evaluator._apply_pruning(evaluator.base_model, p_level)
        (
            img_feats_full,
            text_feats,
            true_labels_full,
            unique_concepts,
            eval_meta_full,
        ) = evaluator._extract_joint_features(pruned_model)

        # Subsample feature tensors and labels
        img_feats = img_feats_full[selected_indices]
        true_labels = [true_labels_full[i] for i in selected_indices]

        # Classification similarity on subsampled feature set
        sim_matrix = torch.matmul(img_feats, text_feats.T)
        top1_indices = torch.argmax(sim_matrix, dim=1).numpy()

        # Compute 2D t-SNE projection
        tsne = TSNE(
            n_components=2, perplexity=25, random_state=42, init="pca"
        )
        coords_2d = tsne.fit_transform(img_feats.numpy())

        # Plot subsampled points
        for i, true_spec in enumerate(true_labels):
            pred_spec = unique_concepts[top1_indices[i]]

            if pred_spec == true_spec:
                err_type = "Correct"
            else:
                true_co = (
                    concept_lookup.loc[true_spec, coord_col]
                    if true_spec in concept_lookup.index
                    else None
                )
                pred_co = (
                    concept_lookup.loc[pred_spec, coord_col]
                    if pred_spec in concept_lookup.index
                    else None
                )
                true_su = (
                    concept_lookup.loc[true_spec, super_col]
                    if true_spec in concept_lookup.index
                    else None
                )
                pred_su = (
                    concept_lookup.loc[pred_spec, super_col]
                    if pred_spec in concept_lookup.index
                    else None
                )

                if true_co == pred_co:
                    err_type = "Coordinate Error"
                elif true_su == pred_su:
                    err_type = "Superordinate Error"
                else:
                    err_type = "Domain Collapse"

            subcat = (
                concept_lookup.loc[true_spec, coord_col]
                if true_spec in concept_lookup.index
                else "Other"
            )
            color = subcat_to_color.get(subcat, "gray")
            marker, size, alpha, lw, edge_color = marker_map[err_type]

            ax.scatter(
                coords_2d[i, 0],
                coords_2d[i, 1],
                c=[color],
                marker=marker,
                s=size,
                alpha=alpha,
                linewidths=lw,
                edgecolors=edge_color if err_type != "Correct" else "none",
            )

        ax.set_title(
            f"Atrophy: {p_level * 100:.0f}%", fontsize=13, fontweight="bold"
        )
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.tick_params(labelsize=8)

    # Construct Legends
    subcat_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=col,
            markersize=9,
            label=sub,
        )
        for sub, col in subcat_to_color.items()
    ]

    error_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="gray",
            markersize=8,
            label="Correct Classification",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="gray",
            markeredgecolor="black",
            markersize=8,
            label="Coordinate Error",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor="gray",
            markeredgecolor="black",
            markersize=8,
            label="Superordinate Error",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="X",
            color="w",
            markerfacecolor="red",
            markeredgecolor="darkred",
            markersize=9,
            label="Domain Collapse",
        ),
    ]

    leg1 = fig.legend(
        handles=subcat_handles,
        title="Subcategories (Coordinate Groups)",
        title_fontsize="11",
        loc="upper center",
        bbox_to_anchor=(0.32, -0.02),
        ncol=min(4, len(subcategories)),
        frameon=True,
        facecolor="#f9f9f9",
    )

    fig.legend(
        handles=error_handles,
        title="Error Types & Markers",
        title_fontsize="11",
        loc="upper center",
        bbox_to_anchor=(0.75, -0.02),
        ncol=2,
        frameon=True,
        facecolor="#f9f9f9",
    )

    fig.add_artist(leg1)

    plt.suptitle(
        f"Joint Space Feature Manifold Trajectory Across 5 Stages of Atrophy (Subsampled {samples_per_class} Points/Class)",
        fontsize=15,
        fontweight="bold",
        y=1.05,
    )
    plt.tight_layout()

    save_path = os.path.join(output_dir, "tsne_5levels_subcategories_key.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved decluttered t-SNE plot to: {save_path}")