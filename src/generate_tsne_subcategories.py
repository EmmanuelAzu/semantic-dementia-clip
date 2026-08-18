import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.manifold import TSNE

# 18-stage pruning schedule (5% increments from 0% to 85%)
PRUNING_LEVELS_5PCT = [
    round(x, 2) for x in np.arange(0.00, 0.86, 0.05).tolist()
]


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


def generate_tsne_grid_with_key(
    evaluator,
    pruning_levels=PRUNING_LEVELS_5PCT,
    samples_per_class=15,
    output_dir="./data/results/tsne",
    n_cols=6,
):
    """Generates a decluttered multi-stage t-SNE grid plot with stratified sampling across 5% pruning increments."""
    print(
        f"[*] Generating decluttered {len(pruning_levels)}-stage t-SNE grid (Max {samples_per_class} samples/class)..."
    )
    os.makedirs(output_dir, exist_ok=True)
    meta = evaluator.metadata.reset_index(drop=True)
    spec_col, coord_col, super_col = _get_taxonomy_mapping(meta)
    concept_lookup = meta.drop_duplicates(spec_col).set_index(spec_col)

    # Color palette for Coordinate Subcategories
    subcategories = sorted(list(meta[coord_col].unique()))
    palette = sns.color_palette("tab10", n_colors=len(subcategories))
    subcat_to_color = {sub: palette[i] for i, sub in enumerate(subcategories)}

    n_stages = len(pruning_levels)
    n_rows = int(np.ceil(n_stages / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3.8 * n_cols, 4.0 * n_rows)
    )
    axes_flat = axes.flatten() if n_stages > 1 else [axes]

    marker_map = {
        "Correct": ("o", 40, 0.75, 0.0, "none"),
        "Coordinate Error": ("s", 50, 0.85, 1.0, "black"),
        "Superordinate Error": ("^", 60, 0.90, 1.0, "black"),
        "Domain Collapse": ("X", 70, 0.95, 1.2, "darkred"),
    }

    # Pre-select stratified random indices to keep identical subsamples across all stages
    np.random.seed(42)
    selected_indices = []
    for concept in meta[spec_col].unique():
        concept_idx = meta[meta[spec_col] == concept].index.values
        n_select = min(len(concept_idx), samples_per_class)
        chosen = np.random.choice(concept_idx, size=n_select, replace=False)
        selected_indices.extend(chosen)

    selected_indices = sorted(selected_indices)
    print(
        f"[*] Subsampled {len(selected_indices)} total points ({samples_per_class}/class) across {len(meta)} dataset items."
    )

    for idx, p_level in enumerate(pruning_levels):
        ax = axes_flat[idx]
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
            f"Atrophy: {p_level * 100:.0f}%", fontsize=11, fontweight="bold"
        )
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.tick_params(labelsize=7)

    # Turn off unused subplots if stages do not fill grid completely
    for idx in range(n_stages, len(axes_flat)):
        axes_flat[idx].axis("off")

    # Construct Legends
    subcat_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=col,
            markersize=8,
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
            markersize=7,
            label="Correct",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="gray",
            markeredgecolor="black",
            markersize=7,
            label="Coordinate Error",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor="gray",
            markeredgecolor="black",
            markersize=7,
            label="Superordinate Error",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="X",
            color="w",
            markerfacecolor="red",
            markeredgecolor="darkred",
            markersize=8,
            label="Domain Collapse",
        ),
    ]

    leg1 = fig.legend(
        handles=subcat_handles,
        title="Coordinate Subcategories",
        title_fontsize="10",
        loc="upper center",
        bbox_to_anchor=(0.35, -0.01),
        ncol=min(4, len(subcategories)),
        frameon=True,
        facecolor="#f9f9f9",
    )

    fig.legend(
        handles=error_handles,
        title="Error Types & Markers",
        title_fontsize="10",
        loc="upper center",
        bbox_to_anchor=(0.75, -0.01),
        ncol=2,
        frameon=True,
        facecolor="#f9f9f9",
    )

    fig.add_artist(leg1)

    plt.suptitle(
        f"Joint Space Feature Manifold Trajectory Across 5% Atrophy Increments (0% to 85%, Subsampled {samples_per_class} Points/Class)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    save_path = os.path.join(output_dir, "tsne_trajectory_5pct_grid.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved extended t-SNE grid plot to: {save_path}")
    return save_path


# Alias for backward compatibility with earlier imports
generate_tsne_5levels_with_key = generate_tsne_grid_with_key