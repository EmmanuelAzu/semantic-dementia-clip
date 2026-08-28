import copy
import os
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE

# 18-stage pruning schedule (5% increments from 0% to 85%)
PRUNING_LEVELS_5PCT = [
    round(x, 2) for x in np.arange(0.00, 0.86, 0.05).tolist()
]

# Preferred colormaps per superordinate domain
DOMAIN_CMAP_PREFERENCES = {
    "animal": "Greens",
    "fauna": "Greens",
    "living": "Greens",
    "vehicle": "Reds",
    "transport": "Reds",
    "artifact": "Blues",
    "object": "Blues",
    "tool": "Oranges",
    "fruit": "Purples",
    "food": "YlOrBr",
    "vegetable": "YlGn",
}

FALLBACK_CMAPS = [
    "Greens", "Reds", "Blues", "Oranges", "Purples",
    "YlOrBr", "PuBu", "YlGn", "RdPu", "GnBu",
]

def _get_taxonomy_mapping(metadata):
    spec_col = next(
        (c for c in ["specific", "concept", "label"] if c in metadata.columns),
        "specific",
    )
    coord_col = next(
        (c for c in ["coordinate", "basic", "category"] if c in metadata.columns),
        "coordinate",
    )
    super_col = next(
        (c for c in ["superordinate", "domain", "macro"] if c in metadata.columns),
        "superordinate",
    )
    return spec_col, coord_col, super_col


def build_hierarchical_concept_palette(meta, spec_col, super_col):
    """Assigns distinct colormap families to superordinate categories and distinct shades to specific concepts."""
    unique_super = sorted(list(meta[super_col].unique()))
    concept_to_color = {}
    super_to_concepts = {}
    used_cmaps = set()

    for idx, super_cat in enumerate(unique_super):
        super_str = str(super_cat).lower()
        matched_cmap = next(
            (cmap for key, cmap in DOMAIN_CMAP_PREFERENCES.items() 
             if key in super_str and cmap not in used_cmaps), 
            None
        )

        if matched_cmap is None:
            matched_cmap = next((cmap for cmap in FALLBACK_CMAPS if cmap not in used_cmaps), None)
        if matched_cmap is None:
            matched_cmap = FALLBACK_CMAPS[idx % len(FALLBACK_CMAPS)]

        used_cmaps.add(matched_cmap)

        concepts_in_domain = sorted(list(meta[meta[super_col] == super_cat][spec_col].unique()))
        super_to_concepts[super_cat] = concepts_in_domain
        cmap_func = cm.get_cmap(matched_cmap)
        n_concepts = len(concepts_in_domain)

        shades = [0.60] if n_concepts == 1 else np.linspace(0.38, 0.85, n_concepts)
        for concept, shade in zip(concepts_in_domain, shades):
            concept_to_color[concept] = cmap_func(shade)

    return concept_to_color, super_to_concepts


def generate_tsne_clean_color_grid(
    evaluator,
    pruning_levels=PRUNING_LEVELS_5PCT,
    samples_per_class=5,
    output_dir="./data/results/tsne",
    n_cols=6,
):
    """Generates an 18-stage t-SNE grid using a globally fitted coordinate system to track true feature drift."""
    print(f"[*] Generating {len(pruning_levels)}-stage hierarchical color t-SNE grid (N={samples_per_class} samples/class)...")
    os.makedirs(output_dir, exist_ok=True)
    
    meta = evaluator.metadata.reset_index(drop=True)
    spec_col, _, super_col = _get_taxonomy_mapping(meta)
    
    concept_to_color, super_to_concepts = build_hierarchical_concept_palette(meta, spec_col, super_col)
    unique_concepts = sorted(list(meta[spec_col].unique()))

    # 1. Deterministic Stratified Sampling
    # Ensures we track the exact same images across all pruning levels
    rng = np.random.default_rng(42)
    selected_indices = []
    for concept in unique_concepts:
        concept_idx = meta[meta[spec_col] == concept].index.values
        n_select = min(len(concept_idx), samples_per_class)
        chosen = rng.choice(concept_idx, size=n_select, replace=False)
        selected_indices.extend(chosen)
    
    selected_indices = sorted(selected_indices)
    n_samples_per_stage = len(selected_indices)

    # 2. Extract Features for ALL stages to establish a global coordinate system
    print("[*] Extracting features across all pruning stages...")
    all_img_feats = []
    true_labels = []

    for p_level in pruning_levels:
        # CRITICAL FIX: Deepcopy prevents consecutive pruning mutations stacking up
        temp_model = copy.deepcopy(evaluator.base_model)
        pruned_model = evaluator._apply_pruning(temp_model, p_level)

        (img_feats_full, _, true_labels_full, _, _) = evaluator._extract_joint_features(pruned_model)

        # Ensure pure, detached numpy arrays
        if isinstance(img_feats_full, torch.Tensor):
            img_feats = img_feats_full[selected_indices].detach().cpu().numpy()
        else:
            img_feats = img_feats_full[selected_indices]
            
        all_img_feats.append(img_feats)

        # Only need to capture labels once since indices are static
        if not true_labels:
            true_labels = [true_labels_full[i] for i in selected_indices]

    # 3. Fit Global t-SNE
    # Stacking ensures a shared topological space; visual changes are real structural decay, not RNG rotation
    X_total = np.vstack(all_img_feats)
    
    # Dynamic perplexity logic to prevent "ring artifacts" on small N
    calc_perp = min(30, max(5, n_samples_per_stage // 4))
    print(f"[*] Fitting global t-SNE on {X_total.shape[0]} total points (perplexity={calc_perp})...")
    
    tsne = TSNE(n_components=2, perplexity=calc_perp, random_state=42, init="pca")
    X_total_2d = tsne.fit_transform(X_total)

    # 4. Plotting Loop
    n_stages = len(pruning_levels)
    n_rows = int(np.ceil(n_stages / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.8 * n_cols, 4.0 * n_rows))
    axes_flat = axes.flatten() if n_stages > 1 else [axes]

    for idx, p_level in enumerate(pruning_levels):
        ax = axes_flat[idx]
        
        # Slice out the coordinates belonging to this specific stage
        start_idx = idx * n_samples_per_stage
        end_idx = start_idx + n_samples_per_stage
        coords_2d = X_total_2d[start_idx:end_idx]

        for i, true_spec in enumerate(true_labels):
            color = concept_to_color.get(true_spec, "gray")
            ax.scatter(
                coords_2d[i, 0], coords_2d[i, 1],
                c=[color], marker="o", s=35, alpha=0.85,
                linewidths=0.3, edgecolors="black",
            )

        ax.set_title(f"Atrophy: {p_level * 100:.0f}%", fontsize=11, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.25)
        ax.tick_params(labelsize=7)
        
        # Fix axis limits across all subplots to match the global min/max
        x_min, x_max = X_total_2d[:, 0].min(), X_total_2d[:, 0].max()
        y_min, y_max = X_total_2d[:, 1].min(), X_total_2d[:, 1].max()
        padding_x = (x_max - x_min) * 0.05
        padding_y = (y_max - y_min) * 0.05
        ax.set_xlim(x_min - padding_x, x_max + padding_x)
        ax.set_ylim(y_min - padding_y, y_max + padding_y)

    for idx in range(n_stages, len(axes_flat)):
        axes_flat[idx].axis("off")

    # Grouped Legend by Superordinate Category
    legend_handles = []
    for super_cat, concepts in super_to_concepts.items():
        legend_handles.append(plt.Line2D([0], [0], color="w", label=f"[{str(super_cat).upper()}]", markersize=0))
        for concept in concepts:
            col = concept_to_color[concept]
            legend_handles.append(
                plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=col, markeredgecolor="black", markeredgewidth=0.3, markersize=7, label=f"  {concept}")
            )

    fig.legend(
        handles=legend_handles, title="Domains & Object Concepts", title_fontsize="11",
        loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=min(6, len(super_to_concepts) + 2),
        frameon=True, facecolor="#f9f9f9", fontsize=8,
    )

    plt.suptitle(
        f"Global Feature Trajectory Across 5% Atrophy Increments (0% to 85%)\n"
        f"Domain-Grouped Palettes (N={samples_per_class} samples/class) | Shared Global t-SNE Space",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    save_path = os.path.join(output_dir, "tsne_concept_dispersion_global_align.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved global trajectory t-SNE grid to:\n    {save_path}")
    return save_path

generate_tsne_grid_with_key = generate_tsne_clean_color_grid