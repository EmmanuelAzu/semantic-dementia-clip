import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.manifold import TSNE

# 18-stage pruning schedule (5% increments from 0% to 85%)
PRUNING_LEVELS_5PCT = [round(x, 2) for x in np.arange(0.00, 0.86, 0.05).tolist()]


def _get_taxonomy_mapping(metadata):
    spec_col = next((c for c in ["specific", "concept", "label"] if c in metadata.columns), "specific")
    coord_col = next((c for c in ["coordinate", "basic", "category"] if c in metadata.columns), "coordinate")
    super_col = next((c for c in ["superordinate", "domain", "macro"] if c in metadata.columns), "superordinate")
    return spec_col, coord_col, super_col


def generate_tsne_clean_color_grid(
    evaluator,
    pruning_levels=PRUNING_LEVELS_5PCT,
    samples_per_class=15,
    output_dir="./data/results/tsne",
    n_cols=6,
    use_single_word_prompts=False,
):
    """Generates an 18-stage t-SNE grid rendering purely concept color clusters with uniform markers."""
    print(f"[*] Generating {len(pruning_levels)}-stage color-dispersion t-SNE grid...")
    os.makedirs(output_dir, exist_ok=True)
    meta = evaluator.metadata.reset_index(drop=True)
    spec_col, _, _ = _get_taxonomy_mapping(meta)

    # Color palette mapped directly to Specific Object Concepts
    unique_concepts = sorted(list(meta[spec_col].unique()))
    palette = sns.color_palette("husl", n_colors=len(unique_concepts))
    concept_to_color = {concept: palette[i] for i, concept in enumerate(unique_concepts)}

    n_stages = len(pruning_levels)
    n_rows = int(np.ceil(n_stages / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.8 * n_cols, 4.0 * n_rows))
    axes_flat = axes.flatten() if n_stages > 1 else [axes]

    # Stratified sample selection across concepts
    np.random.seed(42)
    selected_indices = []
    for concept in unique_concepts:
        concept_idx = meta[meta[spec_col] == concept].index.values
        n_select = min(len(concept_idx), samples_per_class)
        chosen = np.random.choice(concept_idx, size=n_select, replace=False)
        selected_indices.extend(chosen)

    selected_indices = sorted(selected_indices)

    for idx, p_level in enumerate(pruning_levels):
        ax = axes_flat[idx]
        pruned_model = evaluator._apply_pruning(evaluator.base_model, p_level)

        (
            img_feats_full,
            text_feats,
            true_labels_full,
            eval_concepts,
            _,
        ) = evaluator._extract_joint_features(pruned_model)

        if use_single_word_prompts:
            text_tokens = evaluator.tokenizer(eval_concepts).to(evaluator.device)
            with torch.no_grad():
                text_feats = pruned_model.encode_text(text_tokens)
                text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

        img_feats = img_feats_full[selected_indices]
        true_labels = [true_labels_full[i] for i in selected_indices]

        tsne = TSNE(n_components=2, perplexity=25, random_state=42, init="pca")
        coords_2d = tsne.fit_transform(img_feats.numpy())

        # Render uniform markers colored strictly by concept
        for i, true_spec in enumerate(true_labels):
            color = concept_to_color.get(true_spec, "gray")
            ax.scatter(
                coords_2d[i, 0],
                coords_2d[i, 1],
                c=[color],
                marker="o",
                s=28,
                alpha=0.75,
                linewidths=0,
                edgecolors="none",
            )

        ax.set_title(f"Atrophy: {p_level * 100:.0f}%", fontsize=11, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.25)
        ax.tick_params(labelsize=7)

    for idx in range(n_stages, len(axes_flat)):
        axes_flat[idx].axis("off")

    # Legend strictly for Object Concepts
    concept_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=col,
            markersize=8,
            label=concept,
        )
        for concept, col in concept_to_color.items()
    ]

    fig.legend(
        handles=concept_handles,
        title="Object Concepts",
        title_fontsize="11",
        loc="lower center",
        bbox_to_anchor=(0.5, -0.06),
        ncol=min(8, len(unique_concepts)),
        frameon=True,
        facecolor="#f9f9f9",
    )

    plt.suptitle(
        f"Joint Space Feature Mixing Across 5% Atrophy Increments (0% to 85%)\n"
        f"Prompting: {'Single-Word' if use_single_word_prompts else 'Contextual Template'}",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    save_path = os.path.join(output_dir, "tsne_concept_dispersion_clean.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved clean color-dispersion t-SNE grid to: {save_path}")
    return save_path


generate_tsne_grid_with_key = generate_tsne_clean_color_grid