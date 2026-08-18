import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch

# 18-stage pruning schedule (5% increments from 0% to 85%)
PRUNING_LEVELS_5PCT = [
    round(x, 2) for x in np.arange(0.00, 0.86, 0.05).tolist()
]

# Verified valid dataset concepts across animals and fruit
DEFAULT_TARGET_PROMPTS = ["Tabby Cat", "Golden Retriever", "Bullfrog", "Lemon"]


def _get_column_mappings(metadata):
    """Resolve taxonomy column names across varying metadata formats."""
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


def load_image_safely(img_path):
    """Robust image loader attempting multiple relative path resolutions."""
    if not isinstance(img_path, str):
        return None

    cwd = os.getcwd()
    search_paths = [
        img_path,
        os.path.join(cwd, img_path),
        os.path.join(cwd, "data", img_path),
        os.path.join(cwd, "data", "processed", img_path),
        os.path.join(cwd, "data", "processed", "images", img_path),
        os.path.join(cwd, "data", "images", img_path),
    ]

    for path in search_paths:
        if os.path.exists(path) and not os.path.isdir(path):
            try:
                return Image.open(path).convert("RGB")
            except Exception:
                pass
    return None


def run_bozeat_visual_grid(
    evaluator,
    pruning_levels=PRUNING_LEVELS_5PCT,
    target_prompts=DEFAULT_TARGET_PROMPTS,
    output_dir="./data/results/bozeat_experiment",
):
    """Executes Bozeat et al. 'Draw from Prompt' task and generates an 18-stage visual grid."""
    print("=" * 75)
    print(
        f" RUNNING BOZEAT VISUAL RETRIEVAL ({len(pruning_levels)} STAGES: 5% INCREMENTS) "
    )
    print("=" * 75)

    os.makedirs(output_dir, exist_ok=True)
    meta = evaluator.metadata.reset_index(drop=True)
    spec_col, _, _ = _get_column_mappings(meta)

    # Filter target prompts available in metadata
    available_concepts = list(meta[spec_col].unique())
    selected_prompts = [p for p in target_prompts if p in available_concepts]

    # Fallback selection if target prompts are missing
    if len(selected_prompts) < len(target_prompts):
        for c in available_concepts:
            if c not in selected_prompts:
                selected_prompts.append(c)
            if len(selected_prompts) == len(target_prompts):
                break

    print(
        f"[*] Target prompts selected for visual drawing grid:\n    {selected_prompts}"
    )

    retrieval_grid_data = {p: [] for p in selected_prompts}

    for p_level in pruning_levels:
        print(
            f" [*] Processing joint space retrieval at {p_level * 100:.0f}% pruning..."
        )
        pruned_model = evaluator._apply_pruning(evaluator.base_model, p_level)
        img_feats, text_feats, img_labels, unique_concepts, eval_meta = (
            evaluator._extract_joint_features(pruned_model)
        )

        for prompt_concept in selected_prompts:
            if prompt_concept not in unique_concepts:
                continue

            concept_idx = unique_concepts.index(prompt_concept)
            prompt_vec = text_feats[concept_idx].unsqueeze(0)

            # Cosine similarity against dataset features
            sims = torch.matmul(img_feats, prompt_vec.T).squeeze(1)
            best_img_idx = torch.argmax(sims).item()

            retrieved_row = eval_meta.iloc[best_img_idx]
            retrieved_concept = retrieved_row[spec_col]

            path_col = next(
                (
                    c
                    for c in [
                        "filepath",
                        "filename",
                        "image_path",
                        "path",
                        "file_path",
                    ]
                    if c in retrieved_row.index
                ),
                None,
            )
            img_path = retrieved_row[path_col] if path_col else ""

            is_correct = retrieved_concept == prompt_concept
            retrieval_grid_data[prompt_concept].append(
                (p_level, img_path, retrieved_concept, is_correct)
            )

    # --- RENDER 4x18 VISUAL GRID ---
    n_rows = len(selected_prompts)
    n_cols = len(pruning_levels)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(2.2 * n_cols, 2.8 * n_rows)
    )
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for r_idx, prompt_concept in enumerate(selected_prompts):
        records = retrieval_grid_data[prompt_concept]

        for c_idx, (p_level, img_path, ret_concept, is_correct) in enumerate(
            records
        ):
            ax = axes[r_idx, c_idx]
            img = load_image_safely(img_path)

            if img is not None:
                ax.imshow(img)
            else:
                ax.set_facecolor("#e0e0e0")
                ax.text(
                    0.5,
                    0.5,
                    f"{ret_concept}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="black",
                )

            ax.set_xticks([])
            ax.set_yticks([])

            # Compact title styling for 18-column width
            if is_correct:
                title_text = f"'{ret_concept}' ✓"
                title_color = "darkgreen"
                box_color = "#e6f4ea"
            else:
                title_text = f"'{ret_concept}' ✗"
                title_color = "darkred"
                box_color = "#fce8e6"

            ax.set_title(
                title_text,
                fontsize=7.0,
                color=title_color,
                fontweight="bold",
                pad=3,
                bbox=dict(
                    boxstyle="round,pad=0.15",
                    facecolor=box_color,
                    edgecolor=title_color,
                    lw=0.6,
                ),
            )

            # Column Headers (Pruning Levels)
            if r_idx == 0:
                ax.set_xlabel(
                    f"{p_level * 100:.0f}%",
                    fontsize=9.5,
                    fontweight="bold",
                    labelpad=6,
                )
                ax.xaxis.set_label_position("top")

            # Row Labels (Text Prompts)
            if c_idx == 0:
                ax.set_ylabel(
                    f'Prompt:\n"{prompt_concept}"',
                    fontsize=9.5,
                    fontweight="bold",
                    rotation=0,
                    labelpad=40,
                    ha="right",
                    va="center",
                )

    plt.suptitle(
        "Bozeat 'Draw from Prompt' Task: Visual Retrieval Trajectory Across 5% Atrophy Increments (0% to 85%)",
        fontsize=13,
        fontweight="bold",
        y=1.03,
    )
    plt.tight_layout()

    save_path = os.path.join(output_dir, "bozeat_retrieved_images_5pct.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(
        f"\n[+] Success! 18-stage visual Bozeat grid saved to:\n    {save_path}"
    )
    return save_path