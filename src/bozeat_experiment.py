import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch

# Extended 10-stage pruning schedule (0% to 85% at ~10% increments)
PRUNING_LEVELS_EXTENDED = [0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85]

# 4 target prompts spanning living and non-living categories
DEFAULT_TARGET_PROMPTS = ["Tabby Cat", "Golden Retriever", "Sports Car", "Lemon"]


def _get_column_mappings(metadata):
    """Resolve taxonomy column names across varying metadata formats."""
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
    pruning_levels=PRUNING_LEVELS_EXTENDED,
    target_prompts=DEFAULT_TARGET_PROMPTS,
    output_dir="./data/results/bozeat_experiment",
):
    """Executes Bozeat et al. 'Draw from Prompt' task and generates a 4x10 visual grid

    showing actual retrieved image files across 10 pruning increments.
    """
    print("=" * 70)
    print(" RUNNING BOZEAT ET AL. 'DRAW FROM PROMPT' VISUAL RETRIEVAL ")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)
    meta = evaluator.metadata.reset_index(drop=True)
    spec_col, coord_col, super_col = _get_column_mappings(meta)

    # Filter target prompts available in metadata
    available_concepts = list(meta[spec_col].unique())
    selected_prompts = [p for p in target_prompts if p in available_concepts]

    # Fallback selection if target prompts are named slightly differently in metadata
    if len(selected_prompts) < len(target_prompts):
        for c in available_concepts:
            if c not in selected_prompts:
                selected_prompts.append(c)
            if len(selected_prompts) == len(target_prompts):
                break

    print(f"[*] Target prompts selected for visual drawing grid:\n    {selected_prompts}")

    retrieval_grid_data = {p: [] for p in selected_prompts}

    for p_level in pruning_levels:
        print(f" [*] Processing joint space retrieval at {p_level * 100:.0f}% pruning...")
        pruned_model = evaluator._apply_pruning(evaluator.base_model, p_level)
        img_feats, text_feats, img_labels, unique_concepts, eval_meta = (
            evaluator._extract_joint_features(pruned_model)
        )

        for prompt_concept in selected_prompts:
            if prompt_concept not in unique_concepts:
                continue

            concept_idx = unique_concepts.index(prompt_concept)
            prompt_vec = text_feats[concept_idx].unsqueeze(0)

            # Compute similarity of target prompt against all dataset images
            sims = torch.matmul(img_feats, prompt_vec.T).squeeze(1)
            best_img_idx = torch.argmax(sims).item()

            retrieved_row = eval_meta.iloc[best_img_idx]
            retrieved_concept = retrieved_row[spec_col]

            path_col = next(
                (c for c in ["filepath", "filename", "image_path", "path", "file_path"] if c in retrieved_row.index),
                None,
            )
            img_path = retrieved_row[path_col] if path_col else ""

            is_correct = (retrieved_concept == prompt_concept)
            retrieval_grid_data[prompt_concept].append(
                (p_level, img_path, retrieved_concept, is_correct)
            )

    # --- RENDER 4x10 VISUAL GRID ---
    n_rows = len(selected_prompts)
    n_cols = len(pruning_levels)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.8 * n_cols, 3.2 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for r_idx, prompt_concept in enumerate(selected_prompts):
        records = retrieval_grid_data[prompt_concept]

        for c_idx, (p_level, img_path, ret_concept, is_correct) in enumerate(records):
            ax = axes[r_idx, c_idx]
            img = load_image_safely(img_path)

            if img is not None:
                ax.imshow(img)
            else:
                ax.set_facecolor("#e0e0e0")
                ax.text(
                    0.5,
                    0.5,
                    f"Retrieved:\n{ret_concept}\n(Image File\nNot Found)",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color="black",
                )

            ax.set_xticks([])
            ax.set_yticks([])

            # Compact titles for 10-column layout
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
                fontsize=8.5,
                color=title_color,
                fontweight="bold",
                pad=4,
                bbox=dict(boxstyle="round,pad=0.2", facecolor=box_color, edgecolor=title_color, lw=0.8),
            )

            # Column Headers (Pruning Levels)
            if r_idx == 0:
                ax.set_xlabel(
                    f"{p_level * 100:.0f}%",
                    fontsize=11,
                    fontweight="bold",
                    labelpad=8,
                )
                ax.xaxis.set_label_position("top")

            # Row Labels (Text Prompts)
            if c_idx == 0:
                ax.set_ylabel(
                    f"Prompt:\n\"{prompt_concept}\"",
                    fontsize=10.5,
                    fontweight="bold",
                    rotation=0,
                    labelpad=45,
                    ha="right",
                    va="center",
                )

    plt.suptitle(
        "Bozeat 'Draw from Prompt' Task: Visual Retrieval Trajectory Across Atrophy Increments (0% to 85%)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    save_path = os.path.join(output_dir, "bozeat_retrieved_images_extended.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\n[+] Success! Extended visual Bozeat grid saved to:\n    {save_path}")
    return save_path