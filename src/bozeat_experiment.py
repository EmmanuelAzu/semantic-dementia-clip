import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch

# Selected 5-stage pruning schedule
PRUNING_LEVELS_5 = [0.00, 0.15, 0.35, 0.60, 0.85]


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


def run_bozeat_visual_grid_5levels(
    evaluator,
    pruning_levels=PRUNING_LEVELS_5,
    target_prompts=["Tabby Cat", "Golden Retriever", "Sports Car"],
    output_dir="./data/results/bozeat_experiment",
):
    """Executes Bozeat et al. 'Draw from Prompt' task and generates a 5-level visual grid

    showing actual retrieved image files for 3 text prompts.
    """
    print("=" * 70)
    print(" RUNNING BOZEAT ET AL. 'DRAW FROM PROMPT' VISUAL RETRIEVAL (5 LEVELS) ")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)
    meta = evaluator.metadata.reset_index(drop=True)
    spec_col, coord_col, super_col = _get_column_mappings(meta)
    concept_lookup = meta.drop_duplicates(spec_col).set_index(spec_col)

    # Filter target prompts available in metadata
    available_concepts = list(meta[spec_col].unique())
    selected_prompts = [p for p in target_prompts if p in available_concepts]

    # Fallback selection if prompts are named slightly differently
    if len(selected_prompts) < 3:
        for c in available_concepts:
            if c not in selected_prompts:
                selected_prompts.append(c)
            if len(selected_prompts) == 3:
                break

    print(f"[*] Target prompts selected for visual drawing grid:\n    {selected_prompts}")

    # Data structure: {prompt: [(pruning_level, retrieved_image_path, retrieved_concept_name, is_correct)]}
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
            prompt_vec = text_feats[concept_idx].unsqueeze(0)  # Shape: [1, D]

            # Compute similarity of target prompt against all dataset images
            sims = torch.matmul(img_feats, prompt_vec.T).squeeze(1)  # Shape: [N_images]
            best_img_idx = torch.argmax(sims).item()

            retrieved_row = eval_meta.iloc[best_img_idx]
            retrieved_concept = retrieved_row[spec_col]

            # Find image path column
            path_col = next(
                (c for c in ["filepath", "filename", "image_path", "path", "file_path"] if c in retrieved_row.index),
                None,
            )
            img_path = retrieved_row[path_col] if path_col else ""

            is_correct = (retrieved_concept == prompt_concept)
            retrieval_grid_data[prompt_concept].append(
                (p_level, img_path, retrieved_concept, is_correct)
            )

    # --- RENDER 3x5 VISUAL GRID ---
    n_rows = len(selected_prompts)
    n_cols = len(pruning_levels)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.8 * n_cols, 4.0 * n_rows))
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
                # Render placeholder if image file is not on disk
                ax.set_facecolor("#e0e0e0")
                ax.text(
                    0.5,
                    0.5,
                    f"Retrieved:\n{ret_concept}\n(Image File\nNot Found)",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="black",
                )

            ax.set_xticks([])
            ax.set_yticks([])

            # Title styling: Green for exact target, Red/Orange for error
            if is_correct:
                title_text = f"Retrieved:\n'{ret_concept}' ✓"
                title_color = "darkgreen"
                box_color = "#e6f4ea"
            else:
                title_text = f"Retrieved:\n'{ret_concept}' ✗"
                title_color = "darkred"
                box_color = "#fce8e6"

            ax.set_title(
                title_text,
                fontsize=10,
                color=title_color,
                fontweight="bold",
                pad=6,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=box_color, edgecolor=title_color, lw=1),
            )

            # Column Headers (Pruning Level)
            if r_idx == 0:
                ax.set_xlabel(
                    f"Atrophy: {p_level * 100:.0f}%",
                    fontsize=12,
                    fontweight="bold",
                    labelpad=10,
                )
                ax.xaxis.set_label_position("top")

            # Row Labels (Text Prompt Query)
            if c_idx == 0:
                ax.set_ylabel(
                    f"Prompt:\n\"{prompt_concept}\"",
                    fontsize=12,
                    fontweight="bold",
                    rotation=0,
                    labelpad=50,
                    ha="right",
                    va="center",
                )

    plt.suptitle(
        "Bozeat 'Draw from Prompt' Task: Nearest Neighbor Retrieved Images Across 5 Atrophy Stages",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    save_path = os.path.join(output_dir, "bozeat_retrieved_images_5levels.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\n[+] Success! Visual Bozeat grid saved to:\n    {save_path}")
    return save_path