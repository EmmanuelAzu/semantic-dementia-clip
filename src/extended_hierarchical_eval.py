import os
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

try:
    from src.joint_evaluator import JointSpaceEvaluator
except ModuleNotFoundError:
    from joint_evaluator import JointSpaceEvaluator


def _get_column_mappings(metadata):
    """Resolve taxonomy column names across varying dataset schema formats."""
    spec_col = next(
        (c for c in ["specific", "concept", "label"] if c in metadata.columns),
        "specific",
    )
    coord_col = next(
        (c for c in ["coordinate", "category"] if c in metadata.columns),
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


def select_common_simple_concepts(metadata, n_target=15):
    """Select common, easily recognizable objects evenly distributed across coordinate and domain levels."""
    spec_col, coord_col, super_col = _get_column_mappings(metadata)

    # Priority list of simple, highly familiar objects across living and non-living categories
    preferred_objects = [
        "chihuahua", "egyptian_cat", "african_elephant", "goldfish", "toucan",
        "lemon", "bell_pepper", "mushroom", "convertible_car", "motorcycle",
        "airliner", "hammer", "chair", "wall_clock", "water_bottle",
        "tabby_cat", "golden_retriever", "sports_car", "banana", "bathtub",
        "sports car", "tabby cat", "golden retriever", "wall clock", "water bottle"
    ]

    all_concepts = list(metadata[spec_col].dropna().unique())
    matched_preferred = [c for c in preferred_objects if c in all_concepts]

    if len(matched_preferred) >= n_target:
        return sorted(matched_preferred[:n_target])

    # Fallback to balanced sampling across distinct coordinate categories
    selected = set(matched_preferred)
    if coord_col in metadata.columns:
        coords = metadata[coord_col].unique()
        for cd in coords:
            cd_concepts = metadata[metadata[coord_col] == cd][spec_col].unique()
            for c in cd_concepts:
                if c not in selected:
                    selected.add(c)
                    break
                if len(selected) == n_target:
                    break
            if len(selected) == n_target:
                break

    if len(selected) < n_target:
        for c in all_concepts:
            if c not in selected:
                selected.add(c)
            if len(selected) == n_target:
                break

    return sorted(list(selected)[:n_target])


def generate_individual_confusion_matrices(
    evaluator,
    pruning_levels=[0.0, 0.05, 0.15, 0.35, 0.55, 0.70, 0.80, 0.85],
    n_classes=15,
    output_dir="./data/results/confusion_matrices",
):
    """Generate and save separate individual confusion matrix files for each pruning level."""
    print(f"[*] Generating separate confusion matrix plots for {len(pruning_levels)} pruning levels...")
    os.makedirs(output_dir, exist_ok=True)
    meta = evaluator.metadata.reset_index(drop=True)
    spec_col, _, _ = _get_column_mappings(meta)

    sampled_concepts = select_common_simple_concepts(meta, n_target=n_classes)
    print(f"[*] Selected {len(sampled_concepts)} common simple objects:\n    {sampled_concepts}")

    for p_level in pruning_levels:
        pruned_model = evaluator._apply_pruning(evaluator.base_model, p_level)
        img_feats, text_feats, labels, unique_concepts, _ = (
            evaluator._extract_joint_features(pruned_model)
        )

        sim_matrix = torch.matmul(img_feats, text_feats.T)
        top1_preds = torch.argmax(sim_matrix, dim=1).cpu().numpy()

        true_concepts = np.array(labels)
        pred_concepts = np.array([unique_concepts[p] for p in top1_preds])

        mask = np.isin(true_concepts, sampled_concepts)
        sub_true = true_concepts[mask]
        sub_pred = pred_concepts[mask]

        cm = confusion_matrix(
            sub_true, sub_pred, labels=sampled_concepts, normalize="true"
        )

        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(
            cm,
            annot=True,
            fmt=".2f",
            cmap="YlGnBu",
            xticklabels=sampled_concepts,
            yticklabels=sampled_concepts,
            cbar=True,
            vmin=0.0,
            vmax=1.0,
            linewidths=0.5,
            linecolor="whitesmoke"
        )

        plt.title(
            f"Normalized Confusion Matrix ({n_classes} Common Objects) @ {p_level * 100:.0f}% Pruning",
            fontsize=13,
            fontweight="bold",
            pad=12
        )
        plt.xlabel("Predicted Object", fontsize=11, fontweight="bold")
        plt.ylabel("True Target Object", fontsize=11, fontweight="bold")
        plt.xticks(rotation=45, ha="right", fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        plt.tight_layout()

        file_name = f"confusion_matrix_pruning_{int(round(p_level * 100)):02d}pct.png"
        save_path = os.path.join(output_dir, file_name)
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f" [+] Saved individual confusion matrix: {save_path}")


def plot_hierarchical_confusion_matrices(
    evaluator, pruning_levels=[0.0, 0.05, 0.15, 0.35, 0.55, 0.70, 0.80, 0.85]
):
    """Compute and plot row-normalized confusion matrices across Category, Coordinate,
    and Superordinate taxonomy levels across all pruning thresholds.
    """
    print("[*] Generating Hierarchical Multi-Level Confusion Matrices...")
    meta = evaluator.metadata.reset_index(drop=True)
    spec_col, coord_col, super_col = _get_column_mappings(meta)

    output_dir = os.path.join("data", "results", "hierarchical_confusion")
    os.makedirs(output_dir, exist_ok=True)

    for p_level in pruning_levels:
        pruned_model = evaluator._apply_pruning(evaluator.base_model, p_level)
        img_feats, text_feats, labels, unique_concepts, _ = (
            evaluator._extract_joint_features(pruned_model)
        )

        sim_matrix = torch.matmul(img_feats, text_feats.T)
        preds_idx = torch.argmax(sim_matrix, dim=1).numpy()

        true_concepts = np.array(labels)
        pred_concepts = np.array([unique_concepts[p] for p in preds_idx])

        concept_map = meta.drop_duplicates(spec_col).set_index(spec_col)
        true_coords = meta[coord_col].values
        pred_coords = np.array(
            [concept_map.loc[c, coord_col] for c in pred_concepts]
        )

        true_supers = meta[super_col].values
        pred_supers = np.array(
            [concept_map.loc[c, super_col] for c in pred_concepts]
        )

        fig, axes = plt.subplots(1, 3, figsize=(22, 6))
        fig.suptitle(
            f"Hierarchical Confusion Density @ {p_level * 100:.0f}% Joint Atrophy",
            fontsize=16,
            fontweight="bold",
        )

        levels_data = [
            ("Category (Leaf)", true_concepts, pred_concepts, axes[0]),
            ("Coordinate", true_coords, pred_coords, axes[1]),
            ("Superordinate (Domain)", true_supers, pred_supers, axes[2]),
        ]

        for title, true_vals, pred_vals, ax in levels_data:
            classes = sorted(list(set(true_vals)))
            cm = confusion_matrix(
                true_vals, pred_vals, labels=classes, normalize="true"
            )

            sns.heatmap(
                cm,
                annot=len(classes) <= 15,
                fmt=".2f",
                cmap="YlOrRd",
                xticklabels=classes,
                yticklabels=classes,
                ax=ax,
                cbar=True,
            )
            ax.set_title(f"{title} Level", fontsize=12, fontweight="bold")
            ax.set_xlabel("Predicted Class")
            ax.set_ylabel("True Class")
            ax.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        save_path = os.path.join(
            output_dir,
            f"confusion_matrices_pruning_{int(round(p_level * 100)):02d}pct.png",
        )
        plt.savefig(save_path, dpi=300)
        plt.close()
    print(f"[+] Hierarchical confusion matrices saved to: {output_dir}")


def generate_hierarchical_hsv_tsne(
    evaluator, pruning_levels=[0.0, 0.05, 0.15, 0.35, 0.55, 0.70, 0.80, 0.85]
):
    """Plot t-SNE trajectories using HSV-derived color mapping."""
    print("[*] Generating Hierarchical HSV-Color Shaded t-SNE Plots...")
    meta = evaluator.metadata.reset_index(drop=True)
    spec_col, coord_col, super_col = _get_column_mappings(meta)

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

    n_plots = len(pruning_levels)
    n_cols = 4
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4.5 * n_rows))
    axes = axes.flatten()

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


def compute_and_plot_top10_rank_taxonomy(
    evaluator, pruning_levels=[0.0, 0.05, 0.15, 0.35, 0.55, 0.70, 0.80, 0.85]
):
    """Evaluate Top-10 predictions and measure rank degradation across taxonomy tiers."""
    print("[*] Evaluating Top-10 Rank-Aware Taxonomic Breakdown...")
    meta = evaluator.metadata.reset_index(drop=True)
    spec_col, coord_col, super_col = _get_column_mappings(meta)

    concept_map = meta.drop_duplicates(spec_col).set_index(spec_col)
    tier_results = []

    for p_level in pruning_levels:
        pruned_model = evaluator._apply_pruning(evaluator.base_model, p_level)
        img_feats, text_feats, labels, unique_concepts, _ = (
            evaluator._extract_joint_features(pruned_model)
        )

        sim_matrix = torch.matmul(img_feats, text_feats.T)
        top10_k = min(10, sim_matrix.shape[1])
        top10_indices = (
            torch.topk(sim_matrix, k=top10_k, dim=1).indices.cpu().numpy()
        )

        exact_cat, coord_match, super_match, domain_collapse = 0, 0, 0, 0
        total_evals = len(labels) * top10_k

        for i, target_c in enumerate(labels):
            target_co = concept_map.loc[target_c, coord_col]
            target_su = concept_map.loc[target_c, super_col]

            for pred_idx in top10_indices[i]:
                pred_c = unique_concepts[pred_idx]
                pred_co = concept_map.loc[pred_c, coord_col]
                pred_su = concept_map.loc[pred_c, super_col]

                if pred_c == target_c:
                    exact_cat += 1
                elif pred_co == target_co:
                    coord_match += 1
                elif pred_su == target_su:
                    super_match += 1
                else:
                    domain_collapse += 1

        tier_results.append(
            {
                "pruning_level": p_level,
                "Exact Category": exact_cat / total_evals,
                "Coordinate Match": coord_match / total_evals,
                "Superordinate Match": super_match / total_evals,
                "Domain Collapse": domain_collapse / total_evals,
            }
        )

    df_top10 = pd.DataFrame(tier_results)

    plt.figure(figsize=(10, 6))
    plt.plot(
        df_top10["pruning_level"] * 100,
        df_top10["Exact Category"],
        "o-",
        label="Exact Category",
        linewidth=2,
    )
    plt.plot(
        df_top10["pruning_level"] * 100,
        df_top10["Coordinate Match"],
        "s--",
        label="Coordinate Match",
        linewidth=2,
    )
    plt.plot(
        df_top10["pruning_level"] * 100,
        df_top10["Superordinate Match"],
        "^--",
        label="Superordinate Match",
        linewidth=2,
    )
    plt.plot(
        df_top10["pruning_level"] * 100,
        df_top10["Domain Collapse"],
        "x-.",
        label="Domain Collapse",
        linewidth=2,
    )

    plt.title(
        "Top-10 Logit Rank Taxonomic Tier Distribution Across Atrophy Levels",
        fontsize=13,
        fontweight="bold",
    )
    plt.xlabel("Joint Projection Atrophy Level (%)")
    plt.ylabel("Proportion of Top-10 Retrieved Candidates")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(
        "data", "results", "top10_rank_taxonomy_decay.png"
    )
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Top-10 rank taxonomy breakdown plot saved to: {save_path}")
    return df_top10