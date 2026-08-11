import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch


def _normalize_columns(df):
    df_copy = df.copy()
    df_copy.columns = [str(col).lower() for col in df_copy.columns]
    return df_copy


def plot_category_breakdown_suite(df, output_dir="./data/results"):
    os.makedirs(output_dir, exist_ok=True)
    df = _normalize_columns(df)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    prune_pcts = [p * 100 for p in df["pruning_level"]]

    # Empirically Measured Curves
    if "i2t_top1" in df.columns:
        ax1.plot(
            prune_pcts,
            df["i2t_top1"],
            marker="o",
            color="#1f77b4",
            linewidth=2,
            label="Empirical Top-1 Acc",
        )
    if "cka_vision" in df.columns:
        ax1.plot(
            prune_pcts,
            df["cka_vision"],
            marker="^",
            color="#2ca02c",
            linewidth=2,
            label="Vision CKA",
        )

    # Theoretical Expected Baseline Overlay (Semantic Dementia Collapse Bounds)
    exp_decay = [1.0 * (1.0 - (p / 100.0) ** 2) for p in prune_pcts]
    ax1.plot(
        prune_pcts,
        exp_decay,
        linestyle="--",
        color="gray",
        alpha=0.7,
        label="Expected Theory Bound",
    )

    ax1.set_xlabel("Pruning Level (%)")
    ax1.set_ylabel("Score / Similarity")
    ax1.set_title("Empirical Metrics vs Expected Theoretical Bound")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower left", frameon=True)

    # Error Taxonomy Heatmap
    err_cols = [
        c
        for c in [
            "coordinate error",
            "superordinate error",
            "domain error",
            "domain collapse",
        ]
        if c in df.columns
    ]
    if err_cols:
        heatmap_data = df[err_cols].T
        heatmap_data.columns = [f"{int(p * 100)}%" for p in df["pruning_level"]]
        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt="d",
            cmap="YlOrRd",
            ax=ax2,
            cbar=True,
        )
        ax2.set_title("Clinical Taxonomy Error Counts")
        ax2.set_xlabel("Pruning Level (%)")

    plt.tight_layout()
    save_path = os.path.join(output_dir, "category_breakdown_suite.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Saved category breakdown suite to: {save_path}")


def plot_hierarchical_breakdown_suite(df, output_dir="./data/results"):
    os.makedirs(output_dir, exist_ok=True)
    df = _normalize_columns(df)

    plt.figure(figsize=(8, 5))
    prune_pcts = [p * 100 for p in df["pruning_level"]]

    # Measured
    for col, color in [
        ("top1_specific_acc", "#d62728"),
        ("top1_basic_acc", "#ff7f0e"),
        ("top1_super_acc", "#2ca02c"),
    ]:
        if col in df.columns:
            name = col.replace("top1_", "").replace("_acc", "").capitalize()
            plt.plot(
                prune_pcts,
                df[col],
                marker="o",
                linewidth=2,
                color=color,
                label=f"Empirical {name}",
            )

    # Expected Fine-to-Coarse Decay Profiles
    exp_spec = [1.0 - (p / 100.0) ** 1.5 for p in prune_pcts]
    exp_super = [1.0 - 0.3 * (p / 100.0) ** 3 for p in prune_pcts]

    plt.plot(
        prune_pcts,
        exp_spec,
        "--",
        color="#d62728",
        alpha=0.4,
        label="Expected Specific Decay",
    )
    plt.plot(
        prune_pcts,
        exp_super,
        "--",
        color="#2ca02c",
        alpha=0.4,
        label="Expected Superordinate Decay",
    )

    plt.xlabel("Pruning Level (%)")
    plt.ylabel("Accuracy")
    plt.title("Hierarchical Concept Decay: Empirical vs Expected Fine-to-Coarse")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower left", frameon=True)
    plt.tight_layout()

    save_path = os.path.join(output_dir, "hierarchical_breakdown_suite.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Saved hierarchical breakdown suite to: {save_path}")


def plot_signal_noise_distribution_shift(
    evaluator,
    pruning_levels=[0.0, 0.50, 0.75, 0.95],
    output_dir="./data/results",
):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 5))

    ref_img, ref_text, labels, concepts, _ = evaluator._extract_joint_features(
        evaluator.base_model
    )
    c_to_i = {c: i for i, c in enumerate(concepts)}
    targets = torch.tensor([c_to_i[c] for c in labels])

    for p in pruning_levels:
        p_model = evaluator._apply_pruning(evaluator.base_model, p)
        p_img, p_text, _, _, _ = evaluator._extract_joint_features(p_model)
        sims = torch.matmul(p_img, p_text.T)
        target_sims = sims[torch.arange(len(targets)), targets].numpy()
        sns.kdeplot(
            target_sims, label=f"Pruned {int(p*100)}%", fill=True, alpha=0.2
        )

    plt.title("Target Cosine Similarity Distribution Shift Across Pruning")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(output_dir, "signal_noise_distribution_shift.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Saved distribution shift plot to: {save_path}")


def plot_target_rank_waterfall(
    evaluator,
    pruning_levels=[0.0, 0.25, 0.50, 0.75, 0.90, 0.95],
    output_dir="./data/results",
):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 5))

    _, _, labels, concepts, _ = evaluator._extract_joint_features(
        evaluator.base_model
    )
    c_to_i = {c: i for i, c in enumerate(concepts)}
    targets = torch.tensor([c_to_i[c] for c in labels])

    all_ranks = []
    for p in pruning_levels:
        p_model = evaluator._apply_pruning(evaluator.base_model, p)
        p_img, p_text, _, _, _ = evaluator._extract_joint_features(p_model)
        sims = torch.matmul(p_img, p_text.T)

        ranks = [
            (
                (torch.argsort(sims[i], descending=True) == targets[i])
                .nonzero(as_tuple=True)[0]
                .item()
                + 1
            )
            for i in range(len(targets))
        ]
        all_ranks.append(ranks)

    plt.boxplot(all_ranks, labels=[f"{int(p*100)}%" for p in pruning_levels])
    plt.title("Target Concept Rank Waterfall Across Pruning Levels")
    plt.xlabel("Pruning Level (%)")
    plt.ylabel("Target Retrieval Rank (Lower is Better)")
    plt.yscale("log")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    save_path = os.path.join(output_dir, "target_rank_waterfall.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Saved target rank waterfall plot to: {save_path}")


def plot_concept_retrieval_heatmap(
    evaluator,
    pruning_levels=[0.0, 0.25, 0.50, 0.75, 0.90, 0.95],
    output_dir="./data/results",
):
    os.makedirs(output_dir, exist_ok=True)

    _, _, labels, concepts, _ = evaluator._extract_joint_features(
        evaluator.base_model
    )
    c_to_i = {c: i for i, c in enumerate(concepts)}
    targets = torch.tensor([c_to_i[c] for c in labels])

    c_accs = {c: [] for c in concepts}
    for p in pruning_levels:
        p_model = evaluator._apply_pruning(evaluator.base_model, p)
        p_img, p_text, _, _, _ = evaluator._extract_joint_features(p_model)
        sims = torch.matmul(p_img, p_text.T)
        preds = torch.argmax(sims, dim=1)

        for c in concepts:
            mask = targets == c_to_i[c]
            acc = (
                (preds[mask] == targets[mask]).float().mean().item()
                if mask.sum() > 0
                else 0.0
            )
            c_accs[c].append(acc)

    df_heat = pd.DataFrame(
        c_accs, index=[f"{int(p*100)}%" for p in pruning_levels]
    ).T

    plt.figure(figsize=(10, max(6, len(concepts) * 0.35)))
    sns.heatmap(df_heat, annot=True, cmap="viridis", vmin=0.0, vmax=1.0)
    plt.title("Per-Concept Accuracy Breakdown Across Pruning Levels")
    plt.xlabel("Pruning Level (%)")
    plt.ylabel("Concept")
    plt.tight_layout()

    save_path = os.path.join(output_dir, "concept_retrieval_heatmap.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Saved concept retrieval heatmap to: {save_path}")