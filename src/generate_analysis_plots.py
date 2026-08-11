import copy
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.joint_evaluator import JointSpaceEvaluator
from src.pruning_engine import CLIPPruningEngine


def plot_category_breakdown_suite(df, output_path=None):
    if output_path is None:
        output_path = os.path.join(
            PROJECT_ROOT, "data", "results", "category_decay_analysis.png"
        )

    cat_cols = [c for c in df.columns if c.startswith("acc_")]
    cat_names = [c.replace("acc_", "") for c in cat_cols]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(cat_cols))))
    for idx, (col, name) in enumerate(zip(cat_cols, cat_names)):
        ax1.plot(
            df["pruning_level"] * 100,
            df[col] * 100,
            "o-",
            label=name,
            color=colors[idx % len(colors)],
            linewidth=2,
        )

    ax1.set_xlabel(
        "Joint Projection Atrophy Level (%)", fontsize=11, fontweight="bold"
    )
    ax1.set_ylabel("Top-1 Accuracy (%)", fontsize=11, fontweight="bold")
    ax1.set_title(
        "Per-Category Retrieval Accuracy Trajectories",
        fontsize=13,
        fontweight="bold",
    )
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="lower left", frameon=True, facecolor="white", fontsize=9)

    heatmap_data = df[cat_cols].T * 100
    heatmap_data.index = cat_names
    heatmap_data.columns = [f"{int(p*100)}%" for p in df["pruning_level"]]

    sns.heatmap(
        heatmap_data,
        ax=ax2,
        cmap="YlGnBu",
        annot=True,
        fmt=".1f",
        cbar_kws={"label": "Top-1 Acc (%)"},
    )
    ax2.set_xlabel("Atrophy Level", fontsize=11, fontweight="bold")
    ax2.set_title(
        "Domain Vulnerability Matrix", fontsize=13, fontweight="bold"
    )

    plt.suptitle(
        "Superordinate Domain Sensitivity Under Joint Projection Atrophy",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Category degradation plot saved to:\n    {output_path}")


def plot_hierarchical_breakdown_suite(df, output_path=None):
    if output_path is None:
        output_path = os.path.join(
            PROJECT_ROOT, "data", "results", "hierarchical_error_analysis.png"
        )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.plot(
        df["pruning_level"] * 100,
        df["top1_specific_acc"] * 100,
        "o-",
        color="#2ca02c",
        linewidth=2.5,
        label="Specific Top-1 Acc (Exact Concept)",
    )
    ax1.plot(
        df["pruning_level"] * 100,
        df["top1_basic_acc"] * 100,
        "s--",
        color="#1f77b4",
        linewidth=2,
        label="Basic Top-1 Acc (Category)",
    )
    ax1.plot(
        df["pruning_level"] * 100,
        df["top1_domain_acc"] * 100,
        "^--",
        color="#ff7f0e",
        linewidth=2,
        label="Domain Top-1 Acc (Superordinate)",
    )

    ax1.set_xlabel(
        "Joint Projection Atrophy Level (%)", fontsize=11, fontweight="bold"
    )
    ax1.set_ylabel("Accuracy (%)", fontsize=11, fontweight="bold")
    ax1.set_title(
        "Hierarchical Abstraction Decay Trajectories",
        fontsize=13,
        fontweight="bold",
    )
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="lower left", facecolor="white")

    atrophy_pct = df["pruning_level"] * 100
    superordinate_pct = df["pct_superordinate_errors"] * 100
    domain_collapse_pct = df["pct_domain_collapse_errors"] * 100

    ax2.stackplot(
        atrophy_pct,
        superordinate_pct,
        domain_collapse_pct,
        labels=[
            "Superordinate Errors (Within-Domain)",
            "Domain Collapse Errors (Cross-Domain)",
        ],
        colors=["#1f77b4", "#d62728"],
        alpha=0.75,
    )

    ax2.set_xlabel(
        "Joint Projection Atrophy Level (%)", fontsize=11, fontweight="bold"
    )
    ax2.set_ylabel(
        "Proportion of Total Errors (%)", fontsize=11, fontweight="bold"
    )
    ax2.set_title(
        "Taxonomic Error Composition Shift", fontsize=13, fontweight="bold"
    )
    ax2.set_ylim(0, 100)
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc="upper left", facecolor="white")

    plt.suptitle(
        "Hierarchical Concept Decay & Semantic Degradation Dynamics",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(
        f"[+] Hierarchical error analysis plot saved to:\n    {output_path}"
    )


def plot_signal_noise_distribution_shift(
    evaluator, pruning_levels=[0.0, 0.50, 0.75, 0.95], output_path=None
):
    if output_path is None:
        output_path = os.path.join(
            PROJECT_ROOT,
            "data",
            "results",
            "similarity_distribution_shift.png",
        )

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    base_model = evaluator.base_model

    for idx, p in enumerate(pruning_levels):
        ax = axes[idx]
        model_copy = copy.deepcopy(base_model)
        engine = CLIPPruningEngine(model_copy)
        pruned_model = engine.get_pruned_model(
            amount=float(p), encoder_type="joint"
        )

        (
            img_feats,
            text_feats,
            concept_labels,
            unique_concepts,
        ) = evaluator._extract_joint_features(pruned_model)
        sim_matrix = (img_feats @ text_feats.T).float().cpu().numpy()

        concept_to_idx = {c: i for i, c in enumerate(unique_concepts)}
        targets = np.array([concept_to_idx[c] for c in concept_labels])

        n_samples = len(targets)
        pos_sims = sim_matrix[np.arange(n_samples), targets]

        neg_mask = np.ones_like(sim_matrix, dtype=bool)
        neg_mask[np.arange(n_samples), targets] = False
        neg_sims = sim_matrix[neg_mask]

        pos_var = float(np.var(pos_sims)) if len(pos_sims) > 0 else 0.0
        neg_var = float(np.var(neg_sims)) if len(neg_sims) > 0 else 0.0

        if pos_var > 1e-8:
            sns.kdeplot(
                pos_sims,
                ax=ax,
                color="green",
                fill=True,
                label="Matching Pairs (Signal)",
                alpha=0.4,
                linewidth=2,
                warn_singular=False,
            )
        else:
            ax.axvline(
                float(np.mean(pos_sims)),
                color="green",
                linestyle="--",
                linewidth=2,
                label=f"Signal Point Mass ({np.mean(pos_sims):.3f})",
            )

        if neg_var > 1e-8:
            sns.kdeplot(
                neg_sims,
                ax=ax,
                color="red",
                fill=True,
                label="Non-Matching Pairs (Noise)",
                alpha=0.3,
                linewidth=2,
                warn_singular=False,
            )
        else:
            ax.axvline(
                float(np.mean(neg_sims)),
                color="red",
                linestyle=":",
                linewidth=2,
                label=f"Noise Point Mass ({np.mean(neg_sims):.3f})",
            )

        ax.set_title(
            f"Atrophy Level: {int(p * 100)}%", fontsize=12, fontweight="bold"
        )
        ax.set_xlabel("Cosine Similarity", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)

    plt.suptitle(
        "Signal-to-Noise Distribution Collapse in Joint Embedding Space",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Similarity distribution plot saved to:\n    {output_path}")


def plot_target_rank_waterfall(
    evaluator, pruning_levels=[0.0, 0.25, 0.50, 0.75, 0.90, 0.95], output_path=None
):
    if output_path is None:
        output_path = os.path.join(
            PROJECT_ROOT, "data", "results", "retrieval_rank_waterfall.png"
        )

    rank_data = []
    base_model = evaluator.base_model

    for p in pruning_levels:
        model_copy = copy.deepcopy(base_model)
        engine = CLIPPruningEngine(model_copy)
        pruned_model = engine.get_pruned_model(
            amount=float(p), encoder_type="joint"
        )

        (
            img_feats,
            text_feats,
            concept_labels,
            unique_concepts,
        ) = evaluator._extract_joint_features(pruned_model)
        sim_matrix = (img_feats @ text_feats.T).float().cpu().numpy()

        concept_to_idx = {c: i for i, c in enumerate(unique_concepts)}
        targets = np.array([concept_to_idx[c] for c in concept_labels])

        sorted_indices = np.argsort(-sim_matrix, axis=1)
        ranks = np.where(sorted_indices == targets[:, None])[1] + 1

        atrophy_level = int(p * 100)
        for r in ranks:
            rank_data.append({"Atrophy (%)": atrophy_level, "Rank": r})

    rank_df = pd.DataFrame(rank_data)

    plt.figure(figsize=(12, 6))
    sns.boxplot(
        x="Atrophy (%)",
        y="Rank",
        hue="Atrophy (%)",
        data=rank_df,
        palette="Reds",
        showmeans=True,
        legend=False,
    )
    plt.yscale("log")
    plt.title(
        "Target Retrieval Rank Drift Under Joint Projection Head Atrophy",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    plt.xlabel(
        "Joint Projection Atrophy Level (%)", fontsize=11, fontweight="bold"
    )
    plt.ylabel(
        "Target Concept Retrieval Rank (Log Scale)",
        fontsize=11,
        fontweight="bold",
    )
    plt.grid(True, which="both", linestyle="--", alpha=0.3)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Target rank waterfall plot saved to:\n    {output_path}")


def plot_concept_retrieval_heatmap(
    evaluator, pruning_levels=[0.0, 0.25, 0.50, 0.75, 0.90, 0.95], output_path=None
):
    if output_path is None:
        output_path = os.path.join(
            PROJECT_ROOT, "data", "results", "concept_retrieval_heatmap.png"
        )

    base_model = evaluator.base_model
    concept_acc_by_level = {}

    for p in pruning_levels:
        model_copy = copy.deepcopy(base_model)
        engine = CLIPPruningEngine(model_copy)
        pruned_model = engine.get_pruned_model(amount=float(p), encoder_type="joint")

        img_feats, text_feats, concept_labels, unique_concepts = evaluator._extract_joint_features(pruned_model)
        sim_matrix = (img_feats @ text_feats.T).float().cpu().numpy()

        concept_to_idx = {c: i for i, c in enumerate(unique_concepts)}
        targets = np.array([concept_to_idx[c] for c in concept_labels])
        preds = np.argmax(sim_matrix, axis=1)

        p_accs = {}
        for c in unique_concepts:
            indices = np.where(np.array(concept_labels) == c)[0]
            if len(indices) > 0:
                acc = np.mean(preds[indices] == targets[indices])
                p_accs[c] = acc * 100

        concept_acc_by_level[f"{int(p*100)}%"] = pd.Series(p_accs)

    heatmap_df = pd.DataFrame(concept_acc_by_level)

    plt.figure(figsize=(10, max(8, len(heatmap_df) * 0.25)))
    sns.heatmap(
        heatmap_df,
        cmap="YlOrRd_r",
        annot=True,
        fmt=".0f",
        linewidths=0.5,
        cbar_kws={"label": "Top-1 Accuracy (%)"},
    )
    plt.title("Specific Object Retrieval Dynamics Across Atrophy Levels", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Joint Projection Atrophy Level (%)", fontsize=11, fontweight="bold")
    plt.ylabel("Specific Object Concept", fontsize=11, fontweight="bold")
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Object retrieval heatmap saved to:\n    {output_path}")


if __name__ == "__main__":
    evaluator = JointSpaceEvaluator()
    results_df = evaluator.run_eval()

    plot_category_breakdown_suite(results_df)
    plot_hierarchical_breakdown_suite(results_df)
    plot_signal_noise_distribution_shift(evaluator)
    plot_target_rank_waterfall(evaluator)
    plot_concept_retrieval_heatmap(evaluator)