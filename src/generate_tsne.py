import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
import torch


def generate_joint_hierarchical_tsne(
    evaluator,
    pruning_levels=[0.0, 0.25, 0.50, 0.75, 0.90, 0.95],
    output_dir="./data/results",
):
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, p_level in enumerate(pruning_levels):
        pruned_model = evaluator._apply_pruning(evaluator.base_model, p_level)
        (
            img_feats,
            text_feats,
            labels,
            unique_concepts,
            eval_meta,
        ) = evaluator._extract_joint_features(pruned_model)

        combined = torch.cat([img_feats, text_feats], dim=0).cpu().numpy()

        super_labels = (
            eval_meta["superordinate"].tolist()
            if "superordinate" in eval_meta.columns
            else labels
        )
        text_labels = ["Text Prompt"] * len(unique_concepts)
        categories = super_labels + text_labels
        modalities = ["Image"] * len(img_feats) + ["Text"] * len(text_feats)

        tsne = TSNE(
            n_components=2,
            perplexity=min(30, max(5, len(combined) // 5)),
            random_state=42,
        )
        emb_2d = tsne.fit_transform(combined)

        df_tsne = pd.DataFrame(
            {
                "t-SNE 1": emb_2d[:, 0],
                "t-SNE 2": emb_2d[:, 1],
                "Category": categories,
                "Modality": modalities,
            }
        )

        ax = axes[idx]
        sns.scatterplot(
            data=df_tsne,
            x="t-SNE 1",
            y="t-SNE 2",
            hue="Category",
            style="Modality",
            ax=ax,
            alpha=0.8,
            s=45,
            palette="tab10",
        )
        ax.set_title(
            f"Pruning: {int(p_level * 100)}%", fontsize=11, fontweight="bold"
        )
        ax.grid(True, linestyle="--", alpha=0.3)

        if idx != len(pruning_levels) - 1:
            if ax.get_legend():
                ax.get_legend().remove()
        else:
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

    plt.suptitle(
        "Full Dataset Joint Feature Space (Image & Text) Trajectory Under Pruning",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()

    save_path = os.path.join(output_dir, "joint_tsne_trajectory.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved entire dataset joint t-SNE trajectory to: {save_path}")