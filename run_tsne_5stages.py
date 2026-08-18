import torch
from src.generate_tsne_subcategories import (
    PRUNING_LEVELS_5PCT,
    generate_tsne_grid_with_key,
)
from src.joint_evaluator import JointSpaceEvaluator


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Executing 18-stage t-SNE grid pipeline on: {device}")

    evaluator = JointSpaceEvaluator(
        metadata_path="./data/processed/metadata_processed.csv",
        model_name="ViT-B/32",
        device=device,
        batch_size=32,
        balance_taxonomically=True,
    )

    generate_tsne_grid_with_key(
        evaluator,
        pruning_levels=PRUNING_LEVELS_5PCT,
        samples_per_class=15,
        n_cols=6,
    )


if __name__ == "__main__":
    main()