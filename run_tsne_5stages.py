import torch
from src.generate_tsne_subcategories import generate_tsne_5levels_with_key
from src.joint_evaluator import JointSpaceEvaluator


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Executing decluttered 5-stage t-SNE pipeline on: {device}")

    evaluator = JointSpaceEvaluator(
        metadata_path="./data/processed/metadata_processed.csv",
        model_name="ViT-B/32",
        device=device,
        batch_size=32,
        balance_taxonomically=True,
    )

    # You can adjust samples_per_class (e.g., 10, 15, or 20) to control plot density
    generate_tsne_5levels_with_key(evaluator, samples_per_class=15)


if __name__ == "__main__":
    main()