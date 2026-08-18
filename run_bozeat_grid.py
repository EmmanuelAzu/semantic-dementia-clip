import torch
from src.bozeat_experiment import (
    DEFAULT_TARGET_PROMPTS,
    PRUNING_LEVELS_5PCT,
    run_bozeat_visual_grid,
)
from src.joint_evaluator import JointSpaceEvaluator


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Executing 18-stage Bozeat Visual Retrieval Grid on: {device}")

    evaluator = JointSpaceEvaluator(
        metadata_path="./data/processed/metadata_processed.csv",
        model_name="ViT-B/32",
        device=device,
        batch_size=32,
        balance_taxonomically=True,
    )

    run_bozeat_visual_grid(
        evaluator,
        pruning_levels=PRUNING_LEVELS_5PCT,
        target_prompts=DEFAULT_TARGET_PROMPTS,
        output_dir="./data/results/bozeat_experiment",
    )


if __name__ == "__main__":
    main()