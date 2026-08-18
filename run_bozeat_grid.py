import torch
from src.joint_evaluator import JointSpaceEvaluator
from src.bozeat_experiment import run_bozeat_visual_grid_5levels

# 5 Pruning Levels
PRUNING_LEVELS_5 = [0.00, 0.15, 0.35, 0.60, 0.85]

# 3 Prompts across living & non-living domains
TARGET_PROMPTS = ["Tabby Cat", "Golden Retriever", "Sports Car"]

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Executing 5-Level Bozeat Visual Retrieval Grid on device: {device}")

    evaluator = JointSpaceEvaluator(
        metadata_path="./data/processed/metadata_processed.csv",
        model_name="ViT-B/32",
        device=device,
        batch_size=32,
        balance_taxonomically=True
    )

    run_bozeat_visual_grid_5levels(
        evaluator,
        pruning_levels=PRUNING_LEVELS_5,
        target_prompts=TARGET_PROMPTS,
        output_dir="./data/results/bozeat_experiment"
    )

if __name__ == "__main__":
    main()