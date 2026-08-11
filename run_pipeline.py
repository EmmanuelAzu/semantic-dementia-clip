import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.generate_analysis_plots import (
    plot_category_breakdown_suite,
    plot_hierarchical_breakdown_suite,
    plot_signal_noise_distribution_shift,
    plot_target_rank_waterfall,
    plot_concept_retrieval_heatmap,
)
from src.generate_tsne import generate_joint_hierarchical_tsne
from src.joint_evaluator import JointSpaceEvaluator

PRUNING_LEVELS = [0.0, 0.25, 0.50, 0.75, 0.90, 0.95]


def run_joint_pipeline():
    print("=" * 60)
    print(" RUNNING 10% SAMPLE MULTIMODAL EVALUATION PIPELINE ")
    print("=" * 60)

    # Instantiate evaluator with 10% sample fraction (Requirement 1)
    evaluator = JointSpaceEvaluator(sample_frac=0.10)
    results_df = evaluator.run_eval(pruning_levels=PRUNING_LEVELS)

    csv_path = os.path.join(
        PROJECT_ROOT, "data", "results", "joint_space_metrics_10pct.csv"
    )
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    results_df.to_csv(csv_path, index=False)
    print(f"[+] 10% Evaluation metrics saved to:\n    {csv_path}")

    # Generate visual outputs
    plot_category_breakdown_suite(results_df)
    plot_hierarchical_breakdown_suite(results_df)
    plot_signal_noise_distribution_shift(evaluator, pruning_levels=[0.0, 0.50, 0.75, 0.95])
    plot_target_rank_waterfall(evaluator, pruning_levels=PRUNING_LEVELS)
    plot_concept_retrieval_heatmap(evaluator, pruning_levels=PRUNING_LEVELS)
    generate_joint_hierarchical_tsne(evaluator, pruning_levels=PRUNING_LEVELS)

    print("\n[+] Full evaluation suite successfully executed!")


if __name__ == "__main__":
    run_joint_pipeline()