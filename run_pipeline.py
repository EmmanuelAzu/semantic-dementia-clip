import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.generate_analysis_plots import (
    plot_category_breakdown_suite,
    plot_concept_retrieval_heatmap,
    plot_hierarchical_breakdown_suite,
    plot_signal_noise_distribution_shift,
    plot_target_rank_waterfall,
)
from src.generate_tsne import generate_joint_hierarchical_tsne
from src.joint_evaluator import JointSpaceEvaluator

PRUNING_LEVELS = [0.0, 0.25, 0.50, 0.75, 0.90, 0.95]


def run_joint_pipeline(sample_frac=1.0, target_n=None, balance_taxonomically=True):
    print("=" * 60)
    print(" RUNNING MULTIMODAL SEMANTIC DEMENTIA EVALUATION PIPELINE ")
    print("=" * 60)

    evaluator = JointSpaceEvaluator(
        sample_frac=sample_frac,
        target_n=target_n,
        balance_taxonomically=balance_taxonomically,
    )
    results_df = evaluator.run_eval(pruning_levels=PRUNING_LEVELS)

    csv_path = os.path.join(
        PROJECT_ROOT, "data", "results", "joint_space_metrics_full.csv"
    )
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    results_df.to_csv(csv_path, index=False)
    print(f"\n[+] Full Evaluation metrics saved to:\n    {csv_path}")

    # Generate complete visualization suite
    plot_category_breakdown_suite(results_df)
    plot_hierarchical_breakdown_suite(results_df)
    plot_signal_noise_distribution_shift(
        evaluator, pruning_levels=[0.0, 0.50, 0.75, 0.95]
    )
    plot_target_rank_waterfall(evaluator, pruning_levels=PRUNING_LEVELS)
    plot_concept_retrieval_heatmap(evaluator, pruning_levels=PRUNING_LEVELS)
    generate_joint_hierarchical_tsne(evaluator, pruning_levels=PRUNING_LEVELS)

    print(
        "\n[+] Honors project pipeline successfully executed end-to-end!"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Multimodal Semantic Dementia CLIP Pruning Evaluation."
    )
    parser.add_argument(
        "--sample_frac",
        type=float,
        default=1.0,
        help="Fraction of dataset to evaluate",
    )
    parser.add_argument(
        "--target_n",
        type=int,
        default=None,
        help="Explicit sample count (e.g., 800 for oversampled evaluation)",
    )
    parser.add_argument(
        "--no_balance",
        action="store_true",
        help="Disable taxonomic class balancing",
    )
    args = parser.parse_args()

    run_joint_pipeline(
        sample_frac=args.sample_frac,
        target_n=args.target_n,
        balance_taxonomically=not args.no_balance,
    )