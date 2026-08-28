import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.gen_tSNE_subcat_nerr import generate_tsne_clean_color_grid
from src.joint_evaluator import JointSpaceEvaluator

# 18-stage pruning schedule (5% increments from 0% to 85%)
PRUNING_LEVELS_5PCT = [round(i * 0.05, 2) for i in range(18)]


def main():
    parser = argparse.ArgumentParser(
        description="Run Domain-Grouped Concept Dispersion t-SNE Generation."
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
        help="Explicit sample count for evaluation",
    )
    parser.add_argument(
        "--no_balance",
        action="store_true",
        help="Disable taxonomic class balancing",
    )
    parser.add_argument(
        "--samples_per_class",
        type=int,
        default=5,
        help="Number of image samples per concept class (default: 5)",
    )
    # The --single_word_prompts argument has been removed as it is no longer used for this plot
    args = parser.parse_args()

    print("=" * 60)
    print(" RUNNING DOMAIN-GROUPED CONCEPT DISPERSION t-SNE GENERATION ")
    print("=" * 60)

    evaluator = JointSpaceEvaluator(
        sample_frac=args.sample_frac,
        target_n=args.target_n,
        balance_taxonomically=not args.no_balance,
    )

    save_path = generate_tsne_clean_color_grid(
        evaluator=evaluator,
        pruning_levels=PRUNING_LEVELS_5PCT,
        samples_per_class=args.samples_per_class,
        output_dir=os.path.join(PROJECT_ROOT, "data", "results", "tsne"),
    )

    print(
        f"\n[+] Hierarchical concept-dispersion t-SNE grid generated successfully:\n    {save_path}"
    )


if __name__ == "__main__":
    main()