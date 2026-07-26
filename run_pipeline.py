import os
import sys

# Ensure project root is in system path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.build_general_dataset import FastGeneralDatasetBuilder
from run_indexing import run_indexing
from src.testing_harness import TestingHarness
from src.plot_results import plot_clinical_decay

# ==========================================
# MASTER BATCH EXPERIMENT CONFIGURATION
# ==========================================
GLOBAL_CONFIG = {
    "IMAGES_PER_CLASS": 50,  # Thesis Grade Scale
    "BATCH_SIZE": 64
}

# Revised structural matrix: Pruning strictly at the transmodal hub
EXPERIMENTAL_MATRIX = [
    # Linguistic Atrophy Pathway (Aphasia)
    {"encoder": "text"},
    
    # Visual Atrophy Pathway (Visual Agnosia)
    {"encoder": "vision"}
]

def run_automated_matrix():
    print("=" * 70)
    print("   CLIP SEMANTIC DEMENTIA - UNIFIED HUB-LEVEL SIMULATION")
    print("=" * 70)
    print(f"Dataset Scale: {GLOBAL_CONFIG['IMAGES_PER_CLASS']} images/class (~1,000 total images)\n")

    # ONE-TIME SETUPS: Build and Index the healthy state baseline
    print("\n--- [INIT] PHASE 1: DATA CURATION & TAXONOMY BUILDING ---")
    builder = FastGeneralDatasetBuilder()
    builder.build_dataset(max_images_per_class=GLOBAL_CONFIG["IMAGES_PER_CLASS"])

    print("\n--- [INIT] PHASE 2: HEALTHY VISUAL MEMORY INDEXING ---")
    run_indexing(batch_size=GLOBAL_CONFIG["BATCH_SIZE"])

    # Initialize the simulation environment once
    harness = TestingHarness(
        metadata_path="./data/processed/metadata_processed.csv",
        index_tensor_path="./data/processed/image_index.pt"
    )

    # LOOPING THROUGH THE MATRIX: Sequential Neuropathological Execution
    for idx, run in enumerate(EXPERIMENTAL_MATRIX, 1):
        encoder = run["encoder"]
        
        print("\n" + "#" * 60)
        print(f" RUN {idx}/{len(EXPERIMENTAL_MATRIX)}: Target Hub = {encoder.upper()}")
        print("#" * 60)
        
        # PHASE 3: Atrophy Simulation
        print(f"\n--- PHASE 3: SIMULATING {encoder.upper()} ATROPHY AT THE TRANSMODAL HUB ---")
        df_results = harness.run_simulation(
            encoder_type=encoder
        )
        
        # Save distinct CSV data for each specific permutation
        results_filename = f"{encoder}_hub_simulation.csv"
        results_path = os.path.join("./data/results", results_filename)
        os.makedirs("./data/results", exist_ok=True)
        df_results.to_csv(results_path, index=False)
        print(f"[+] Results safely stored at: {results_path}")

        # PHASE 4: Generating Thesis Figures
        print(f"\n--- PHASE 4: PLOTTING DECAY CURVE FOR {encoder.upper()} HUB ---")
        plot_clinical_decay(csv_path=results_path, output_dir="./data/results/")
        
    print("\n" + "=" * 70)
    print("   ALL EXPERIMENTAL MATRIX CONFIGURATIONS EXECUTED SUCCESSFULLY")
    print("   Check './data/results/' for all CSVs and PNG figures.")
    print("=" * 70)

if __name__ == "__main__":
    run_automated_matrix()