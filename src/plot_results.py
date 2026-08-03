import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_clinical_decay(csv_path, output_dir):
    print(f"[*] Loading simulation data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Extract the base name of the CSV to name the plot dynamically
    base_name = os.path.basename(csv_path).replace('.csv', '')

    # Set up the academic plot style
    sns.set_theme(style="whitegrid", palette="muted")
    plt.figure(figsize=(12, 7))

    # Convert Pruning Level to percentages for the X-axis
    x_vals = df['Pruning_Level'] * 100

    # Plot each error trajectory
    plt.plot(x_vals, df['Correct'], marker='o', label='Correct Retrieval', linewidth=2.5, color='forestgreen')
    plt.plot(x_vals, df['Coordinate Error'], marker='s', label='Coordinate Error (e.g., Dog -> Cat)', linewidth=2, color='darkorange')
    plt.plot(x_vals, df['Superordinate Error'], marker='^', label='Superordinate Error (e.g., Dog -> Animal)', linewidth=2, color='firebrick')
    plt.plot(x_vals, df['Domain Error'], marker='x', label='Domain Error', linewidth=2, color='purple')
    plt.plot(x_vals, df['Domain Collapse'], marker='d', label='Domain Collapse (e.g., Living -> Non-Living)', linewidth=2, color='black', linestyle='--')

    # Formatting the Graph for your Thesis
    # Dynamically update the title based on the simulation type
    sim_type = "Aphasia (Text Hub)" if "text" in base_name else "Visual Agnosia (Vision Hub)"
    plt.title(f'Simulated Semantic Dementia: {sim_type}\nHierarchical Conceptual Breakdown via Network Atrophy', fontsize=16, fontweight='bold', pad=15)
    
    plt.xlabel('Network Pruning Intensity (% of Weights Zeroed)', fontsize=14)
    plt.ylabel('Number of Retrievals', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    plt.legend(title="Response Type", title_fontsize='13', fontsize='11', loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)

    # Save the output dynamically
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f'{base_name}_curve.png')
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    
    print(f"[+] Plot successfully saved to: {out_file}")
    
    # Close the plot automatically so the pipeline can continue to the next iteration without waiting for user input
    plt.close()

if __name__ == "__main__":
    # Updated to match the new hub-focused CSV naming convention
    csv_file = "./data/results/text_hub_simulation.csv"
    out_folder = "./data/results/"
    
    if os.path.exists(csv_file):
        plot_clinical_decay(csv_file, out_folder)
        plt.show()  # Only show it interactively if running this script directly
    else:
        print(f"[!] Could not find {csv_file}. Please ensure you ran the testing harness first.")