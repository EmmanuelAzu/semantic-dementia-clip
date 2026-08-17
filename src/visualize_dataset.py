import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

def visualize_random_samples(metadata_path, img_dir, num_samples=9):
    """
    Loads the processed metadata and plots a grid of random images 
    along with their clinical taxonomy mappings.
    """
    print(f"[*] Reading dataset from: {metadata_path}")
    if not os.path.exists(metadata_path):
        print(f"[!] Error: Could not find processed metadata. Please run the pipeline first!")
        return

    df = pd.read_csv(metadata_path)
    
    # Grab a random sample of images from the dataset
    samples = df.sample(n=min(num_samples, len(df))).reset_index(drop=True)
    
    # Calculate grid dimensions (e.g., 3x3 for 9 samples)
    grid_size = int(num_samples ** 0.5)
    if grid_size * grid_size < num_samples:
        grid_size += 1

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    axes = axes.flatten()

    for i, row in samples.iterrows():
        img_path = os.path.join(img_dir, row['filename'])
        ax = axes[i]
        
        if os.path.exists(img_path):
            img = Image.open(img_path)
            ax.imshow(img)
            
            # Format title to show the hierarchical taxonomy path
            title_text = (
                f"Domain: {row['domain'].upper()}\n"
                f"Super: {row['superordinate'].capitalize()}\n"
                f"coordinate: {row['coordinate'].capitalize()}\n"
                f"Specific: {row['specific'].capitalize()}"
            )
            ax.set_title(title_text, fontsize=10, fontweight='bold', pad=8)
        else:
            ax.text(0.5, 0.5, f"Missing Image:\n{row['filename']}", 
                    ha='center', va='center', color='red', fontsize=10)
            
        ax.axis('off')

    # Hide any unused subplot slots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle("Sample of Images & Clinical Taxonomy Mappings in Your Model", 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save a copy for your thesis appendix
    output_path = "./data/results/dataset_samples.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[+] Saved visualization sheet to: {output_path}")
    
    plt.show()

if __name__ == "__main__":
    visualize_random_samples(
        metadata_path="./data/processed/metadata_processed.csv",
        img_dir="./data/raw",
        num_samples=9
    )