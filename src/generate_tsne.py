import os
import sys

# Ensure root folder is accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import pandas as pd
import clip
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.manifold import TSNE
from src.pruning_engine import CLIPPruningEngine

def generate_tsne_grid(
    metadata_path="./data/processed/metadata_processed.csv",
    img_dir="./data/raw",
    output_plot_path="./data/results/tsne_cluster_decay.png",
    encoder_type="text",
    batch_size=64
):
    """
    Generates a 2x2 t-SNE grid showing vector cluster breakdown across
    0%, 30%, 60%, and 90% transmodal hub atrophy using fast GPU batching.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}.")
        
    metadata = pd.read_csv(metadata_path)
    coord_col = 'coordinate' if 'coordinate' in metadata.columns else 'basic'
    
    # Load base model
    base_model, preprocess = clip.load("ViT-B/32", device=device)
    pruning_engine = CLIPPruningEngine(base_model)
    
    pruning_levels = [0.0, 0.3, 0.6, 0.9]
    unique_coords = metadata[coord_col].unique()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    
    # Setup consistent color palette
    palette = sns.color_palette("tab10", n_colors=len(unique_coords))
    color_map = dict(zip(unique_coords, palette))

    print(f"[*] Extracting embeddings (Batched) for {encoder_type.upper()} hub atrophy...")

    for i, p in enumerate(pruning_levels):
        ax = axes[i]
        
        # 1. Apply structured pruning
        atrophied_model = pruning_engine.get_pruned_model(amount=p, encoder_type=encoder_type)
        atrophied_model.eval()
        
        with torch.no_grad():
            if encoder_type == "text":
                # BATCH OPTIMIZATION: Tokenize all prompts simultaneously
                prompts = [f"a photo of a {row['specific']}" for _, row in metadata.iterrows()]
                text_inputs = clip.tokenize(prompts).to(device)
                
                # Single GPU forward pass for all items
                feats = atrophied_model.encode_text(text_inputs)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                X = feats.cpu().numpy()
                labels = metadata[coord_col].tolist()

            else:
                # BATCH OPTIMIZATION: Load and encode images in chunks
                embeddings = []
                labels = []
                batch_images = []
                
                for idx, row in metadata.iterrows():
                    img_path = os.path.join(img_dir, row['filename'])
                    if os.path.exists(img_path):
                        img = Image.open(img_path).convert("RGB")
                        batch_images.append(preprocess(img))
                        labels.append(row[coord_col])
                    
                    if len(batch_images) == batch_size or idx == len(metadata) - 1:
                        if batch_images:
                            batch_tensor = torch.stack(batch_images).to(device)
                            feats = atrophied_model.encode_image(batch_tensor)
                            feats = feats / feats.norm(dim=-1, keepdim=True)
                            embeddings.append(feats.cpu().numpy())
                            batch_images = []
                            
                X = np.vstack(embeddings)
        
        # 2. Fast t-SNE reduction
        perplexity_val = min(30, max(5, len(X) // 4))
        tsne = TSNE(
            n_components=2, 
            perplexity=perplexity_val, 
            random_state=42, 
            init='pca', 
            learning_rate='auto',
            n_jobs=-1  # Multi-threaded CPU execution
        )
        X_2d = tsne.fit_transform(X)
        
        # 3. Plotting
        df_tsne = pd.DataFrame({
            'tsne_1': X_2d[:, 0],
            'tsne_2': X_2d[:, 1],
            'coordinate': labels
        })
        
        sns.scatterplot(
            data=df_tsne,
            x='tsne_1',
            y='tsne_2',
            hue='coordinate',
            palette=color_map,
            ax=ax,
            s=80,
            alpha=0.85,
            legend=(i == 0)
        )
        
        ax.set_title(f"Atrophy Level: {int(p*100)}%", fontsize=14, fontweight='bold')
        ax.set_xlabel("t-SNE Dimension 1", fontsize=10)
        ax.set_ylabel("t-SNE Dimension 2", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5)

    # Position legend outside subplots
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, title="Coordinate Category", loc="center right", bbox_to_anchor=(1.12, 0.5))
    axes[0].get_legend().remove()

    plt.suptitle(f"Semantic Cluster Dissolution Under {encoder_type.upper()} Hub Atrophy (t-SNE)", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 0.88, 0.95])
    
    os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)
    plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[+] Fast t-SNE Plot saved to: {output_plot_path}")

if __name__ == "__main__":
    generate_tsne_grid(encoder_type="text")