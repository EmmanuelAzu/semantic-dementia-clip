import os
import torch
import pandas as pd
import clip
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class ScaledImageDataset(Dataset):
    def __init__(self, csv_path, img_dir, preprocess):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.preprocess = preprocess
        
        # Verify files exist to prevent runtime crashes
        self.valid_indices = []
        for idx, row in self.df.iterrows():
            img_path = os.path.join(self.img_dir, row['filename'])
            if os.path.exists(img_path):
                self.valid_indices.append(idx)
                
        self.df = self.df.iloc[self.valid_indices].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['filename'])
        image = Image.open(img_path).convert("RGB")
        tensor = self.preprocess(image)
        return tensor, idx

def run_indexing(batch_size=64, num_workers=4):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Initializing Scaled Indexer on Device: {device.upper()}")

    # Setup directories
    raw_csv = "./tests/metadata_raw.csv"
    img_dir = "./data/raw"
    output_dir = "./data/processed"
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(raw_csv):
        print(f"[!] Error: Raw metadata not found at {raw_csv}. Run data download/builder first!")
        return

    # Load pre-trained CLIP
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    # Create dataset & loader
    dataset = ScaledImageDataset(raw_csv, img_dir, preprocess)
    
    # On Windows, num_workers > 0 can sometimes cause multiprocessing issues. 
    # We'll set it to 0 if we aren't on a system that plays nice, but default to 4 for speed.
    workers = num_workers if os.name != 'nt' else 0
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)

    all_features = []
    print(f"[*] Visual Memory Bank: Indexing {len(dataset)} images in batches of {batch_size}...")

    with torch.no_grad():
        for batch_imgs, _ in tqdm(loader, desc="Generating Visual Embeddings"):
            batch_imgs = batch_imgs.to(device)
            # Generate L2-normalized image features
            image_features = model.encode_image(batch_imgs)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            all_features.append(image_features.cpu())

    # Concatenate all batches into a single master tensor
    index_tensor = torch.cat(all_features, dim=0)

    # Save outputs
    torch.save(index_tensor, os.path.join(output_dir, "image_index.pt"))
    dataset.df.to_csv(os.path.join(output_dir, "metadata_processed.csv"), index=False)

    print("\n[+] SUCCESS: Offline visual memory bank built!")
    print(f"[+] Saved index tensor shape: {index_tensor.shape}")
    print(f"[+] Processed metadata saved to: {output_dir}/metadata_processed.csv")

if __name__ == "__main__":
    run_indexing()