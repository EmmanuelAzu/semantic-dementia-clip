import os
import pandas as pd
import torch
import clip
from PIL import Image
from tqdm import tqdm

# Ensure reproducibility
torch.manual_seed(42)

class DatasetEncoder:
    def __init__(self, metadata_path, image_dir, output_tensor_path, output_csv_path, batch_size=16):
        """
        Initializes the feature extraction pipeline.
        
        Args:
            metadata_path (str): Path to raw CSV mapping filenames to the 4-tier taxonomy.
            image_dir (str): Folder containing the physical images (THINGS/BOSS subset).
            output_tensor_path (str): Destination for cached, normalized PyTorch tensors.
            output_csv_path (str): Destination for synchronized final metadata.
            batch_size (int): Mini-batch size to prevent GPU/CPU RAM overflow.
        """
        self.metadata_path = metadata_path
        self.image_dir = image_dir
        self.output_tensor_path = output_tensor_path
        self.output_csv_path = output_csv_path
        self.batch_size = batch_size
        
        # Determine execution hardware
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[*] Running feature extraction on: {self.device.upper()}")
        
        # Load pre-trained CLIP model (ViT-B/32)
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.model.eval()  # Freeze weights for extraction
        
    def load_and_verify_metadata(self):
        """Loads and checks the integrity of the 4-tier taxonomy."""
        df = pd.read_csv(self.metadata_path)
        required_columns = ["filename", "domain", "superordinate", "basic", "specific"]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"[!] Missing required column '{col}' in metadata CSV.")
        print(f"[*] Metadata verified. Found {len(df)} indexed items across {df['domain'].nunique()} domains.")
        return df

    def extract_features(self):
        df = self.load_and_verify_metadata()
        embeddings = []
        valid_rows = []
        
        print("[*] Commencing feature extraction batch loop...")
        
        # Extract features in mini-batches to prevent system memory lockups
        for i in tqdm(range(0, len(df), self.batch_size)):
            batch_df = df.iloc[i:i+self.batch_size]
            batch_images = []
            batch_indices = []
            
            for idx, row in batch_df.iterrows():
                img_path = os.path.join(self.image_dir, row["filename"])
                
                if not os.path.exists(img_path):
                    print(f"\n[!] Warning: Image {row['filename']} not found. Skipping entry.")
                    continue
                
                try:
                    # Preprocess raw image based on standard CLIP requirements (224x224 resize, center crop, normalize)
                    img = Image.open(img_path).convert("RGB")
                    processed_img = self.preprocess(img)
                    batch_images.append(processed_img)
                    batch_indices.append(idx)
                except Exception as e:
                    print(f"\n[!] Error processing image {row['filename']}: {e}. Skipping.")
            
            if not batch_images:
                continue
                
            # Stack images into a single tensor batch: (Batch_Size, 3, 224, 224)
            image_tensor = torch.stack(batch_images).to(self.device)
            
            with torch.no_grad():
                # Extract dense multi-dimensional visual features
                image_features = self.model.encode_image(image_tensor)
                
                # Perform L2 Normalization to scale vector lengths to unit length 1.0
                image_features /= image_features.norm(dim=-1, keepdim=True)
                
                embeddings.append(image_features.cpu())
                valid_rows.append(df.iloc[batch_indices])
                
        # Merge batch elements into a static, centralized tensor matrix
        final_embeddings = torch.cat(embeddings, dim=0)
        final_df = pd.concat(valid_rows).reset_index(drop=True)
        
        # Update Row Identifiers to match embedding index (The "Index Lock")
        final_df["matrix_index"] = final_df.index
        
        # Save output structures
        torch.save(final_embeddings, self.output_tensor_path)
        final_df.to_csv(self.output_csv_path, index=False)
        
        print(f"\n[+] Success! Shared embedding matrix saved to: {self.output_tensor_path}")
        print(f"[+] Matrix Dimension shape: {list(final_embeddings.shape)} (N_Images, 512-d)")
        print(f"[+] Aligned clinical taxonomy CSV saved to: {self.output_csv_path}")

if __name__ == "__main__":
    # Test directory creation
    os.makedirs("./data", exist_ok=True)
    os.makedirs("./data/images", exist_ok=True)
    
    # Instance paths (Edit these as files are settled in)
    encoder = DatasetEncoder(
        metadata_path="./data/metadata_raw.csv",
        image_dir="./data/images",
        output_tensor_path="./data/image_index.pt",
        output_csv_path="./data/metadata_processed.csv",
        batch_size=8
    )
    # encoder.extract_features() # Uncomment to execute once inputs exist!