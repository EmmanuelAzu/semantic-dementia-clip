import os
import pandas as pd
import torch
import clip
from PIL import Image
from tqdm import tqdm

torch.manual_seed(42)

class DatasetEncoder:
    def __init__(self, metadata_path, image_dir, output_tensor_path, output_csv_path, batch_size=16):
        self.metadata_path = metadata_path
        self.image_dir = image_dir
        self.output_tensor_path = output_tensor_path
        self.output_csv_path = output_csv_path
        self.batch_size = batch_size
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[*] Running feature extraction on: {self.device.upper()}")
        
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.model.eval()
        
    def load_and_verify_metadata(self):
        df = pd.read_csv(self.metadata_path)
        required_columns = ["filename", "domain", "superordinate", "coordinate", "specific"]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"[!] Missing required column '{col}' in metadata CSV.")
        print(f"[*] Metadata verified. Found {len(df)} indexed items.")
        return df

    def extract_features(self):
        df = self.load_and_verify_metadata()
        embeddings = []
        valid_rows = []
        
        for i in tqdm(range(0, len(df), self.batch_size), desc="Encoding batches"):
            batch_df = df.iloc[i:i+self.batch_size]
            batch_images = []
            batch_indices = []
            
            for idx, row in batch_df.iterrows():
                img_path = os.path.join(self.image_dir, row["filename"])
                if not os.path.exists(img_path):
                    continue
                
                try:
                    img = Image.open(img_path).convert("RGB")
                    batch_images.append(self.preprocess(img))
                    batch_indices.append(idx)
                except Exception as e:
                    print(f"\n[!] Error processing image {row['filename']}: {e}")
            
            if not batch_images:
                continue
                
            image_tensor = torch.stack(batch_images).to(self.device)
            
            with torch.no_grad():
                image_features = self.model.encode_image(image_tensor)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                embeddings.append(image_features.cpu())
                valid_rows.append(df.loc[batch_indices])
                
        if not embeddings:
            raise RuntimeError("[!] No images were successfully encoded. Check your input directory.")

        final_embeddings = torch.cat(embeddings, dim=0)
        final_df = pd.concat(valid_rows).reset_index(drop=True)
        final_df["matrix_index"] = final_df.index
        
        os.makedirs(os.path.dirname(self.output_tensor_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.output_csv_path), exist_ok=True)
        
        torch.save(final_embeddings, self.output_tensor_path)
        final_df.to_csv(self.output_csv_path, index=False)
        print(f"[+] Saved embeddings ({final_embeddings.shape}) to {self.output_tensor_path}")

if __name__ == "__main__":
    encoder = DatasetEncoder(
        metadata_path="./data/metadata_raw.csv",
        image_dir="./data/images",
        output_tensor_path="./data/image_index.pt",
        output_csv_path="./data/processed/metadata_processed.csv",
        batch_size=16
    )
    encoder.extract_features()