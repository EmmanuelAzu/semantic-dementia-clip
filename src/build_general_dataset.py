import os
import zipfile
import pandas as pd
import nltk
from nltk.corpus import wordnet as wn
from collections import defaultdict
from tqdm import tqdm

# Download WordNet data
nltk.download('wordnet', quiet=True)

class FastGeneralDatasetBuilder:
    def __init__(self, workspace_dir="./data"):
        self.zip_path = os.path.join(workspace_dir, "tiny-imagenet-200.zip")
        self.raw_output_dir = os.path.join(workspace_dir, "raw")
        self.metadata_path = "./tests/metadata_raw.csv"
        
        os.makedirs(self.raw_output_dir, exist_ok=True)

    def get_taxonomy_from_wnid(self, wnid):
        """Dynamically extracts a 4-tier taxonomy using WordNet's hypernym tree."""
        try:
            offset = int(wnid[1:])
            synset = wn.synset_from_pos_and_offset('n', offset)
            paths = synset.hypernym_paths()
            if not paths: return None
            
            names = [s.name().split('.')[0] for s in paths[0]]
            
            # Domain
            if 'organism' in names or 'living_thing' in names: domain = "living"
            elif 'artifact' in names or 'object' in names or 'matter' in names: domain = "non-living"
            else: domain = "other"
                
            superordinate = names[4] if len(names) > 4 else names[-1]
            basic = names[-2] if len(names) > 1 else names[-1]
            specific = names[-1]
            
            return {"domain": domain, "superordinate": superordinate, "basic": basic, "specific": specific}
        except Exception:
            return None

    def build_dataset(self, max_images_per_class=10):
        print("[*] Reading zip file dynamically in memory (bypassing Windows Explorer)...")
        
        if not os.path.exists(self.zip_path):
            print(f"[!] Error: Cannot find {self.zip_path}. Please make sure it's in the data folder.")
            return

        records = []
        
        with zipfile.ZipFile(self.zip_path, 'r') as z:
            # Get list of all files in the zip without extracting them
            all_files = z.namelist()
            
            # Find only the training images
            train_files = [f for f in all_files if f.startswith("tiny-imagenet-200/train/") and f.endswith(".JPEG")]
            
            # Group files by their WordNet ID (Class)
            class_files = defaultdict(list)
            for f in train_files:
                parts = f.split('/')
                wnid = parts[2]
                class_files[wnid].append(f)
                
            print(f"[*] Found {len(class_files)} classes. Filtering taxonomy and extracting target images...")
            
            # Only process the exact number of files we need
            for wnid, files in tqdm(class_files.items(), desc="Building Semantic Space"):
                taxonomy = self.get_taxonomy_from_wnid(wnid)
                if not taxonomy: continue
                    
                selected_files = files[:max_images_per_class]
                
                for zip_file_path in selected_files:
                    img_name = os.path.basename(zip_file_path)
                    new_filename = f"{taxonomy['specific']}_{img_name}"
                    dest_img_path = os.path.join(self.raw_output_dir, new_filename)
                    
                    # Extract ONLY this specific file directly from the zip memory stream
                    with z.open(zip_file_path) as source, open(dest_img_path, "wb") as target:
                        target.write(source.read())
                        
                    records.append({
                        "filename": new_filename,
                        "domain": taxonomy['domain'],
                        "superordinate": taxonomy['superordinate'],
                        "basic": taxonomy['basic'],
                        "specific": taxonomy['specific']
                    })

        df = pd.DataFrame(records)
        df = df[df['domain'] != 'other'] 
        df.to_csv(self.metadata_path, index=False)
        
        print(f"\n[+] SUCCESS: Bypassed Windows extraction!")
        print(f"[+] Extracted exactly {len(df)} clinically-mapped images in record time.")
        print(f"[+] Taxonomy Index File written to: {self.metadata_path}")

if __name__ == "__main__":
    builder = FastGeneralDatasetBuilder()
    builder.build_dataset(max_images_per_class=10) # 10 images * ~200 classes = ~2000 total images