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
        self.metadata_path = "./data/processed/metadata_processed.csv"
        
        os.makedirs(self.raw_output_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.metadata_path), exist_ok=True)

        # Curated Tiny-ImageNet WNIDs mapped to dense coordinate clusters
        # Ensures dense sibling neighborhoods in the memory index (~100 images total)
        self.target_taxonomy = {
            # LIVING - CANINES
            "n02106662": {"domain": "Living", "superordinate": "Animal", "coordinate": "Canine", "specific": "German Shepherd"},
            "n02113712": {"domain": "Living", "superordinate": "Animal", "coordinate": "Canine", "specific": "Miniature Poodle"},
            "n02099601": {"domain": "Living", "superordinate": "Animal", "coordinate": "Canine", "specific": "Golden Retriever"},
            
            # LIVING - FELINES
            "n02123045": {"domain": "Living", "superordinate": "Animal", "coordinate": "Feline", "specific": "Tabby Cat"},
            "n02124075": {"domain": "Living", "superordinate": "Animal", "coordinate": "Feline", "specific": "Egyptian Cat"},
            
            # LIVING - ARTHROPODS
            "n01770393": {"domain": "Living", "superordinate": "Animal", "coordinate": "Arthropod", "specific": "Scorpion"},
            "n01773504": {"domain": "Living", "superordinate": "Animal", "coordinate": "Arthropod", "specific": "Tarantula"},
            
            # LIVING - AQUATIC / BIRDS
            "n01443537": {"domain": "Living", "superordinate": "Animal", "coordinate": "Aquatic", "specific": "Goldfish"},
            "n02007558": {"domain": "Living", "superordinate": "Animal", "coordinate": "Bird", "specific": "Flamingo"},

            # NON-LIVING - VEHICLES
            "n02691156": {"domain": "Non-Living", "superordinate": "Vehicle", "coordinate": "Automobile", "specific": "Airplane"},
            "n04254680": {"domain": "Non-Living", "superordinate": "Vehicle", "coordinate": "Automobile", "specific": "Sports Car"},
            "n03977966": {"domain": "Non-Living", "superordinate": "Vehicle", "coordinate": "Automobile", "specific": "Police Van"},
            "n03791053": {"domain": "Non-Living", "superordinate": "Vehicle", "coordinate": "Automobile", "specific": "Motor Scooter"},

            # NON-LIVING - STRUCTURES / OBJECTS
            "n02808440": {"domain": "Non-Living", "superordinate": "Object", "coordinate": "Fixture", "specific": "Bathtub"},
            "n03085013": {"domain": "Non-Living", "superordinate": "Object", "coordinate": "Electronics", "specific": "Computer Keyboard"},
            "n03888257": {"domain": "Non-Living", "superordinate": "Object", "coordinate": "Gear", "specific": "Parachute"}
        }

    def build_dataset(self, max_images_per_class=6):
        print("[*] Reading zip file dynamically in memory...")
        
        if not os.path.exists(self.zip_path):
            print(f"[!] Error: Cannot find {self.zip_path}. Please place it in the data folder.")
            return

        records = []
        
        with zipfile.ZipFile(self.zip_path, 'r') as z:
            all_files = z.namelist()
            train_files = [f for f in all_files if f.startswith("tiny-imagenet-200/train/") and f.endswith(".JPEG")]
            
            # Group files by their WordNet ID (Class)
            class_files = defaultdict(list)
            for f in train_files:
                parts = f.split('/')
                wnid = parts[2]
                if wnid in self.target_taxonomy:  # Only collect target WNIDs
                    class_files[wnid].append(f)
                
            print(f"[*] Found {len(class_files)} target sibling classes. Extracting {max_images_per_class} images per class...")
            
            for wnid, files in tqdm(class_files.items(), desc="Building Fine-Grained Semantic Space"):
                taxonomy = self.target_taxonomy[wnid]
                selected_files = files[:max_images_per_class]
                
                for zip_file_path in selected_files:
                    img_name = os.path.basename(zip_file_path)
                    clean_specific_name = taxonomy['specific'].lower().replace(' ', '_')
                    new_filename = f"{clean_specific_name}_{img_name}"
                    dest_img_path = os.path.join(self.raw_output_dir, new_filename)
                    
                    # Extract directly from zip memory stream
                    with z.open(zip_file_path) as source, open(dest_img_path, "wb") as target:
                        target.write(source.read())
                        
                    records.append({
                        "filename": new_filename,
                        "domain": taxonomy['domain'],
                        "superordinate": taxonomy['superordinate'],
                        "coordinate": taxonomy['coordinate'],
                        "specific": taxonomy['specific']
                    })

        df = pd.DataFrame(records)
        df.to_csv(self.metadata_path, index=False)
        
        print(f"\n[+] SUCCESS: Dataset built with {len(df)} images across {len(class_files)} classes.")
        print(f"[+] Multi-tier taxonomy saved to: {self.metadata_path}")

if __name__ == "__main__":
    builder = FastGeneralDatasetBuilder()
    builder.build_dataset(max_images_per_class=6)  # 16 classes * 6 images = 96 total images