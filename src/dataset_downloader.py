import os
import zipfile
import urllib.request
import shutil
import pandas as pd
from tqdm import tqdm

# Define the exact mapping of Tiny ImageNet WordNet IDs to our 4-tier clinical taxonomy
# This ensures a balanced selection of Living vs Non-Living entities
# Update TAXONOMY_MAP in your curation pipeline script (Source 2)
TAXONOMY_MAP = {
    # --- LIVING DOMAIN (10 Classes) ---
    # Animals
    "n02114304": {"domain": "living", "superordinate": "animal", "basic": "canine", "specific": "chihuahua"},
    "n02124075": {"domain": "living", "superordinate": "animal", "basic": "feline", "specific": "egyptian_cat"},
    "n02504458": {"domain": "living", "superordinate": "animal", "basic": "large_mammal", "specific": "african_elephant"},
    "n01443537": {"domain": "living", "superordinate": "animal", "basic": "fish", "specific": "goldfish"},
    "n01843383": {"domain": "living", "superordinate": "animal", "basic": "bird", "specific": "toucan"},
    # Plants & Produce
    "n07742439": {"domain": "living", "superordinate": "plant_food", "basic": "fruit", "specific": "lemon"},
    "n07753275": {"domain": "living", "superordinate": "plant_food", "basic": "fruit", "specific": "pineapple"},
    "n07720875": {"domain": "living", "superordinate": "plant_food", "basic": "vegetable", "specific": "bell_pepper"},
    "n07739344": {"domain": "living", "superordinate": "plant_food", "basic": "vegetable", "specific": "granny_smith_apple"},
    "n07697313": {"domain": "living", "superordinate": "plant_food", "basic": "fungi", "specific": "mushroom"},

    # --- NON-LIVING DOMAIN (10 Classes) ---
    # Vehicles
    "n03100240": {"domain": "non-living", "superordinate": "vehicle", "basic": "land_transport", "specific": "convertible_car"},
    "n03791053": {"domain": "non-living", "superordinate": "vehicle", "basic": "land_transport", "specific": "motorcycle"},
    "n02690373": {"domain": "non-living", "superordinate": "vehicle", "basic": "air_transport", "specific": "airliner"},
    "n04090263": {"domain": "non-living", "superordinate": "vehicle", "basic": "water_transport", "specific": "rifle_boat"},
    "n04467665": {"domain": "non-living", "superordinate": "vehicle", "basic": "rail_transport", "specific": "trolleybus"},
    # Everyday Artifacts & Tools
    "n03485407": {"domain": "non-living", "superordinate": "artifact", "basic": "tool", "specific": "hammer"},
    "n03001627": {"domain": "non-living", "superordinate": "artifact", "basic": "household_item", "specific": "chair"},
    "n03047056": {"domain": "non-living", "superordinate": "artifact", "basic": "household_item", "specific": "wall_clock"},
    "n03980874": {"domain": "non-living", "superordinate": "artifact", "basic": "kitchen_utensil", "specific": "corkscrew"},
    "n04356056": {"domain": "non-living", "superordinate": "artifact", "basic": "container", "specific": "water_bottle"}
}

class TinyImageNetCurationPipeline:
    def __init__(self, workspace_dir="./data"):
        self.workspace_dir = workspace_dir
        self.zip_path = os.path.join(workspace_dir, "tiny-imagenet-200.zip")
        self.extract_dir = os.path.join(workspace_dir, "tiny-imagenet-200")
        self.raw_output_dir = os.path.join(workspace_dir, "raw")
        self.metadata_path = "./tests/metadata_raw.csv"
        
        # Ensure directories exist
        os.makedirs(self.workspace_dir, exist_ok=True)
        os.makedirs(self.raw_output_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.metadata_path), exist_ok=True)

    def download_dataset(self):
        """Downloads Tiny ImageNet if not already cached locally."""
        url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
        if not os.path.exists(self.zip_path):
            print(f"[*] Downloading academic Tiny ImageNet archive (~120MB) from Stanford...")
            with urllib.request.urlopen(url) as response, open(self.zip_path, 'wb') as out_file:
                # Use a simple progress bar wrapper
                meta = response.info()
                file_size = int(meta.get("Content-Length"))
                chunk_size = 1024 * 1024  # 1MB chunks
                
                with tqdm(total=file_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                    while True:
                        buffer = response.read(chunk_size)
                        if not buffer:
                            break
                        out_file.write(buffer)
                        pbar.update(len(buffer))
            print("[+] Download complete.")
        else:
            print("[*] Found cached Tiny ImageNet archive. Skipping download.")

    def extract_and_filter(self):
        """Extracts files and builds the balanced taxonomy dataset."""
        if not os.path.exists(self.extract_dir):
            print("[*] Extracting zip file...")
            with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.workspace_dir)
            print("[+] Extraction complete.")

        print("[*] Filtering and moving taxonomic targets into data/raw/ ...")
        records = []
        train_dir = os.path.join(self.extract_dir, "train")

        for wnid, meta_info in tqdm(TAXONOMY_MAP.items(), desc="Processing classes"):
            class_img_dir = os.path.join(train_dir, wnid, "images")
            if not os.path.exists(class_img_dir):
                print(f"[!] Warning: Directory for class {wnid} not found.")
                continue

            # Grab up to 100 images per class for balanced, high-volume representation
            images = sorted(os.listdir(class_img_dir))[:100]
            
            for img_name in images:
                src_img_path = os.path.join(class_img_dir, img_name)
                # Rename file to make it easily searchable and cleanly categorized
                new_filename = f"{meta_info['specific']}_{img_name}"
                dest_img_path = os.path.join(self.raw_output_dir, new_filename)
                
                # Copy image file
                shutil.copy(src_img_path, dest_img_path)
                
                # Append row metadata
                records.append({
                    "filename": new_filename,
                    "domain": meta_info["domain"],
                    "superordinate": meta_info["superordinate"],
                    "basic": meta_info["basic"],
                    "specific": meta_info["specific"]
                })

        # Generate the unified project metadata raw CSV
        df = pd.DataFrame(records)
        df.to_csv(self.metadata_path, index=False)
        print(f"\n[+] SUCCESS: Curated {len(df)} balanced academic-grade image samples.")
        print(f"[+] Output Folder: {self.raw_output_dir}")
        print(f"[+] Taxonomy Index File written to: {self.metadata_path}")

        # Clean up the massive raw extraction directory to save local hard drive space
        print("[*] Cleaning up temporary extraction files...")
        if os.path.exists(self.extract_dir):
            shutil.rmtree(self.extract_dir)
        print("[+] Cleanup complete. Keeping compressed zip for reproducibility cached at data/.")

if __name__ == "__main__":
    pipeline = TinyImageNetCurationPipeline()
    pipeline.download_dataset()
    pipeline.extract_and_filter()