import io
import json
import os
import zipfile
import pandas as pd
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

TINY_IMAGENET_DRAWABLE_TAXONOMY = {
    # ==================== LIVING DOMAIN (16) ====================
    # --- ANIMALS (8) ---
    "n02123045": {
        "specific": "Cat",
        "coordinate": "Mammal",
        "superordinate": "Animal",
        "domain": "Living",
    },
    "n02106662": {
        "specific": "Dog",
        "coordinate": "Mammal",
        "superordinate": "Animal",
        "domain": "Living",
    },
    "n02381460": {
        "specific": "Horse",
        "coordinate": "Mammal",
        "superordinate": "Animal",
        "domain": "Living",
    },
    "n02504458": {
        "specific": "Elephant",
        "coordinate": "Mammal",
        "superordinate": "Animal",
        "domain": "Living",
    },
    "n01443537": {
        "specific": "Goldfish",
        "coordinate": "NonMammal",
        "superordinate": "Animal",
        "domain": "Living",
    },
    "n01514859": {
        "specific": "Hen",
        "coordinate": "NonMammal",
        "superordinate": "Animal",
        "domain": "Living",
    },
    "n01665541": {
        "specific": "Turtle",
        "coordinate": "NonMammal",
        "superordinate": "Animal",
        "domain": "Living",
    },
    "n01774750": {
        "specific": "Tarantula",
        "coordinate": "NonMammal",
        "superordinate": "Animal",
        "domain": "Living",
    },
    # --- PLANT & FOOD (8) ---
    "n07749582": {
        "specific": "Lemon",
        "coordinate": "Produce",
        "superordinate": "Plant",
        "domain": "Living",
    },
    "n07753592": {
        "specific": "Banana",
        "coordinate": "Produce",
        "superordinate": "Plant",
        "domain": "Living",
    },
    "n07583066": {
        "specific": "Guacamole",
        "coordinate": "Produce",
        "superordinate": "Plant",
        "domain": "Living",
    },
    "n07920052": {
        "specific": "Espresso",
        "coordinate": "Produce",
        "superordinate": "Plant",
        "domain": "Living",
    },
    "n07714990": {
        "specific": "Broccoli",
        "coordinate": "Flora",
        "superordinate": "Plant",
        "domain": "Living",
    },
    "n07718472": {
        "specific": "Cucumber",
        "coordinate": "Flora",
        "superordinate": "Plant",
        "domain": "Living",
    },
    "n11939491": {
        "specific": "Daisy",
        "coordinate": "Flora",
        "superordinate": "Plant",
        "domain": "Living",
    },
    "n12057211": {
        "specific": "Mushroom",
        "coordinate": "Flora",
        "superordinate": "Plant",
        "domain": "Living",
    },
    # ==================== NON-LIVING DOMAIN (16) ====================
    # --- VEHICLES (8) ---
    "n04254680": {
        "specific": "Car",
        "coordinate": "Land",
        "superordinate": "Vehicle",
        "domain": "Non-Living",
    },
    "n03977966": {
        "specific": "Van",
        "coordinate": "Land",
        "superordinate": "Vehicle",
        "domain": "Non-Living",
    },
    "n03791053": {
        "specific": "Scooter",
        "coordinate": "Land",
        "superordinate": "Vehicle",
        "domain": "Non-Living",
    },
    "n03393912": {
        "specific": "Tractor",
        "coordinate": "Land",
        "superordinate": "Vehicle",
        "domain": "Non-Living",
    },
    "n02691156": {
        "specific": "Airplane",
        "coordinate": "WaterAir",
        "superordinate": "Vehicle",
        "domain": "Non-Living",
    },
    "n02950826": {
        "specific": "Catamaran",
        "coordinate": "WaterAir",
        "superordinate": "Vehicle",
        "domain": "Non-Living",
    },
    "n03637318": {
        "specific": "Lifeboat",
        "coordinate": "WaterAir",
        "superordinate": "Vehicle",
        "domain": "Non-Living",
    },
    "n03445924": {
        "specific": "Gondola",
        "coordinate": "WaterAir",
        "superordinate": "Vehicle",
        "domain": "Non-Living",
    },
    # --- ARTIFACTS (8) ---
    "n04099969": {
        "specific": "Chair",
        "coordinate": "Furniture",
        "superordinate": "Artifact",
        "domain": "Non-Living",
    },
    "n03201208": {
        "specific": "Table",
        "coordinate": "Furniture",
        "superordinate": "Artifact",
        "domain": "Non-Living",
    },
    "n02808440": {
        "specific": "Bathtub",
        "coordinate": "Furniture",
        "superordinate": "Artifact",
        "domain": "Non-Living",
    },
    "n02948072": {
        "specific": "Candle",
        "coordinate": "Furniture",
        "superordinate": "Artifact",
        "domain": "Non-Living",
    },
    "n03045698": {
        "specific": "Phone",
        "coordinate": "Tools",
        "superordinate": "Artifact",
        "domain": "Non-Living",
    },
    "n03642806": {
        "specific": "Laptop",
        "coordinate": "Tools",
        "superordinate": "Artifact",
        "domain": "Non-Living",
    },
    "n03594945": {
        "specific": "Shoe",
        "coordinate": "Tools",
        "superordinate": "Artifact",
        "domain": "Non-Living",
    },
    "n02747177": {
        "specific": "Backpack",
        "coordinate": "Tools",
        "superordinate": "Artifact",
        "domain": "Non-Living",
    },
}


def extract_dataset(zip_path, extract_dir, max_images_per_class=500):
    os.makedirs(extract_dir, exist_ok=True)
    raw_images_dir = os.path.join(extract_dir, "raw")
    os.makedirs(raw_images_dir, exist_ok=True)

    extracted_records = []

    with zipfile.ZipFile(zip_path, "r") as z:
        all_files = z.namelist()

        found_classes = [
            wnid
            for wnid in TINY_IMAGENET_DRAWABLE_TAXONOMY
            if any(
                f"/train/{wnid}/" in f or f"/{wnid}/" in f for f in all_files
            )
        ]
        print(
            f"[*] Found {len(found_classes)} / {len(TINY_IMAGENET_DRAWABLE_TAXONOMY)} target classes in zip archive."
        )

        for wnid in tqdm(found_classes, desc="Extracting Images"):
            tax_info = TINY_IMAGENET_DRAWABLE_TAXONOMY[wnid]
            class_files = [
                f
                for f in all_files
                if f"/{wnid}/images/" in f and f.endswith(".JPEG")
            ][:max_images_per_class]

            for idx, file_path in enumerate(class_files):
                img_data = z.read(file_path)
                img = Image.open(io.BytesIO(img_data)).convert("RGB")

                clean_name = tax_info["specific"].lower().replace(" ", "_")
                local_filename = f"{clean_name}_{wnid}_{idx}.jpg"
                save_path = os.path.join(raw_images_dir, local_filename)
                img.save(save_path)

                extracted_records.append(
                    {
                        "filepath": save_path,
                        "filename": local_filename,
                        "wnid": wnid,
                        "specific": tax_info["specific"],
                        "coordinate": tax_info["coordinate"],
                        "superordinate": tax_info["superordinate"],
                        "domain": tax_info["domain"],
                    }
                )

    return extracted_records


def main():
    data_dir = os.path.join(PROJECT_ROOT, "data")
    zip_path = os.path.join(data_dir, "tiny-imagenet-200.zip")

    if not os.path.exists(zip_path):
        fallback_zip = os.path.join(data_dir, "raw", "tiny-imagenet-200.zip")
        if os.path.exists(fallback_zip):
            zip_path = fallback_zip
        else:
            raise FileNotFoundError(
                f"[!] Could not locate tiny-imagenet-200.zip at {zip_path}"
            )

    print(f"[*] Extracting dataset from: {zip_path}")
    records = extract_dataset(zip_path, data_dir, max_images_per_class=100)

    df = pd.DataFrame(records)
    csv_path = os.path.join(data_dir, "processed", "metadata_processed.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)

    tax_path = os.path.join(data_dir, "taxonomy_drawable_32.json")
    with open(tax_path, "w") as f:
        json.dump(TINY_IMAGENET_DRAWABLE_TAXONOMY, f, indent=4)

    print(
        f"\n[+] SUCCESS: Extracted {len(records)} images across {df['wnid'].nunique()} curated classes."
    )
    print(f"[+] Metadata saved to: {csv_path}")


if __name__ == "__main__":
    main()