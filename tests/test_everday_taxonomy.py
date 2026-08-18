import json
import os
import pandas as pd


def test_everyday_taxonomy():
    csv_path = "./data/processed/everyday_taxonomy.csv"
    json_path = "./data/processed/everyday_taxonomy.json"

    # 1. Validate file existence
    assert os.path.exists(csv_path), f"[!] Missing CSV: {csv_path}"
    assert os.path.exists(json_path), f"[!] Missing JSON: {json_path}"

    # 2. Validate DataFrame columns and non-null values
    df = pd.read_csv(csv_path)
    required_cols = {
        "wnid",
        "specific",
        "coordinate",
        "superordinate",
        "domain",
    }
    assert required_cols.issubset(
        set(df.columns)
    ), f"[!] Missing columns: {required_cols - set(df.columns)}"
    assert len(df) > 0, "[!] Taxonomy CSV is empty!"
    assert (
        df.isnull().sum().sum() == 0
    ), "[!] Found unexpected null values in metadata."

    # 3. Validate JSON format consistency
    with open(json_path, "r") as f:
        json_data = json.load(f)
    assert len(json_data) == len(
        df
    ), "[!] JSON key count does not match CSV row count!"

    print("=" * 60)
    print("[✓] ALL TAXONOMY SCHEMA ASSERTIONS PASSED")
    print("=" * 60)
    print(f"Total Verified Classes: {len(df)}")
    print("\nClass Counts by Superordinate Category:")
    print(df["superordinate"].value_counts().to_string())
    print("\nClass Counts by Domain:")
    print(df["domain"].value_counts().to_string())


if __name__ == "__main__":
    test_everyday_taxonomy()