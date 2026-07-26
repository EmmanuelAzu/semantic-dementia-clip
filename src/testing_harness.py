import os
import sys

# This forces Python to recognize the root project folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import pandas as pd
import clip
from tqdm import tqdm
from PIL import Image
from src.pruning_engine import CLIPPruningEngine

class TestingHarness:
    def __init__(self, metadata_path, index_tensor_path):
        """Initializes the evaluation environment."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.metadata = pd.read_csv(metadata_path)
        
        # Load the healthy, offline visual memory (Phase A output)
        self.visual_memory = torch.load(index_tensor_path).to(self.device)
        
        # Load the base model and retain the preprocessing pipeline
        self.base_model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.pruning_engine = CLIPPruningEngine(self.base_model)
        
    def classify_error(self, target_row, predicted_row):
        """
        Implements the Clinical Error Diagnostic.
        Compares the target taxonomy to the model's actual retrieval.
        """
        if target_row['specific'] == predicted_row['specific']:
            return "Correct"
        elif target_row['basic'] == predicted_row['basic']:
            return "Coordinate Error" 
        elif target_row['superordinate'] == predicted_row['superordinate']:
            return "Superordinate Error" 
        elif target_row['domain'] == predicted_row['domain']:
            return "Domain Error"
        else:
            return "Domain Collapse"
        
    # ... (Keep __init__ and classify_error exactly as they were) ...

    def run_simulation(self, encoder_type="text", max_pruning=0.9, step=0.1):
        """Runs the progressive atrophy simulation on the transmodal hub."""
        print(f"\n[*] Starting Atrophy Simulation | Target Hub: {encoder_type.upper()}")
        
        results = []
        pruning_levels = [round(x * step, 2) for x in range(int(max_pruning / step) + 1)]
        
        # We will test the model's ability to recall 1 unique class from each of the concepts
        test_queries = self.metadata.drop_duplicates(subset=['specific']).to_dict('records')
        print(f"[*] Generated {len(test_queries)} unique clinical queries for testing.")
        
        for p in pruning_levels:
            print(f"    -> Testing Atrophy Level: {p*100:.0f}%")
            
            # 1. Damage the network hub
            atrophied_model = self.pruning_engine.get_pruned_model(amount=p, encoder_type=encoder_type)
            atrophied_model.eval()
            
            # 2. Resolve Visual Memory Index based on Atrophy Target
            if encoder_type == "vision":
                print(f"       [*] Re-indexing visual memory with atrophied vision encoder (simulating visual agnosia)...")
                img_dir = "./data/raw"
                batch_images = []
                atrophied_memory_list = []
                
                # Re-encode all images using the damaged vision encoder in batches for performance
                for idx, row in tqdm(self.metadata.iterrows(), total=len(self.metadata), desc="       Encoding Images", leave=False):
                    img_path = os.path.join(img_dir, row['filename'])
                    if os.path.exists(img_path):
                        img = Image.open(img_path).convert("RGB")
                        img_t = self.preprocess(img)
                        batch_images.append(img_t)
                    
                    # Execute batch forward pass when batch is full or dataset ends
                    if len(batch_images) == 64 or idx == len(self.metadata) - 1:
                        if batch_images:
                            batch_tensor = torch.stack(batch_images).to(self.device)
                            with torch.no_grad():
                                img_feats = atrophied_model.encode_image(batch_tensor)
                                img_feats /= img_feats.norm(dim=-1, keepdim=True)
                                atrophied_memory_list.append(img_feats)
                            batch_images = []
                
                current_visual_memory = torch.cat(atrophied_memory_list, dim=0)
            else:
                # Text atrophy successfully uses the pre-cached, healthy visual index
                current_visual_memory = self.visual_memory
            
            level_results = {"Correct": 0, "Coordinate Error": 0, "Superordinate Error": 0, "Domain Error": 0, "Domain Collapse": 0}
            
            with torch.no_grad():
                for query_item in test_queries:
                    # Create textual probe
                    text_input = clip.tokenize([f"a photo of a {query_item['specific']}"]).to(self.device)
                    
                    # Generate text embedding (atrophied if text encoder is selected, healthy if vision)
                    text_features = atrophied_model.encode_text(text_input)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    
                    # Compute Cosine Similarity against the active visual memory layout
                    similarities = (100.0 * text_features @ current_visual_memory.T).softmax(dim=-1)
                    
                    # Get top 1 prediction index
                    best_match_idx = similarities.argmax(dim=-1).item()
                    predicted_row = self.metadata.iloc[best_match_idx]
                    
                    # Categorize the clinical error
                    error_type = self.classify_error(query_item, predicted_row)
                    level_results[error_type] += 1
            
            # Store data for this pruning level
            results.append({
                "Pruning_Level": p,
                **level_results
            })
            
        return pd.DataFrame(results)

if __name__ == "__main__":
    harness = TestingHarness(
        metadata_path="./data/processed/metadata_processed.csv",
        index_tensor_path="./data/processed/image_index.pt"
    )
    
    df_results = harness.run_simulation(encoder_type="text")
    
    os.makedirs("./data/results", exist_ok=True)
    df_results.to_csv("./data/results/aphasia_simulation.csv", index=False)
    print("\n[+] Simulation complete! Results saved to ./data/results/aphasia_simulation.csv")
    print("\n--- CLINICAL DECAY RESULTS ---")
    print(df_results)