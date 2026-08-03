import os
import sys

# Dynamically resolve and set project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import torch
import pandas as pd
import clip
from tqdm import tqdm
from PIL import Image
from src.pruning_engine import CLIPPruningEngine

class TestingHarness:
    def __init__(self, metadata_path=None, index_tensor_path=None):
        """Initializes the Semantic Dementia multimodal evaluation environment."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if metadata_path is None:
            metadata_path = os.path.join(PROJECT_ROOT, "data", "processed", "metadata_processed.csv")
        if index_tensor_path is None:
            index_tensor_path = os.path.join(PROJECT_ROOT, "data", "processed", "image_index.pt")
            
        self.metadata_path = metadata_path
        self.index_tensor_path = index_tensor_path
        
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found at {self.metadata_path}. Run build_general_dataset.py first.")
            
        self.metadata = pd.read_csv(self.metadata_path)
        
        # Load baseline visual memory index if pre-cached
        if os.path.exists(self.index_tensor_path):
            self.baseline_visual_memory = torch.load(self.index_tensor_path, map_location=self.device)
        else:
            self.baseline_visual_memory = None
        
        # Load CLIP ViT-B/32 backbone
        self.base_model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.pruning_engine = CLIPPruningEngine(self.base_model)
        
        # Standard CLIP Prompt Templates for Ensembling (Reduces zero-shot baseline noise)
        self.prompt_templates = [
            "a photo of a {}",
            "a close-up photo of a {}",
            "a rendering of a {}",
            "a picture of a {}",
            "a good photo of a {}"
        ]
        
        # Hierarchical Taxonomic Penalty Scale
        self.distance_costs = {
            "Correct": 0.0,
            "Coordinate Error": 1.0,
            "Superordinate Error": 2.0,
            "Domain Error": 3.0,
            "Domain Collapse": 4.0
        }

    def classify_error(self, target_row: dict, predicted_row: pd.Series) -> str:
        """
        Implements the Clinical Diagnostic across 5 hierarchical error tiers.
        Safely falls back between 'coordinate' and legacy 'basic' schema column names.
        """
        target_coord = target_row.get('coordinate', target_row.get('basic'))
        pred_coord = predicted_row.get('coordinate', predicted_row.get('basic'))

        if target_row['specific'] == predicted_row['specific']:
            return "Correct"
        elif target_coord is not None and target_coord == pred_coord:
            return "Coordinate Error" 
        elif target_row['superordinate'] == predicted_row['superordinate']:
            return "Superordinate Error" 
        elif target_row['domain'] == predicted_row['domain']:
            return "Domain Error"
        else:
            return "Domain Collapse"

    def _reindex_visual_memory(self, model, img_dir=None, batch_size=64):
        """Encodes visual memory using a specified atrophied or healthy model state."""
        if img_dir is None:
            img_dir = os.path.join(PROJECT_ROOT, "data", "raw")
            
        batch_images = []
        memory_chunks = []
        
        for idx, row in self.metadata.iterrows():
            img_path = os.path.join(img_dir, row['filename'])
            if os.path.exists(img_path):
                img = Image.open(img_path).convert("RGB")
                batch_images.append(self.preprocess(img))
            
            if len(batch_images) == batch_size or idx == len(self.metadata) - 1:
                if batch_images:
                    batch_tensor = torch.stack(batch_images).to(self.device)
                    with torch.no_grad():
                        feats = model.encode_image(batch_tensor)
                        feats = feats / feats.norm(dim=-1, keepdim=True)
                        memory_chunks.append(feats)
                    batch_images = []
                    
        if memory_chunks:
            return torch.cat(memory_chunks, dim=0)
        else:
            raise FileNotFoundError(f"No valid image files were found in {img_dir} to index.")

    def run_simulation(self, encoder_type="joint", max_pruning=0.9, step=0.1, batch_size=64, top_k=5):
        """
        Runs progressive hub atrophy simulation.
        
        Parameters:
            encoder_type (str):
                - "text": Simulates verbal semantic loss (svPPA)
                - "vision": Simulates visual agnosia
                - "joint": Simulates true Amodal Semantic Dementia (Bilateral ATL Hub breakdown)
            max_pruning (float): Maximum fraction of projection weights zeroed out (e.g. 0.9 = 90%)
            step (float): Pruning increment step
            batch_size (int): GPU batch size for memory re-indexing
            top_k (int): Size of nearest-neighbor neighborhood for continuous cost tracking
        """
        print(f"\n[*] Starting Atrophy Simulation | Mode: {encoder_type.upper()}")
        
        results = []
        pruning_levels = [round(x * step, 2) for x in range(int(max_pruning / step) + 1)]
        
        # Extract unique specific concepts for testing
        test_queries = self.metadata.drop_duplicates(subset=['specific']).to_dict('records')
        print(f"[*] Probing {len(test_queries)} unique specific categories across {len(pruning_levels)} atrophy steps...")
        
        for p in pruning_levels:
            print(f"    -> Atrophy Level: {p*100:.0f}%")
            
            # 1. Apply structured L2 pruning to target projection heads
            if encoder_type == "joint":
                # Prune both modality projection heads to simulate central amodal hub degeneration
                atrophied_model = self.pruning_engine.get_pruned_model(amount=p, encoder_type="text")
                atrophied_model = self.pruning_engine.get_pruned_model(amount=p, encoder_type="vision", model=atrophied_model)
            else:
                atrophied_model = self.pruning_engine.get_pruned_model(amount=p, encoder_type=encoder_type)
                
            atrophied_model.eval()
            
            # 2. Resolve Active Visual Memory Index
            if encoder_type in ["vision", "joint"]:
                current_visual_memory = self._reindex_visual_memory(atrophied_model, batch_size=batch_size)
            else:
                if self.baseline_visual_memory is None:
                    print("       [*] Generating healthy visual memory baseline...")
                    self.baseline_visual_memory = self._reindex_visual_memory(self.base_model, batch_size=batch_size)
                current_visual_memory = self.baseline_visual_memory

            # Metrics accumulators
            top1_counts = {
                "Correct": 0, 
                "Coordinate Error": 0, 
                "Superordinate Error": 0, 
                "Domain Error": 0, 
                "Domain Collapse": 0
            }
            total_expected_taxonomic_cost = 0.0

            # 3. Clinical Diagnostic Pass (Prompt Ensembling + Top-K Neighborhood Evaluation)
            with torch.no_grad():
                for query_item in test_queries:
                    # Construct multi-prompt ensemble representation
                    prompts = [tmpl.format(query_item['specific']) for tmpl in self.prompt_templates]
                    text_inputs = clip.tokenize(prompts).to(self.device)
                    
                    # Encode and normalize ensembled representation
                    text_features = atrophied_model.encode_text(text_inputs)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    ensembled_vector = text_features.mean(dim=0, keepdim=True)
                    ensembled_vector = ensembled_vector / ensembled_vector.norm(dim=-1, keepdim=True)
                    
                    # Compute Cosine Similarity against active visual memory space
                    similarities = (100.0 * ensembled_vector @ current_visual_memory.T).softmax(dim=-1)
                    
                    # Retrieve Top-K nearest neighbor indices
                    topk_indices = torch.topk(similarities, k=top_k, dim=-1).indices.squeeze(0).tolist()
                    
                    # Top-1 Categorical Diagnosis
                    top1_pred_row = self.metadata.iloc[topk_indices[0]]
                    top1_error = self.classify_error(query_item, top1_pred_row)
                    top1_counts[top1_error] += 1
                    
                    # Top-K Neighborhood Expected Taxonomic Cost Calculation
                    query_neighborhood_cost = 0.0
                    for neighbor_idx in topk_indices:
                        neighbor_row = self.metadata.iloc[neighbor_idx]
                        err = self.classify_error(query_item, neighbor_row)
                        query_neighborhood_cost += self.distance_costs[err]
                    
                    total_expected_taxonomic_cost += (query_neighborhood_cost / top_k)
            
            mean_taxonomic_cost = total_expected_taxonomic_cost / len(test_queries)
            
            results.append({
                "Pruning_Level": p,
                **top1_counts,
                "Expected_Taxonomic_Cost": round(mean_taxonomic_cost, 4)
            })
            
        return pd.DataFrame(results)

if __name__ == "__main__":
    harness = TestingHarness()
    
    # Run joint amodal simulation (true Semantic Dementia)
    df_sd = harness.run_simulation(encoder_type="joint", step=0.1, top_k=5)
    
    results_dir = os.path.join(PROJECT_ROOT, "data", "results")
    os.makedirs(results_dir, exist_ok=True)
    
    out_csv = os.path.join(results_dir, "amodal_sd_simulation.csv")
    df_sd.to_csv(out_csv, index=False)
    
    print(f"\n[+] True Semantic Dementia (Joint Hub) Simulation Complete! Results saved to {out_csv}")
    print("\n--- CLINICAL SD DECAY TRAJECTORY ---")
    print(df_sd[["Pruning_Level", "Correct", "Coordinate Error", "Superordinate Error", "Domain Error", "Domain Collapse", "Expected_Taxonomic_Cost"]])