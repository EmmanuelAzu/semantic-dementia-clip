import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import torch
import pandas as pd
import clip
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
        self.valid_metadata = self.metadata.copy()
        
        if os.path.exists(self.index_tensor_path):
            self.baseline_visual_memory = torch.load(self.index_tensor_path, map_location=self.device)
        else:
            self.baseline_visual_memory = None
        
        self.base_model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.pruning_engine = CLIPPruningEngine(self.base_model)
        
        self.prompt_templates = [
            "a photo of a {}",
            "a close-up photo of a {}",
            "a rendering of a {}",
            "a picture of a {}",
            "a good photo of a {}",
            "a cropped photo of the {}",
            "a bright photo of a {}",
            "a photo of the small {}",
            "a photo of the large {}",
            "a detailed photo of a {}"
        ]
        
        self.distance_costs = {
            "Correct": 0.0,
            "Coordinate Error": 1.0,
            "Superordinate Error": 2.0,
            "Domain Error": 3.0,
            "Domain Collapse": 4.0
        }

    def classify_error(self, target_row: dict, predicted_row: pd.Series) -> str:
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

    def _reindex_visual_memory(self, model, img_dir=None, batch_size=256):
        """Encodes visual memory using a specified atrophied or healthy model state."""
        if img_dir is None:
            img_dir = os.path.join(PROJECT_ROOT, "data", "raw")
            
        batch_images = []
        memory_chunks = []
        valid_rows = []
        
        for idx, row in self.metadata.iterrows():
            img_path = row.get("filepath", None)
            if not img_path or pd.isna(img_path) or not os.path.exists(img_path):
                filename = row.get("filename", "")
                img_path = os.path.join(img_dir, filename)
                if not os.path.exists(img_path):
                    for root, _, files in os.walk(img_dir):
                        if filename in files:
                            img_path = os.path.join(root, filename)
                            break
            
            if os.path.exists(img_path):
                img = Image.open(img_path).convert("RGB")
                batch_images.append(self.preprocess(img))
                valid_rows.append(idx)
            
            if len(batch_images) == batch_size or idx == len(self.metadata) - 1:
                if batch_images:
                    batch_tensor = torch.stack(batch_images).to(self.device)
                    with torch.no_grad():
                        feats = model.encode_image(batch_tensor)
                        feats = feats / feats.norm(dim=-1, keepdim=True)
                        memory_chunks.append(feats)
                    batch_images = []
                    
        if memory_chunks:
            self.valid_metadata = self.metadata.iloc[valid_rows].reset_index(drop=True)
            return torch.cat(memory_chunks, dim=0)
        else:
            raise FileNotFoundError(
                f"No valid image files found in '{img_dir}' matching metadata records. "
                f"Please ensure dataset images exist or run build_general_dataset.py."
            )

    def run_simulation(self, encoder_type="joint", max_pruning=0.9, step=0.1, batch_size=256, top_k=5):
        print(f"\n[*] Starting Atrophy Simulation | Mode: {encoder_type.upper()} | Batch Size: {batch_size}")
        
        test_queries = self.metadata.drop_duplicates(subset=['specific']).to_dict('records')
        num_queries = len(test_queries)
        num_templates = len(self.prompt_templates)
        
        print(f"[*] Probe Size: {num_queries} categories x {num_templates} templates = {num_queries * num_templates} total prompt vectors.")
        
        results = []
        pruning_levels = [round(x * step, 2) for x in range(int(max_pruning / step) + 1)]
        
        for p in pruning_levels:
            print(f"    -> Atrophy Level: {p*100:.0f}%")
            
            if encoder_type == "joint":
                atrophied_model = self.pruning_engine.get_pruned_model(amount=p, encoder_type="text")
                atrophied_model = self.pruning_engine.get_pruned_model(amount=p, encoder_type="vision", model=atrophied_model)
            else:
                atrophied_model = self.pruning_engine.get_pruned_model(amount=p, encoder_type=encoder_type)
                
            atrophied_model.eval()
            
            if encoder_type in ["vision", "joint"]:
                current_visual_memory = self._reindex_visual_memory(atrophied_model, batch_size=batch_size)
            else:
                if self.baseline_visual_memory is None:
                    print("       [*] Generating healthy visual memory baseline...")
                    self.baseline_visual_memory = self._reindex_visual_memory(self.base_model, batch_size=batch_size)
                current_visual_memory = self.baseline_visual_memory

            all_prompts = []
            for q in test_queries:
                for tmpl in self.prompt_templates:
                    all_prompts.append(tmpl.format(q['specific']))
            
            all_tokens = clip.tokenize(all_prompts).to(self.device)
            
            encoded_text_chunks = []
            with torch.no_grad():
                for i in range(0, len(all_tokens), batch_size):
                    batch_tokens = all_tokens[i:i + batch_size]
                    feats = atrophied_model.encode_text(batch_tokens)
                    feats = feats / feats.norm(dim=-1, keepdim=True)
                    encoded_text_chunks.append(feats)
                    
            all_text_feats = torch.cat(encoded_text_chunks, dim=0)
            all_text_feats = all_text_feats.view(num_queries, num_templates, -1)
            ensembled_vectors = all_text_feats.mean(dim=1)
            ensembled_vectors = ensembled_vectors / ensembled_vectors.norm(dim=-1, keepdim=True)
            
            num_images = current_visual_memory.shape[0]
            k_adj = min(top_k, num_images)

            with torch.no_grad():
                sim_matrix = (100.0 * ensembled_vectors @ current_visual_memory.T).softmax(dim=-1)
                topk_indices = torch.topk(sim_matrix, k=k_adj, dim=-1).indices.cpu().numpy()
            
            top1_counts = {
                "Correct": 0, 
                "Coordinate Error": 0, 
                "Superordinate Error": 0, 
                "Domain Error": 0, 
                "Domain Collapse": 0
            }
            total_expected_taxonomic_cost = 0.0

            for q_idx, query_item in enumerate(test_queries):
                query_topk = topk_indices[q_idx]
                
                top1_row = self.valid_metadata.iloc[query_topk[0]]
                top1_err = self.classify_error(query_item, top1_row)
                top1_counts[top1_err] += 1
                
                q_cost = 0.0
                for n_idx in query_topk:
                    neighbor_row = self.valid_metadata.iloc[n_idx]
                    err = self.classify_error(query_item, neighbor_row)
                    q_cost += self.distance_costs[err]
                
                total_expected_taxonomic_cost += (q_cost / k_adj)

            mean_taxonomic_cost = total_expected_taxonomic_cost / num_queries
            
            results.append({
                "Pruning_Level": p,
                **top1_counts,
                "Expected_Taxonomic_Cost": round(mean_taxonomic_cost, 4)
            })
            
        return pd.DataFrame(results)

if __name__ == "__main__":
    harness = TestingHarness()
    
    df_sd = harness.run_simulation(encoder_type="joint", step=0.1, batch_size=256, top_k=10)
    
    results_dir = os.path.join(PROJECT_ROOT, "data", "results")
    os.makedirs(results_dir, exist_ok=True)
    
    out_csv = os.path.join(results_dir, "amodal_sd_simulation.csv")
    df_sd.to_csv(out_csv, index=False)
    
    print(f"\n[+] True Semantic Dementia (Joint Hub) Simulation Complete! Saved to {out_csv}")
    print("\n--- CLINICAL SD DECAY TRAJECTORY ---")
    print(df_sd[["Pruning_Level", "Correct", "Coordinate Error", "Superordinate Error", "Domain Error", "Domain Collapse", "Expected_Taxonomic_Cost"]])