import copy
import os
import clip
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from src.metrics import (
    compute_category_breakdown,
    compute_cka,
    compute_hierarchical_breakdown,
    compute_mrr,
    compute_neighborhood_preservation,
    compute_top10_breakdown_depth,
)
from src.pruning_engine import CLIPPruningEngine

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class EvaluationImageDataset(Dataset):
    def __init__(self, metadata, preprocess, spec_col):
        self.metadata = metadata
        self.preprocess = preprocess
        self.spec_col = spec_col
        self.valid_rows = []
        self.valid_indices = []

        for idx, row in metadata.iterrows():
            img_path = row.get("filepath", None)
            if not img_path or pd.isna(img_path):
                if "filename" in row and pd.notna(row["filename"]):
                    img_path = os.path.join(PROJECT_ROOT, "data", "raw", row["filename"])
                else:
                    continue
            if not os.path.isabs(img_path):
                img_path = os.path.join(PROJECT_ROOT, img_path)

            if os.path.exists(img_path):
                self.valid_rows.append((img_path, row[self.spec_col]))
                self.valid_indices.append(idx)

    def __len__(self):
        return len(self.valid_rows)

    def __getitem__(self, idx):
        img_path, concept = self.valid_rows[idx]
        image = self.preprocess(Image.open(img_path).convert("RGB"))
        return image, concept


class JointSpaceEvaluator:
    def __init__(self, metadata_path=None, device=None, sample_frac=1.0, seed=42, batch_size=64):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        
        if metadata_path is None:
            metadata_path = os.path.join(PROJECT_ROOT, "data", "processed", "metadata_processed.csv")
        full_meta = pd.read_csv(metadata_path)

        if sample_frac < 1.0:
            self.metadata = full_meta.sample(frac=sample_frac, random_state=seed).reset_index(drop=True)
            print(f"[*] Subsampled dataset to {len(self.metadata)} items ({sample_frac*100}%).")
        else:
            self.metadata = full_meta.reset_index(drop=True)
            print(f"[*] Loaded full evaluation dataset with {len(self.metadata)} samples.")

        self.base_model, self.preprocess = clip.load("ViT-B/32", device=self.device)

    def _extract_joint_features(self, model):
        model.eval()
        spec_col = "specific" if "specific" in self.metadata.columns else "concept"
        
        dataset = EvaluationImageDataset(self.metadata, self.preprocess, spec_col)
        if len(dataset) == 0:
            raise ValueError("[!] No valid images found in evaluation dataset.")

        # Sync self.metadata strictly to valid extracted rows
        self.metadata = self.metadata.iloc[dataset.valid_indices].reset_index(drop=True)

        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=0, pin_memory=True)

        img_feats_list, concept_labels = [], []
        
        with torch.no_grad():
            for imgs, concepts in dataloader:
                imgs = imgs.to(self.device)
                feats = model.encode_image(imgs)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                img_feats_list.append(feats)
                concept_labels.extend(concepts)

            img_feats = torch.cat(img_feats_list, dim=0)

            unique_concepts = sorted(list(set(concept_labels)))
            text_prompts = [f"a photo of a {c}" for c in unique_concepts]
            text_tokens = clip.tokenize(text_prompts).to(self.device)

            text_feats = model.encode_text(text_tokens)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

        return img_feats, text_feats, concept_labels, unique_concepts

    def run_eval(self, pruning_levels=[0.0, 0.25, 0.50, 0.75, 0.90, 0.95]):
        results = []
        base_img_feats, base_text_feats, _, _ = self._extract_joint_features(self.base_model)
        base_img_np = base_img_feats.float().cpu().numpy()
        base_text_np = base_text_feats.float().cpu().numpy()

        for p in pruning_levels:
            p_val = float(p)
            model_copy = copy.deepcopy(self.base_model)
            engine = CLIPPruningEngine(model_copy)
            pruned_model = engine.get_pruned_model(amount=p_val, encoder_type="joint")

            img_feats, text_feats, concept_labels, unique_concepts = self._extract_joint_features(pruned_model)
            img_np = img_feats.float().cpu().numpy()
            text_np = text_feats.float().cpu().numpy()

            sim_matrix = (img_feats @ text_feats.T).float().cpu().numpy()
            concept_to_idx = {c: idx for idx, c in enumerate(unique_concepts)}
            targets = np.array([concept_to_idx[c] for c in concept_labels])

            top1_i2t = float((np.argmax(sim_matrix, axis=1) == targets).mean())
            top5_k = min(5, sim_matrix.shape[1])
            top5_i2t = float(
                np.mean([targets[i] in np.argsort(sim_matrix[i])[-top5_k:] for i in range(len(targets))])
            )

            t2i_top1_accs = []
            for c_idx in range(len(unique_concepts)):
                matching_imgs = np.where(targets == c_idx)[0]
                if len(matching_imgs) > 0:
                    retrieved_img = np.argmax(sim_matrix[:, c_idx])
                    t2i_top1_accs.append(retrieved_img in matching_imgs)
            top1_t2i = float(np.mean(t2i_top1_accs)) if t2i_top1_accs else 0.0

            mean_cosine = float(np.mean([sim_matrix[i, targets[i]] for i in range(len(targets))]))

            mrr_score = compute_mrr(sim_matrix, targets)
            cka_vision = compute_cka(base_img_np, img_np)
            cka_text = compute_cka(base_text_np, text_np)
            npr_vision = compute_neighborhood_preservation(base_img_np, img_np, k=min(5, len(img_np) - 1))
            cat_accs = compute_category_breakdown(sim_matrix, targets, self.metadata)
            hier_metrics = compute_hierarchical_breakdown(sim_matrix, targets, self.metadata)
            top10_metrics = compute_top10_breakdown_depth(sim_matrix, targets, self.metadata, unique_concepts)

            row_dict = {
                "pruning_level": p_val,
                "i2t_top1": top1_i2t,
                "i2t_top5": top5_i2t,
                "t2i_top1": top1_t2i,
                "mean_cosine": mean_cosine,
                "mrr": mrr_score,
                "cka_vision": cka_vision,
                "cka_text": cka_text,
                "npr_vision": npr_vision,
            }
            row_dict.update(hier_metrics)
            row_dict.update(top10_metrics)

            for cat, acc in cat_accs.items():
                row_dict[f"acc_{cat}"] = acc

            results.append(row_dict)

        return pd.DataFrame(results)