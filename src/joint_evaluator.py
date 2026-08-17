import copy
import os
import clip
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class EvaluationImageDataset(Dataset):
    """Dataset loader for evaluation images with relative path resolution."""

    def __init__(self, metadata, preprocess, concept_col="specific"):
        self.metadata = metadata
        self.preprocess = preprocess

        if concept_col not in metadata.columns:
            for alt in ["specific", "coordinate", "concept", "label", "class"]:
                if alt in metadata.columns:
                    concept_col = alt
                    break
        self.concept_col = concept_col

        path_col = None
        for alt in [
            "filename",
            "image_path",
            "filepath",
            "file_path",
            "path",
            "img_path",
        ]:
            if alt in metadata.columns:
                path_col = alt
                break

        if not path_col:
            raise ValueError(
                f"[!] Could not find an image path column in metadata. Present columns: {list(metadata.columns)}"
            )

        self.valid_indices = []
        self.samples = []

        cwd = os.getcwd()
        search_dirs = [
            cwd,
            os.path.join(cwd, "data"),
            os.path.join(cwd, "data", "processed"),
            os.path.join(cwd, "data", "processed", "images"),
            os.path.join(cwd, "data", "images"),
            os.path.join(cwd, "data", "raw"),
            os.path.join(cwd, "data", "tiny-imagenet-200"),
        ]

        for idx, row in metadata.iterrows():
            raw_path = str(row.get(path_col, "")).strip()
            if not raw_path or raw_path.lower() == "nan":
                continue

            cleaned_rel = raw_path.lstrip("./")
            candidate_paths = [
                raw_path,
                os.path.abspath(raw_path),
                os.path.join(cwd, cleaned_rel),
            ]
            for s_dir in search_dirs:
                candidate_paths.append(os.path.join(s_dir, cleaned_rel))
                candidate_paths.append(
                    os.path.join(s_dir, os.path.basename(raw_path))
                )

            valid_path = None
            for p in candidate_paths:
                if os.path.isfile(p):
                    valid_path = p
                    break

            if valid_path:
                self.valid_indices.append(idx)
                self.samples.append((valid_path, row[self.concept_col]))

        if len(self.samples) == 0:
            raise ValueError("[!] No valid image paths found in metadata.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, concept = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image_tensor = self.preprocess(image)
        return image_tensor, concept


class JointSpaceEvaluator:
    """Evaluates CLIP representation degradation under magnitude pruning."""

    def __init__(
        self,
        metadata_path="./data/processed/metadata_processed.csv",
        model_name="ViT-B/32",
        device="cpu",
        batch_size=32,
        sample_frac=1.0,
        target_n=None,
        balance_taxonomically=True,
    ):
        self.device = torch.device(device)
        self.batch_size = batch_size

        if self.device.type == "cpu":
            torch.set_num_threads(os.cpu_count() or 4)

        print(f"[*] Loading metadata from: {metadata_path}")
        df = pd.read_csv(metadata_path)

        spec_col = "specific" if "specific" in df.columns else "concept"

        # Equal allocation across object classes
        if balance_taxonomically and spec_col in df.columns:
            n_classes = df[spec_col].nunique()

            if target_n is not None and target_n > 0:
                samples_per_class = max(1, target_n // n_classes)
            else:
                samples_per_class = df.groupby(spec_col).size().min()

            df = (
                df.groupby(spec_col, group_keys=False)
                .sample(n=samples_per_class, replace=True, random_state=42)
                .reset_index(drop=True)
            )
            print(
                f"[*] Taxonomically balanced: {len(df)} total samples strictly equalized at {samples_per_class} samples/class across {n_classes} classes."
            )
        elif target_n is not None and target_n > 0:
            df = df.sample(n=target_n, replace=True, random_state=42).reset_index(
                drop=True
            )
            print(f"[*] Oversampled dataset to {len(df)} samples (replace=True).")
        elif sample_frac < 1.0:
            df = df.sample(frac=sample_frac, random_state=42).reset_index(
                drop=True
            )
            print(f"[*] Downsampled dataset to {len(df)} samples.")

        self.metadata = df

        print(f"[*] Loading CLIP model ({model_name}) on {self.device}...")
        self.base_model, self.preprocess = clip.load(
            model_name, device=self.device
        )
        self.base_model.eval()

    def _apply_pruning(self, model, pruning_level):
        if pruning_level <= 0.0:
            return model

        pruned_model = copy.deepcopy(model)
        with torch.no_grad():
            for name, param in pruned_model.named_parameters():
                if "weight" in name and param.dim() > 1:
                    tensor = param.data
                    abs_tensor = torch.abs(tensor)
                    threshold = float(
                        np.quantile(abs_tensor.cpu().numpy(), pruning_level)
                    )
                    mask = abs_tensor > threshold
                    param.data.mul_(mask.float())
        return pruned_model

    def _extract_joint_features(self, model):
        model.eval()
        spec_col = (
            "specific" if "specific" in self.metadata.columns else "concept"
        )
        dataset = EvaluationImageDataset(
            self.metadata, self.preprocess, spec_col
        )
        eval_metadata = self.metadata.iloc[dataset.valid_indices].reset_index(
            drop=True
        )

        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
        )

        img_feats_list, concept_labels = [], []
        with torch.no_grad():
            for imgs, concepts in tqdm(
                dataloader, desc="Extracting Features", leave=False
            ):
                imgs = imgs.to(self.device)
                feats = model.encode_image(imgs)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                img_feats_list.append(feats.cpu())
                concept_labels.extend(concepts)

            img_feats = torch.cat(img_feats_list, dim=0)
            unique_concepts = sorted(list(set(concept_labels)))
            text_prompts = [f"a photo of a {c}" for c in unique_concepts]
            text_tokens = clip.tokenize(text_prompts).to(self.device)

            text_feats = model.encode_text(text_tokens)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

        return (
            img_feats,
            text_feats.cpu(),
            concept_labels,
            unique_concepts,
            eval_metadata,
        )

    def _compute_cka(self, X, Y):
        X = X - X.mean(dim=0, keepdim=True)
        Y = Y - Y.mean(dim=0, keepdim=True)
        hsic_xy = torch.norm(torch.matmul(X.T, Y), p="fro") ** 2
        hsic_xx = torch.norm(torch.matmul(X.T, X), p="fro") ** 2
        hsic_yy = torch.norm(torch.matmul(Y.T, Y), p="fro") ** 2
        denom = torch.sqrt(hsic_xx * hsic_yy)
        return (hsic_xy / denom).item() if denom != 0 else 0.0

    def _compute_npr(self, X, Y, k=5):
        dist_X = torch.cdist(X, X)
        dist_Y = torch.cdist(Y, Y)
        topk_X = torch.topk(dist_X, k=k + 1, largest=False).indices[:, 1:]
        topk_Y = torch.topk(dist_Y, k=k + 1, largest=False).indices[:, 1:]

        matches = sum(
            len(set(topk_X[i].tolist()).intersection(set(topk_Y[i].tolist())))
            for i in range(len(X))
        )
        return matches / (len(X) * k)

    def run_eval(self, pruning_levels=[0.0, 0.25, 0.50, 0.75, 0.90, 0.95]):
        print("[*] Extracting base unpruned features...")
        ref_img, ref_text, labels, unique_concepts, eval_meta = (
            self._extract_joint_features(self.base_model)
        )

        spec_col = (
            "specific" if "specific" in self.metadata.columns else "concept"
        )
        concept_to_idx = {c: i for i, c in enumerate(unique_concepts)}
        target_indices = torch.tensor([concept_to_idx[c] for c in labels])

        results = []

        for p_level in pruning_levels:
            print(f"\n[+] Evaluating Pruning Level: {p_level * 100:.1f}%")
            pruned_model = self._apply_pruning(self.base_model, p_level)
            p_img, p_text, _, _, _ = self._extract_joint_features(pruned_model)

            sim_matrix = torch.matmul(p_img, p_text.T)
            top1_preds = torch.argmax(sim_matrix, dim=1)
            correct_mask = top1_preds == target_indices

            # Hierarchical Accuracy Calculations
            spec_acc = correct_mask.float().mean().item()

            coordinate_acc, super_acc = spec_acc, spec_acc
            if (
                "coordinate" in eval_meta.columns
                and "superordinate" in eval_meta.columns
            ):
                coordinate_correct, super_correct = 0, 0
                for i, pred_idx in enumerate(top1_preds):
                    pred_c = unique_concepts[pred_idx.item()]
                    pred_match = self.metadata[
                        self.metadata[spec_col] == pred_c
                    ]
                    if not pred_match.empty:
                        p_row = pred_match.iloc[0]
                        t_row = eval_meta.iloc[i]
                        if p_row.get("coordinate") == t_row.get("coordinate"):
                            coordinate_correct += 1
                        if p_row.get("superordinate") == t_row.get(
                            "superordinate"
                        ):
                            super_correct += 1
                coordinate_acc = coordinate_correct / len(eval_meta)
                super_acc = super_correct / len(eval_meta)

            # MRR
            ranks = [
                1.0
                / (
                    (
                        torch.argsort(sim_matrix[i], descending=True)
                        == target_indices[i]
                    )
                    .nonzero(as_tuple=True)[0]
                    .item()
                    + 1
                )
                for i in range(len(target_indices))
            ]
            mrr = float(np.mean(ranks))

            # Error Taxonomy Classification
            coord_err, super_err, domain_err, collapse_err = 0, 0, 0, 0
            for i, is_corr in enumerate(correct_mask):
                if not is_corr:
                    pred_c = unique_concepts[top1_preds[i].item()]
                    t_row = eval_meta.iloc[i]
                    p_match = self.metadata[self.metadata[spec_col] == pred_c]
                    if not p_match.empty:
                        p_row = p_match.iloc[0]
                        if (
                            "coordinate" in t_row
                            and pred_c == t_row.get("coordinate")
                        ):
                            coord_err += 1
                        elif (
                            "superordinate" in t_row
                            and p_row.get("superordinate")
                            == t_row.get("superordinate")
                        ):
                            super_err += 1
                        elif (
                            "domain" in t_row
                            and p_row.get("domain") == t_row.get("domain")
                        ):
                            domain_err += 1
                        else:
                            collapse_err += 1
                    else:
                        collapse_err += 1

            results.append(
                {
                    "pruning_level": p_level,
                    "i2t_top1": spec_acc,
                    "top1_specific_acc": spec_acc,
                    "top1_coordinate_acc": coordinate_acc,
                    "top1_super_acc": super_acc,
                    "mrr": mrr,
                    "cka_vision": self._compute_cka(ref_img, p_img),
                    "cka_text": self._compute_cka(ref_text, p_text),
                    "npr_vision": self._compute_npr(ref_img, p_img, k=5),
                    "coordinate error": coord_err,
                    "superordinate error": super_err,
                    "domain error": domain_err,
                    "domain collapse": collapse_err,
                }
            )

        return pd.DataFrame(results)