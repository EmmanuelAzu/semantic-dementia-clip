import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

def compute_mrr(sim_matrix, targets):
    sorted_indices = np.argsort(-sim_matrix, axis=1)
    ranks = np.where(sorted_indices == targets[:, None])[1] + 1
    return float(np.mean(1.0 / ranks))

def compute_cka(X, Y):
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    dot_product = np.linalg.norm(Y.T @ X, "fro") ** 2
    norm_x = np.linalg.norm(X.T @ X, "fro")
    norm_y = np.linalg.norm(Y.T @ Y, "fro")
    if norm_x == 0 or norm_y == 0:
        return 0.0
    return float(dot_product / (norm_x * norm_y))

def compute_neighborhood_preservation(X, Y, k=5):
    if len(X) <= 1:
        return 0.0
    k_adj = min(k + 1, len(X))
    nbrs_X = NearestNeighbors(n_neighbors=k_adj).fit(X).kneighbors(X, return_distance=False)[:, 1:]
    nbrs_Y = NearestNeighbors(n_neighbors=k_adj).fit(Y).kneighbors(Y, return_distance=False)[:, 1:]
    intersection = [len(set(nx).intersection(set(ny))) for nx, ny in zip(nbrs_X, nbrs_Y)]
    return float(np.mean(intersection) / max(1, k_adj - 1))

def compute_category_breakdown(sim_matrix, targets, metadata):
    cat_col = next((c for c in ["domain", "basic", "superordinate"] if c in metadata.columns), None)
    if not cat_col:
        return {}

    preds = np.argmax(sim_matrix, axis=1)
    correct = (preds == targets)

    meta_aligned = metadata.reset_index(drop=True)
    cat_accs = {}
    for cat, group in meta_aligned.groupby(cat_col):
        indices = group.index.values
        if len(indices) > 0 and max(indices) < len(correct):
            cat_accs[str(cat)] = float(correct[indices].mean())
    return cat_accs

def compute_hierarchical_breakdown(sim_matrix, targets, metadata):
    meta_aligned = metadata.reset_index(drop=True)
    spec_col = next((c for c in ["specific", "concept"] if c in meta_aligned.columns), "specific")
    basic_col = next((c for c in ["basic", "coordinate"] if c in meta_aligned.columns), "basic")
    domain_col = next((c for c in ["domain", "superordinate"] if c in meta_aligned.columns), "domain")

    preds = np.argmax(sim_matrix, axis=1)
    unique_concepts = sorted(list(meta_aligned[spec_col].unique()))
    pred_concepts = [unique_concepts[p] for p in preds]

    concept_to_basic = meta_aligned.drop_duplicates(spec_col).set_index(spec_col)[basic_col].to_dict()
    concept_to_domain = meta_aligned.drop_duplicates(spec_col).set_index(spec_col)[domain_col].to_dict()

    true_concepts = meta_aligned[spec_col].values
    true_basics = [concept_to_basic.get(c) for c in true_concepts]
    true_domains = [concept_to_domain.get(c) for c in true_concepts]

    pred_basics = [concept_to_basic.get(c) for c in pred_concepts]
    pred_domains = [concept_to_domain.get(c) for c in pred_concepts]

    correct_count = int(sum(tc == pc for tc, pc in zip(true_concepts, pred_concepts)))
    coord_err_count = int(sum((tc != pc) and (tb == pb) for tc, pc, tb, pb in zip(true_concepts, pred_concepts, true_basics, pred_basics)))
    super_err_count = int(sum((tb != pb) and (td == pd) for tb, pb, td, pd in zip(true_basics, pred_basics, true_domains, pred_domains)))
    domain_err_count = int(sum(td != pd for td, pd in zip(true_domains, pred_domains)))

    total = len(true_concepts)
    total_errors = total - correct_count

    return {
        "top1_specific_acc": float(correct_count / total) if total > 0 else 0.0,
        "top1_basic_acc": float(sum(tb == pb for tb, pb in zip(true_basics, pred_basics)) / total) if total > 0 else 0.0,
        "top1_domain_acc": float(sum(td == pd for td, pd in zip(true_domains, pred_domains)) / total) if total > 0 else 0.0,
        "pct_superordinate_errors": float(super_err_count / total_errors) if total_errors > 0 else 0.0,
        "pct_domain_collapse_errors": float(domain_err_count / total_errors) if total_errors > 0 else 0.0,
        "Correct": correct_count,
        "Coordinate Error": coord_err_count,
        "Superordinate Error": super_err_count,
        "Domain Error": domain_err_count,
        "Domain Collapse": domain_err_count,
    }

def compute_top10_breakdown_depth(sim_matrix, targets, metadata, unique_concepts):
    meta_aligned = metadata.reset_index(drop=True)
    spec_col = next((c for c in ["specific", "concept"] if c in meta_aligned.columns), "specific")
    basic_col = next((c for c in ["basic", "coordinate"] if c in meta_aligned.columns), "basic")
    domain_col = next((c for c in ["domain", "superordinate"] if c in meta_aligned.columns), "domain")

    concept_to_basic = meta_aligned.drop_duplicates(spec_col).set_index(spec_col)[basic_col].to_dict()
    concept_to_domain = meta_aligned.drop_duplicates(spec_col).set_index(spec_col)[domain_col].to_dict()

    top10_k = min(10, sim_matrix.shape[1])
    top10_indices = np.argsort(-sim_matrix, axis=1)[:, :top10_k]

    same_basic_counts, same_domain_counts = [], []

    for i, top_k in enumerate(top10_indices):
        target_concept = unique_concepts[targets[i]]
        target_basic = concept_to_basic.get(target_concept)
        target_domain = concept_to_domain.get(target_concept)

        retrieved_concepts = [unique_concepts[idx] for idx in top_k]
        retrieved_basics = [concept_to_basic.get(c) for c in retrieved_concepts]
        retrieved_domains = [concept_to_domain.get(c) for c in retrieved_concepts]

        same_basic_counts.append(sum(1 for b in retrieved_basics if b == target_basic) / float(top10_k))
        same_domain_counts.append(sum(1 for d in retrieved_domains if d == target_domain) / float(top10_k))

    return {
        "top10_basic_ratio": float(np.mean(same_basic_counts)),
        "top10_domain_ratio": float(np.mean(same_domain_counts)),
    }