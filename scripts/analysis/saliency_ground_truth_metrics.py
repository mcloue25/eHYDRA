'''
Metrics for comparing a computed attribution phi against the synthetic ground-truth attribution Phi
'''
import numpy as np


def cosine_similarity(phi, Phi):
    ''' Calculates cosine similarity 
    '''
    phi = np.asarray(phi, dtype=np.float64)
    Phi = np.asarray(Phi, dtype=np.float64)
    denom = np.linalg.norm(phi) * np.linalg.norm(Phi)
    if denom < 1e-12:
        return np.nan
    return float(np.dot(phi, Phi) / denom)


def confusion_matrix_metrics(phi, Phi, magnitude_tol=0.05):
    ''' Calculates confusion matrix & other metrics 
    Returns:
        precision, recall, f1 and the raw counts
    '''
    phi = np.asarray(phi, dtype=np.float64)
    Phi = np.asarray(Phi, dtype=np.float64)

    max_abs_phi = np.max(np.abs(phi))
    phi_relevant = np.abs(phi) > (magnitude_tol * max_abs_phi) if max_abs_phi > 0 else np.zeros_like(phi, dtype=bool)
    Phi_relevant = Phi != 0

    same_sign = np.sign(phi) == np.sign(Phi)

    true_relevant = phi_relevant & Phi_relevant & same_sign
    true_irrelevant = (~phi_relevant) & (~Phi_relevant)
    false_relevant = phi_relevant & ~(Phi_relevant & same_sign)
    false_irrelevant = (~phi_relevant) & Phi_relevant

    n_tr, n_ti = int(true_relevant.sum()), int(true_irrelevant.sum())
    n_fr, n_fi = int(false_relevant.sum()), int(false_irrelevant.sum())

    precision = n_tr / (n_tr + n_fr) if (n_tr + n_fr) > 0 else np.nan
    recall = n_tr / (n_tr + n_fi) if (n_tr + n_fi) > 0 else np.nan
    if np.isnan(precision) or np.isnan(recall) or (precision + recall) == 0:
        f1 = np.nan
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "precision": precision, "recall": recall, "f1": f1,
        "n_true_relevant": n_tr, "n_true_irrelevant": n_ti,
        "n_false_relevant": n_fr, "n_false_irrelevant": n_fi,
    }


def topk_overlap(phi, Phi, q):
    ''' Pointwise top-q% overlap between |phi| and the ground-truth nonzero-Phi region. 
    Returns:
        precision (fraction of the top-q% phi points that fall inside the true region), 
        recall (fraction of the true region covered by the top-q% phi points) 
        IoU
    '''
    phi = np.asarray(phi, dtype=np.float64)
    Phi = np.asarray(Phi, dtype=np.float64)
    T = len(phi)

    true_region = Phi != 0
    n_true = int(true_region.sum())
    if n_true == 0:
        return {"precision": np.nan, "recall": np.nan, "iou": np.nan, "window_size": 0}

    w = max(1, round(q * T))
    top_idx = np.argsort(-np.abs(phi))[:w]
    top_mask = np.zeros(T, dtype=bool)
    top_mask[top_idx] = True

    intersection = int((top_mask & true_region).sum())
    union = int((top_mask | true_region).sum())

    precision = intersection / w if w > 0 else np.nan
    recall = intersection / n_true
    iou = intersection / union if union > 0 else np.nan
    return {"precision": precision, "recall": recall, "iou": iou, "window_size": w}


def evaluate_one_sample(phi, Phi, topk_fractions=(0.05, 0.10, 0.20), magnitude_tol=0.05):
    ''' Call all funcs above for one sample
    '''
    result = {"cosine_similarity": cosine_similarity(phi, Phi)}
    cm = confusion_matrix_metrics(phi, Phi, magnitude_tol=magnitude_tol)
    result.update({f"confmat_{k}": v for k, v in cm.items()})
    for q in topk_fractions:
        tk = topk_overlap(phi, Phi, q)
        result.update({f"topk{int(q*100)}_{k}": v for k, v in tk.items()})
    return result
