'''
Metrics for comparing saliency maps to each other 
'''

import numpy as np
from scipy.stats import pearsonr, spearmanr

from utils.explainability import select_contiguous_window


def pearson_correlation(saliency_a: np.ndarray, saliency_b: np.ndarray):
    ''' Pearson correlation between two saliency vectors of the same length
    '''
    a = np.asarray(saliency_a, dtype=np.float64)
    b = np.asarray(saliency_b, dtype=np.float64)

    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

    if np.std(a) == 0 or np.std(b) == 0:
        # A constant saliency map has undefined correlation with anything.
        return np.nan

    r, _ = pearsonr(a, b)
    return float(r)


def spearman_correlation(saliency_a: np.ndarray, saliency_b: np.ndarray):
    ''' Spearman rank correlation between two saliency vectors of the same length
    '''
    a = np.asarray(saliency_a, dtype=np.float64)
    b = np.asarray(saliency_b, dtype=np.float64)

    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan

    rho, _ = spearmanr(a, b)
    return float(rho)


def top_window_iou(saliency_a: np.ndarray, saliency_b: np.ndarray, fraction: float = 0.10):
    ''' IoU between the top-saliency contiguous windows selected from two saliency maps.
    '''
    from utils.mask_utils import compare_masks

    importance_a = np.abs(np.asarray(saliency_a, dtype=np.float32))
    importance_b = np.abs(np.asarray(saliency_b, dtype=np.float32))

    mask_a = select_contiguous_window(importance_a, fraction=fraction, mode="top")
    mask_b = select_contiguous_window(importance_b, fraction=fraction, mode="top")
    return compare_masks(mask_a, mask_b)["iou"]


def saliency_concentration(saliency: np.ndarray, eps: float = 1e-12):
    ''' Normalised Shannon entropy and a Gini-style concentration score for one map
    '''
    importance = np.abs(np.asarray(saliency, dtype=np.float64))
    T = len(importance)

    total = importance.sum()
    if total <= eps or T <= 1:
        return {"entropy": np.nan, "normalised_entropy": np.nan, "concentration": np.nan}

    p = importance / total
    p_nonzero = p[p > eps]
    entropy = float(-(p_nonzero * np.log(p_nonzero)).sum())
    max_entropy = np.log(T)
    normalised_entropy = entropy / max_entropy if max_entropy > 0 else np.nan

    return {
        "entropy": entropy,
        "normalised_entropy": normalised_entropy,
        "concentration": 1.0 - normalised_entropy if not np.isnan(normalised_entropy) else np.nan,
    }
