'''
Pure-numpy helpers for working with boolean window masks.
'''

import numpy as np


def mask_to_start(mask):
    ''' Return the first selected time index in a boolean mask
    '''
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return None
    return int(idx[0])


def window_to_mask(start, length, series_length):
    ''' Convert a contiguous window start/length into a boolean mask
    '''
    mask = np.zeros(series_length, dtype=bool)
    mask[start:start + length] = True
    return mask


def compare_masks(mask_a, mask_b):
    ''' Compare two selected windows using overlap and centre-distance metrics
    '''
    mask_a = np.asarray(mask_a, dtype=bool)
    mask_b = np.asarray(mask_b, dtype=bool)

    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()

    iou = intersection / union if union > 0 else 0.0
    denom = min(mask_a.sum(), mask_b.sum())
    overlap_fraction = intersection / denom if denom > 0 else 0.0

    a_idx = np.where(mask_a)[0]
    b_idx = np.where(mask_b)[0]

    if len(a_idx) == 0 or len(b_idx) == 0:
        centre_distance = np.nan
    else:
        a_centre = 0.5 * (a_idx[0] + a_idx[-1])
        b_centre = 0.5 * (b_idx[0] + b_idx[-1])
        centre_distance = abs(a_centre - b_centre) / len(mask_a)

    return {
        "iou": float(iou),
        "overlap_fraction": float(overlap_fraction),
        "normalised_centre_distance": float(centre_distance),
    }
