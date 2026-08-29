'''
Perturbation operators used to replace a selected window of a time series.
'''

import numpy as np
from scipy.ndimage import uniform_filter1d


def validate(x: np.ndarray, mask: np.ndarray):
    x = np.asarray(x, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)

    if x.shape != mask.shape:
        raise ValueError(f"x and mask must have the same shape, got {x.shape} vs {mask.shape}")

    return x, mask


def global_mean_perturbation(x: np.ndarray, mask: np.ndarray, rng: np.random.Generator | None = None):
    ''' Replace the masked window with the mean of the whole series.
    '''
    x, mask = validate(x, mask)
    x = x.copy()
    x[mask] = float(np.mean(x))
    return x


def context_window_bounds(mask: np.ndarray, context_multiplier: float = 1.0):
    ''' Return (left_start, left_end, right_start, right_end) index bounds for the
        unmasked context immediately surrounding a contiguous masked window.
    '''
    idx = np.where(mask)[0]
    T = len(mask)
    start, end = int(idx[0]), int(idx[-1]) + 1
    window_len = end - start
    context_len = max(1, int(round(window_len * context_multiplier)))

    left_start = max(0, start - context_len)
    left_end = start
    right_start = end
    right_end = min(T, end + context_len)
    return left_start, left_end, right_start, right_end


def local_mean_perturbation(x: np.ndarray, mask: np.ndarray, rng: np.random.Generator | None = None):
    ''' Replace the masked window with the mean of the immediately surrounding context
    '''
    x, mask = validate(x, mask)

    if mask.all():
        return global_mean_perturbation(x, mask)

    left_start, left_end, right_start, right_end = context_window_bounds(mask)
    context_values = np.concatenate([x[left_start:left_end], x[right_start:right_end]])

    if context_values.size == 0:
        return global_mean_perturbation(x, mask)

    x_out = x.copy()
    x_out[mask] = float(np.mean(context_values))
    return x_out


def linear_interpolation_perturbation(x: np.ndarray, mask: np.ndarray, rng: np.random.Generator | None = None):
    ''' Replace the masked window by linearly interpolating between its boundary values
    '''
    x, mask = validate(x, mask)

    if mask.all():
        return global_mean_perturbation(x, mask)

    idx = np.where(mask)[0]
    start, end = int(idx[0]), int(idx[-1]) + 1
    T = len(x)

    has_left = start > 0
    has_right = end < T

    x_out = x.copy()
    window_idx = np.arange(start, end)

    if has_left and has_right:
        left_val = x[start - 1]
        right_val = x[end]
        # Interpolate across the gap including both boundary points, then
        # discard the boundary points themselves (they are not masked).
        ramp = np.linspace(left_val, right_val, num=(end - start) + 2, dtype=np.float32)
        x_out[window_idx] = ramp[1:-1]
    elif has_left:
        x_out[window_idx] = x[start - 1]
    elif has_right:
        x_out[window_idx] = x[end]
    else:
        # No boundary on either side: window is the whole series.
        return global_mean_perturbation(x, mask)

    return x_out


def gaussian_noise_perturbation(x: np.ndarray, mask: np.ndarray, rng: np.random.Generator | None = None):
    ''' Replace the masked window with Gaussian noise matched to the global series mean/std
    '''
    x, mask = validate(x, mask)
    rng = np.random.default_rng() if rng is None else rng

    mean = float(np.mean(x))
    std = float(np.std(x)) + 1e-8

    x_out = x.copy()
    x_out[mask] = rng.normal(loc=mean, scale=std, size=int(mask.sum())).astype(np.float32)
    return x_out


def blur_perturbation(x: np.ndarray, mask: np.ndarray, rng=None, sigma: int = 5):
    ''' Replace the masked window with a heavily smoothed version of itself.
        Suppresses local temporal detail while preserving the coarse signal shape
    '''
    x, mask = validate(x, mask)
    x_out = x.copy()
    smoothed = uniform_filter1d(x, size=sigma)
    x_out[mask] = smoothed[mask]
    return x_out


def local_noise_matched_variance_perturbation(x: np.ndarray, mask: np.ndarray, rng: np.random.Generator | None = None):
    ''' Replace the masked window with noise matched to the *local* context mean/std.
    '''
    x, mask = validate(x, mask)
    rng = np.random.default_rng() if rng is None else rng

    if mask.all():
        return gaussian_noise_perturbation(x, mask, rng=rng)

    left_start, left_end, right_start, right_end = context_window_bounds(mask)
    context_values = np.concatenate([x[left_start:left_end], x[right_start:right_end]])

    if context_values.size < 2:
        return gaussian_noise_perturbation(x, mask, rng=rng)

    mean = float(np.mean(context_values))
    std = float(np.std(context_values)) + 1e-8
    x_out = x.copy()
    x_out[mask] = rng.normal(loc=mean, scale=std, size=int(mask.sum())).astype(np.float32)
    return x_out




# NOTE - Registry used by utils/explainability.py and the perturbation-robustness evaluator 
PERTURBATIONS = {
    "global_mean": global_mean_perturbation,
    "local_mean": local_mean_perturbation,
    "linear_interpolation": linear_interpolation_perturbation,
    "blur": blur_perturbation,
    "gaussian_noise": gaussian_noise_perturbation,
    "local_noise_matched_variance": local_noise_matched_variance_perturbation,
}

# NOTE - The three operators used for the core perturbation-operator robustness, noise based operators are available in the 
# registry as a stretch addition but are not part of the primary 3-operator comparison
CORE_PERTURBATIONS = ("global_mean", "local_mean", "linear_interpolation", "blur")

PERTURBATION_LABELS = {
    "global_mean": "Global mean",
    "local_mean": "Local mean",
    "linear_interpolation": "Linear interpolation",
    "gaussian_noise": "Gaussian noise (global)",
    "local_noise_matched_variance": "Local noise (matched variance)",
}


def get_perturbation(name: str):
    ''' Look up a perturbation operator by name
    '''
    if name not in PERTURBATIONS:
        raise ValueError(
            f"Unknown perturbation operator: {name!r}. "
            f"Available operators: {sorted(PERTURBATIONS)}"
        )
    return PERTURBATIONS[name]
