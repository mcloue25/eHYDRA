'''
FrequencyBurstGenerator: parametrised implementation of the TSHAP paper's DoubleFreqTest (Le Nguyen & Ifrim, tshap/synthetic.py
Used for the High-frequency / high-curvature & Short / moderately-rough clusters which share the same hypothetical model but differ in calibrated params 
'''
from dataclasses import dataclass, field, asdict
from typing import Optional, List
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
 
from .base import SyntheticGroundTruthGenerator
 
 
@dataclass
class FrequencyBurstConfig:
    cluster_key: str
    cluster_name: str
    tslength: int = 200
    splength: int = 40
    sp_idx: Optional[List[int]] = None  # None = auto two windows, evenly spaced
    min_f: float = 10.0
    max_f: float = 50.0
    f_base: float = 0.0  # background wave frequency (0 = silent)
    wave: str = "sine"  # "sine" | "square"
    noise_std: float = 0.0
    clf_threshold: Optional[float] = None  # None = min_f + max_f
    legacy_rng: bool = False  # True only for exact paper reproduction
 
    def __post_init__(self):
        if self.sp_idx is None:
            # NOTE - Split it evenly as gap, window, gap, window, gap
            gap = (self.tslength - 2 * self.splength) // 3
            self.sp_idx = [gap, self.tslength - gap - self.splength]

        if self.clf_threshold is None:
            # NOTE - balanced per-segment frequency is (min_f+max_f)/2 & threshold is the sum of min_f + max_f
            self.clf_threshold = self.min_f + self.max_f

        if self.sp_idx[1] + self.splength > self.tslength:
            raise ValueError(
                f"Second window [{self.sp_idx[1]}, "
                f"{self.sp_idx[1] + self.splength}) exceeds tslength={self.tslength}"
            )
 
 
# core signal/estimator functions copied from tshap/synthetic.py
def generate_feature(n_points, n_support, start_idx, wave, f_support, f_base):
    ''' Main function used to generate a feature 
    '''
    # background wave over the full series
    x_feature = np.sin(np.linspace(0, 2 * np.pi * f_base, n_points)).reshape(-1, 1)
    x_feature *= 0.5
 
    if wave == "sine":
        x_tmp = np.sin(np.linspace(0, 2 * np.pi * f_support, n_points)).reshape(-1, 1)
        # overlay the burst at its window only
        x_feature[start_idx:start_idx + n_support, 0] += x_tmp[:n_support, 0] 

    elif wave == "square":
        x_tmp = np.sign(
            np.sin(np.linspace(0, 2 * np.pi * f_support, n_points))
        ).reshape(-1, 1)  # squarewave via sign() of a sine
        x_feature[start_idx:start_idx + n_support, 0] += x_tmp[:n_support, 0]

    else:
        raise ValueError("wave must be one of 'sine', 'square'")
    return x_feature.reshape(n_points)
 


 
def window_frequency(window, ts_length):
    ''' window: (n, 1, w) array
    Returns:
        (n,) estimated frequency per sample, rescaled to the f_support
    '''
    window_lengths = np.zeros(window.shape[0])
    # True where the sign flips from - to +
    positive_zcr = np.diff(np.sign(window)) > 0
    x, _, z = np.where(positive_zcr)
 
    for xi in range(window.shape[0]):
        zcr = z[x == xi]
        if len(zcr) == 0:
            window_lengths[xi] = 1
        else:
            # span between first and last crossing
            window_lengths[xi] = np.max(zcr) - np.min(zcr) + 1

    # crossings per sample
    positive_zcr_counts = np.sum(positive_zcr > 0, axis=2).reshape(window.shape[0])
    # Fixing divide-by-zero below
    positive_zcr_counts[positive_zcr_counts == 0] += 1  
    # cycles rescaled to full-series-length units
    return ((positive_zcr_counts - 1) * ts_length) / window_lengths
 
 
# NOTE - sklearn-compatible wrappers for the hypothetical model
 
class GTRegressor(RegressorMixin, BaseEstimator):
    '''Wraps a plain predict function as an sklearn compatible regressor
    '''
    def __init__(self, predict_fnc):
        self.is_fitted = True
        self.predict_fnc = predict_fnc
 
    def fit(self, X=None, y=None):
        self.is_fitted_ = True
        return self
 
    def predict(self, X):
        return self.predict_fnc(X)


 
 
class GTClassifier(ClassifierMixin, BaseEstimator):
    ''' Thresholds a GTRegressor's output into a binary classifier with a logistic predict_proba
    '''
    def __init__(self, regressor, threshold):
        self.is_fitted = True
        self.threshold = threshold
        self.regressor = regressor
 
    def fit(self, X=None, y=None):
        self.is_fitted_ = True
        return self
 
    def predict(self, X):
        # class = which side of the threshold the regressor lands on
        return self.regressor.predict(X) > self.threshold
 
    def decision_function(self, X):
        # signed distance from the threshold
        return self.regressor.predict(X) - self.threshold
 
    def predict_proba(self, X):
        ''' 
        '''
        # scale for the logistic squash
        span = np.min((np.abs(self.threshold - 10), np.abs(self.threshold - 50)))
        span = span if span > 0 else 1.0
        y_pred = (self.regressor.predict(X) - self.threshold) * 4 / span
        y_prob_pos = 1 / (1 + np.exp(-y_pred))  # sigmoid
        return np.vstack((y_prob_pos, 1 - y_prob_pos)).T
 



 
class FrequencyBurstGenerator(SyntheticGroundTruthGenerator):
    ''' High-frequency & Short/rough clusters: two sine (or square) bursts at fixed locations
    '''
    def __init__(self, config: FrequencyBurstConfig):
        self.config = config
        self.cluster_key = config.cluster_key
        self.cluster_name = config.cluster_name

 
    def sample_frequencies(self, n_samples, seed):
        c = self.config
        if c.legacy_rng:
            np.random.seed(seed)
            # original paper's integer-frequency sampling, for bit-exact reproduction
            freqs = np.random.randint(low=int(c.min_f), high=int(c.max_f) + 1, size=(n_samples, 2)).astype(float)

        else:
            rng = np.random.RandomState(seed)
            # continuous frequencies for calibrated variants
            freqs = rng.uniform(c.min_f, c.max_f, size=(n_samples, 2))
        return freqs


 
    def noise_rng(self, seed):
        c = self.config
        if c.legacy_rng:
            return np.random  # continues the global state seeded above
        return np.random.RandomState(seed + 1)  # offset so noise != freq draws


 
    def generate_regression_data_and_attribs(self, n_samples, seed=42):
        ''' 
        '''
        c = self.config
        frequencies = self.sample_frequencies(n_samples, seed)
        rng = self.noise_rng(seed)
 
        y = np.sum(frequencies, axis=1)  # regression target: sum of the two window frequencies
        X = np.zeros((n_samples, 1, c.tslength))
        attribs = np.zeros(X.shape)
        for i in range(n_samples):
            X[i, 0, :] = generate_feature(c.tslength, c.splength, c.sp_idx[0], c.wave, frequencies[i, 0], c.f_base)
            X[i, 0, :] += generate_feature(c.tslength, c.splength, c.sp_idx[1], c.wave, frequencies[i, 1], c.f_base)
 
            if c.noise_std > 0:
                # Added Gaussian noise on top of both bursts
                X[i, 0, :] += rng.normal(0, c.noise_std, size=c.tslength)

            # ground truth = each burst's own frequency
            attribs[i, 0, c.sp_idx[0]:c.sp_idx[0] + c.splength] = frequencies[i, 0]
            attribs[i, 0, c.sp_idx[1]:c.sp_idx[1] + c.splength] = frequencies[i, 1]
        return X, y, attribs


 
    def generate_data_and_attribs(self, n_samples, seed=42):
        ''' 
        '''
        c = self.config
        X, y, attribs = self.generate_regression_data_and_attribs(n_samples, seed)
        attribs[..., c.sp_idx[0]:c.sp_idx[0] + c.splength] -= c.clf_threshold / 2  # centre attribution on the classification threshold
        attribs[..., c.sp_idx[1]:c.sp_idx[1] + c.splength] -= c.clf_threshold / 2
        return X, (y > c.clf_threshold), attribs  # label = whether the frequency sum clears the threshold


 
    def generate_background_sample(self):
        c = self.config
        half = c.clf_threshold / 2  # each burst set to exactly half the threshold -> sample sits on the decision boundary
        bg = np.zeros((1, 1, c.tslength))
        bg[0, 0, :] = generate_feature(c.tslength, c.splength, c.sp_idx[0], c.wave, half, c.f_base)
        bg[0, 0, :] += generate_feature(c.tslength, c.splength, c.sp_idx[1], c.wave, half, c.f_base)
        return bg


 
    def predict(self, X):
        '''Main predict function
        '''
        if X.ndim == 2:
            X = X.reshape(1, X.shape[0], X.shape[1])  # accept a single unbatched sample too
        c = self.config
        w1 = X[..., c.sp_idx[0]:c.sp_idx[0] + c.splength]
        w2 = X[..., c.sp_idx[1]:c.sp_idx[1] + c.splength]
        return window_frequency(w1, X.shape[-1]) + window_frequency(w2, X.shape[-1])  # sum of the two window frequency estimates


 
    def get_regression_model(self):
        return GTRegressor(predict_fnc=self.predict)


 
    def get_classifier_model(self):
        return GTClassifier(regressor=self.get_regression_model(), threshold=self.config.clf_threshold)
 
    def metadata(self):
        meta = super().metadata()
        meta["config"] = asdict(self.config)
        return meta
 
 
def estimate_noise_robustness(config: FrequencyBurstConfig, n_samples=200, seed=123):
    ''' Compares the model's recovered frequency sum on noiseless vs noisy versions of the same underlying series
    Returns:
        The correlation and MAE between the two
    '''
    clean_cfg = FrequencyBurstConfig(**{**asdict(config), "noise_std": 0.0})  # same config, noise forced off
    clean_gen = FrequencyBurstGenerator(clean_cfg)
    noisy_gen = FrequencyBurstGenerator(config)
 
    frequencies = clean_gen.sample_frequencies(n_samples, seed)  # same frequency draws for both, only noise differs
    c = config
    X_clean = np.zeros((n_samples, 1, c.tslength))
    X_noisy = np.zeros((n_samples, 1, c.tslength))
    noise_rng = np.random.RandomState(seed + 1)
    for i in range(n_samples):
        base = generate_feature(c.tslength, c.splength, c.sp_idx[0], c.wave, frequencies[i, 0], c.f_base)
        base += generate_feature(c.tslength, c.splength, c.sp_idx[1], c.wave, frequencies[i, 1], c.f_base)
        X_clean[i, 0, :] = base
        X_noisy[i, 0, :] = base + noise_rng.normal(0, c.noise_std, size=c.tslength)  # same base signal, noise added only here
 
    model = clean_gen.get_regression_model()
    pred_clean = model.predict(X_clean)
    pred_noisy = model.predict(X_noisy)
    true_sum = frequencies.sum(axis=1)
 
    return {
        "n_samples": n_samples,
        "noise_std": c.noise_std,
        "corr_clean_vs_true": float(np.corrcoef(pred_clean, true_sum)[0, 1]),
        "corr_noisy_vs_true": float(np.corrcoef(pred_noisy, true_sum)[0, 1]),  # big drop from clean = noise is corrupting the estimator
        "mae_clean_vs_true": float(np.mean(np.abs(pred_clean - true_sum))),
        "mae_noisy_vs_true": float(np.mean(np.abs(pred_noisy - true_sum))),
    }
 
