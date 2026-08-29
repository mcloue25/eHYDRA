'''
LevelShiftGenerator for the Smooth / low-complexity cluster.
'''
from dataclasses import dataclass, asdict
from typing import Optional, List
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin

from .base import SyntheticGroundTruthGenerator


@dataclass
class LevelShiftConfig:
    cluster_key: str
    cluster_name: str
    tslength: int = 500
    splength: int = 100
    sp_idx: Optional[List[int]] = None
    min_level: float = 10.0
    max_level: float = 50.0
    drift_freq: float = 4.0  # cycles over the full series (low -> slow trend)
    drift_amplitude: float = 0.5
    baseline_len: Optional[int] = None  # None -> = splength
    clf_threshold: Optional[float] = None

    def __post_init__(self):
        if self.sp_idx is None:
            # NOTE - Split it evenly as gap, window, gap, window, gap
            gap = (self.tslength - 2 * self.splength) // 3 
            self.sp_idx = [gap, self.tslength - gap - self.splength]

        if self.clf_threshold is None:
            self.clf_threshold = self.min_level + self.max_level

        if self.baseline_len is None:
            self.baseline_len = self.splength

        if self.sp_idx[1] + self.splength > self.tslength:
            raise ValueError(f"Second window [{self.sp_idx[1]}, " f"{self.sp_idx[1] + self.splength}) exceeds tslength={self.tslength}")




def generate_drift(tslength, drift_freq, drift_amplitude, phase):
    t = np.arange(tslength)
    return drift_amplitude * np.sin(2 * np.pi * drift_freq * t / tslength + phase)  # slow background sinusoid


def window_level(x, start, length, baseline_len):
    '''x: (n, 1, T). Local level = window mean minus flanking-region mean,
    which cancels a slowly-varying background trend. Flank is taken
    immediately before the window, or immediately after if the window
    starts too close to t=0.'''
    T = x.shape[-1]
    window = x[..., start:start + length]
    if start - baseline_len >= 0:
        flank = x[..., start - baseline_len:start]  # flank before the window
    else:
        flank_end = min(start + length + baseline_len, T)
        flank = x[..., start + length:flank_end]  # window too close to t=0, flank after instead
    # window/flank are (n, 1, w) -- mean over the last axis leaves a stray channel dim (n, 1)
    # that np.corrcoef would misread as "1 observation of n variables", so squeeze to (n,)
    level = window.mean(axis=-1) - flank.mean(axis=-1)
    return level.reshape(x.shape[0])


class GTRegressor(RegressorMixin, BaseEstimator):
    ''' Wraps a plain predict function as an sklearn compatible regressor
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
    def __init__(self, regressor, threshold, scale):
        self.is_fitted = True
        self.regressor = regressor
        self.threshold = threshold
        self.scale = scale


    def fit(self, X=None, y=None):
        self.is_fitted_ = True
        return self


    def predict(self, X):
        return self.regressor.predict(X) > self.threshold


    def decision_function(self, X):
        return self.regressor.predict(X) - self.threshold


    def predict_proba(self, X):
        y_pred = (self.regressor.predict(X) - self.threshold) * 4 / self.scale
        y_prob_pos = 1 / (1 + np.exp(-y_pred))  # sigmoid
        return np.vstack((y_prob_pos, 1 - y_prob_pos)).T





class LevelShiftGenerator(SyntheticGroundTruthGenerator):
    ''' Smooth/low-complexity cluster uses two level-shift segments over a slow sinusoidal drift background
    '''
    def __init__(self, config: LevelShiftConfig):
        self.config = config
        self.cluster_key = config.cluster_key
        self.cluster_name = config.cluster_name


    def make_series(self, levels, phase):
        c = self.config
        x = generate_drift(c.tslength, c.drift_freq, c.drift_amplitude, phase)
        x[c.sp_idx[0]:c.sp_idx[0] + c.splength] += levels[0]  # inject level shift into each window
        x[c.sp_idx[1]:c.sp_idx[1] + c.splength] += levels[1]
        return x


    def generate_regression_data_and_attribs(self, n_samples, seed=42):
        '''
        '''
        c = self.config
        rng = np.random.RandomState(seed)
        levels = rng.uniform(c.min_level, c.max_level, size=(n_samples, 2))
        # random drift phase per sample so it isn't a fixed background
        phases = rng.uniform(0, 2 * np.pi, size=n_samples)  

        # NOTE - regression target = sum of the two window levels
        y = levels.sum(axis=1)
        X = np.zeros((n_samples, 1, c.tslength))
        attribs = np.zeros(X.shape)
        for i in range(n_samples):
            X[i, 0, :] = self.make_series(levels[i], phases[i])
            # NOTE - Ground truth = each window's own level
            attribs[i, 0, c.sp_idx[0]:c.sp_idx[0] + c.splength] = levels[i, 0]
            attribs[i, 0, c.sp_idx[1]:c.sp_idx[1] + c.splength] = levels[i, 1]
        return X, y, attribs



    def generate_data_and_attribs(self, n_samples, seed=42):
        ''' 
        '''
        c = self.config
        X, y, attribs = self.generate_regression_data_and_attribs(n_samples, seed)
        attribs[..., c.sp_idx[0]:c.sp_idx[0] + c.splength] -= c.clf_threshold / 2  # centre attribution on the classification threshold
        attribs[..., c.sp_idx[1]:c.sp_idx[1] + c.splength] -= c.clf_threshold / 2
        return X, (y > c.clf_threshold), attribs  # label = whether the level sum clears the threshold


    def generate_background_sample(self):
        c = self.config
        half = c.clf_threshold / 2  # each window set to exactly half the threshold -> sample sits on the decision boundary
        bg = np.zeros((1, 1, c.tslength))
        bg[0, 0, :] = self.make_series([half, half], phase=0.0)
        return bg


    def predict(self, X):
        if X.ndim == 2:
            X = X.reshape(1, X.shape[0], X.shape[1])  # accept a single unbatched sample too
        c = self.config
        l1 = window_level(X, c.sp_idx[0], c.splength, c.baseline_len)
        l2 = window_level(X, c.sp_idx[1], c.splength, c.baseline_len)
        return l1 + l2  # sum of the two window level estimates


    def get_regression_model(self):
        return GTRegressor(predict_fnc=self.predict)


    def get_classifier_model(self):
        c = self.config
        scale = np.min((np.abs(c.clf_threshold - c.min_level * 2),
                         np.abs(c.clf_threshold - c.max_level * 2)))  # distance from threshold to either extreme, for the sigmoid scale
        scale = scale if scale > 0 else 1.0
        return GTClassifier(regressor=self.get_regression_model(), threshold=c.clf_threshold, scale=scale)


    def metadata(self):
        meta = super().metadata()
        meta["config"] = asdict(self.config)
        return meta




def estimate_baseline_bias(config: LevelShiftConfig, n_samples=200, seed=321):
    '''Checks how much the drift background alone biases the window_level estimator away from 0. 
    Returns:
        mean/std of the estimator's output on pure-drift series
    '''
    gen = LevelShiftGenerator(config)
    rng = np.random.RandomState(seed)
    phases = rng.uniform(0, 2 * np.pi, size=n_samples)
    c = config
    vals = []
    for phase in phases:
        x = generate_drift(c.tslength, c.drift_freq, c.drift_amplitude, phase)  # drift only, no injected level
        x = x.reshape(1, 1, c.tslength)
        l1 = window_level(x, c.sp_idx[0], c.splength, c.baseline_len)[0]
        l2 = window_level(x, c.sp_idx[1], c.splength, c.baseline_len)[0]
        vals.append(l1 + l2)  # should be ~0 if the flanking-baseline trick is working
    vals = np.array(vals)
    return {
        "n_samples": n_samples,
        "bias_mean": float(vals.mean()),
        "bias_std": float(vals.std()),
        "typical_level_scale": (config.min_level + config.max_level) / 2,
    }