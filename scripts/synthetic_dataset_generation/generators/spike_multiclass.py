'''SpikeMultiClassGenerator v2 -- Spiky/multi-class cluster.
'''
from dataclasses import dataclass, asdict
from typing import Optional
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from .base import SyntheticGroundTruthGenerator


@dataclass
class SpikeMultiClassConfig:
    cluster_key: str
    cluster_name: str
    tslength: int = 500
    onset: int = 100  # fixed window start -- NOT class-relevant, unlike v1
    window_length: Optional[int] = None  # None -> auto, see __post_init__
    n_tiers: int = 5
    min_duration: int = 10
    max_duration: int = 80
    drift_rate: float = 0.03  # plateau level per burst timestep = drift_rate * duration
    burst_noise_std: float = 0.6  # elevated-variance noise during the burst itself
    bg_noise_std: float = 0.15  # background noise everywhere else (incl. the plateau)
    duration_readout_len: int = 15  # width of the plateau-only readout region at window end
    readout_margin: int = 10  # required gap between max_duration and window_length

    @property
    def tier_bounds(self):
        '''n_tiers contiguous, non-overlapping duration bands spanning [min_duration, max_duration]. 
            Class c's duration is sampled uniformly from tier_bounds[c]
        '''
        edges = np.linspace(self.min_duration, self.max_duration, self.n_tiers + 1)
        return [(float(edges[i]), float(edges[i + 1])) for i in range(self.n_tiers)]


    @property
    def tier_centers(self):
        return [(lo + hi) / 2 for lo, hi in self.tier_bounds]


    @property
    def reference_duration(self):
        return (self.min_duration + self.max_duration) / 2


    def __post_init__(self):
        if self.window_length is None:
            self.window_length = self.max_duration + self.duration_readout_len + self.readout_margin
        min_required = self.max_duration + self.duration_readout_len
        if self.window_length < min_required:
            raise ValueError(
                f"window_length={self.window_length} too small: the longest possible "
                f"burst (max_duration={self.max_duration}) must leave at least "
                f"duration_readout_len={self.duration_readout_len} clean plateau samples "
                f"before the window ends, so window_length must be >= {min_required}."
            )
        
        if self.onset + self.window_length > self.tslength:
            raise ValueError(
                f"window [{self.onset}, {self.onset + self.window_length}) "
                f"exceeds tslength={self.tslength}"
            )
        
        if self.onset < self.duration_readout_len:
            raise ValueError(
                f"onset={self.onset} too close to t=0: the pre-window baseline region "
                f"used by the estimator needs at least duration_readout_len="
                f"{self.duration_readout_len} samples before the window starts."
            )


def make_series(rng, config, duration):
    ''' 
    '''
    c = config
    x = rng.normal(0.0, c.bg_noise_std, size=c.tslength)  # quiet background everywhere

    burst_end = c.onset + duration
    window_end = c.onset + c.window_length
    final_level = c.drift_rate * duration

    if duration > 0:
        ramp = np.linspace(c.drift_rate, final_level, duration)  # linear ramp up to final_level
        x[c.onset:burst_end] = ramp + rng.normal(0.0, c.burst_noise_std, size=duration)

    plateau_len = window_end - burst_end
    if plateau_len > 0:
        x[burst_end:window_end] = final_level + rng.normal(0.0, c.bg_noise_std, size=plateau_len)  # settle at final_level

    return x


def estimate_duration(X, config):
    ''' 
    Args:
        X: (n, 1, T) or (n, T). Closed-form duration estimate: plateau readout level minus outside-window baseline, divided by the known drift_rate. 
    '''
    if X.ndim == 3:
        X = X[:, 0, :]
    c = config
    window_end = c.onset + c.window_length

    if window_end < c.tslength:
        outside = np.concatenate([X[:, :c.onset], X[:, window_end:]], axis=1)  # everything outside the window
    else:
        outside = X[:, :c.onset]
    baseline = outside.mean(axis=-1)

    readout = X[:, window_end - c.duration_readout_len:window_end]  # plateau-only region at the window's end
    level = readout.mean(axis=-1) - baseline

    duration_est = level / c.drift_rate  # invert the known drift rate to recover duration
    return np.clip(duration_est, c.min_duration, c.max_duration)


class GTMultiClassifier(ClassifierMixin, BaseEstimator):
    ''' Tier distance multiclass wrapper: decision_function returns per-tier closeness scores
        predict = argmax, predict_proba = softmax
    '''
    def __init__(self, score_fnc, n_classes, temperature=0.15):
        self.is_fitted = True
        self.score_fnc = score_fnc
        self.n_classes = n_classes
        self.temperature = temperature


    def fit(self, X=None, y=None):
        self.is_fitted_ = True
        return self


    def decision_function(self, X):
        return self.score_fnc(X)


    def predict(self, X):
        return np.argmax(self.decision_function(X), axis=-1)


    def predict_proba(self, X):
        s = self.decision_function(X) * self.temperature
        s = s - s.max(axis=-1, keepdims=True)  # numerically stable softmax
        e = np.exp(s)
        return e / e.sum(axis=-1, keepdims=True)




class SpikeMultiClassGenerator(SyntheticGroundTruthGenerator):
    ''' Spiky/multi-class cluster data generator which generates a single fixed-location burst to plateau regime shift whose DURATION encodes the class
    '''
    def __init__(self, config: SpikeMultiClassConfig):
        self.config = config
        self.cluster_key = config.cluster_key
        self.cluster_name = config.cluster_name

    def generate_data_and_attribs(self, n_samples, seed=42):
        c = self.config
        rng = np.random.RandomState(seed)
        bounds = c.tier_bounds

        y = rng.randint(0, c.n_tiers, size=n_samples)  # class = tier index
        # NOTE - Duration is drawn uniformly from the sampled class's own tier band
        durations = np.array([rng.uniform(bounds[cls][0], bounds[cls][1]) for cls in y])

        X = np.zeros((n_samples, 1, c.tslength))
        attribs = np.zeros_like(X)
        window_end = c.onset + c.window_length

        for i in range(n_samples):
            duration_i = int(round(durations[i]))
            X[i, 0, :] = make_series(rng, c, duration_i)
            attribs[i, 0, c.onset:window_end] = durations[i] - c.reference_duration  # ground truth: signed offset from reference

        return X, y, attribs



    def generate_background_sample(self):
        ''' Uses the reference (range-midpoint) duration. Unlike the binary generators, this isn't a decision-boundary sample 
            with n_tiers >= 3 there's no single boundary, so the midpoint lands near the middle tier's center instead (predict_proba is 
            confidently mid-tier, not near-uniform: [0.04, 0.93, 0.03] for n_tiers=3). Closer to TSHAP's "centroid" background
            (Sec 3.1, Eq. 5) than its "threshold" option
        '''
        c = self.config
        rng = np.random.RandomState(0)
        bg = np.zeros((1, 1, c.tslength))
        bg[0, 0, :] = make_series(rng, c, int(round(c.reference_duration)))
        return bg
    

    def get_classifier_model(self):
        c = self.config
        tier_centers = np.array(c.tier_centers)

        def score_fnc(X):
            duration_est = estimate_duration(X, c)  # (n,)
            dist = np.abs(duration_est[:, None] - tier_centers[None, :])  # (n, n_tiers)
            return -dist  # closer to a tier's center -> higher score

        return GTMultiClassifier(score_fnc=score_fnc, n_classes=c.n_tiers)


    def metadata(self):
        meta = super().metadata()
        meta["config"] = asdict(self.config)
        meta["config"]["tier_bounds"] = self.config.tier_bounds
        meta["config"]["tier_centers"] = self.config.tier_centers
        return meta