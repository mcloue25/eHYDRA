import numpy as np
import os
from aeon.datasets import load_classification
from sklearn.preprocessing import LabelEncoder
from dataclasses import dataclass


def get_data_dir() -> str:
    current_dir: str = os.path.dirname(__file__)
    data_dir: str = os.path.join(current_dir, "..", "..", "data")
    return data_dir


def get_cmj_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data_dir: str = get_data_dir()
    cmj_dir: str = os.path.join(data_dir, "cmj")
    x_train: np.ndarray = np.load(os.path.join(cmj_dir, "X_train_magnitude.npy"))
    y_train: np.ndarray = np.load(os.path.join(cmj_dir, "CMJ_y_train.npy"))
    x_test: np.ndarray = np.load(os.path.join(cmj_dir, "X_test_magnitude.npy"))
    y_test: np.ndarray = np.load(os.path.join(cmj_dir, "CMJ_y_test.npy"))
    return x_train, y_train, x_test, y_test



@dataclass
class RunResult:
    dataset: str
    model: str
    seed: int
    n_train: int = 0
    n_test: int = 0
    series_length: int = 0
    n_channels: int = 1
    n_classes: int = 0
    total_time: float = 0.0
    peak_memory_mb: float = 0.0
    accuracy: float = 0.0
    f1_macro: float = 0.0
    f1_weighted: float = 0.0
    status: str = "ok"
    error_msg: str = ""


def _coerce_univariate(x: np.ndarray, split_name: str) -> np.ndarray:
    '''
    Convert aeon output to (N, T) for univariate datasets only.
    Raises on multivariate input so explainability code does not silently break.
    '''
    x = np.asarray(x)

    if x.ndim == 2:
        return x

    if x.ndim == 3:
        n_samples, n_channels, series_length = x.shape
        if n_channels != 1:
            raise ValueError(
                f"{split_name} is multivariate with shape {x.shape}. "
                "Current explainability pipeline only supports univariate datasets."
            )
        return x[:, 0, :]

    raise ValueError(
        f"Unexpected {split_name} shape {x.shape}. Expected (N, T) or (N, 1, T)."
    )


def load_dataset(name: str):
    '''Load UCR dataset using canonical train/test splits, restricted to univariate.'''
    x_train, y_train = load_classification(name, split="train")
    x_test, y_test = load_classification(name, split="test")

    x_train = _coerce_univariate(x_train, "x_train")
    x_test = _coerce_univariate(x_test, "x_test")

    le = LabelEncoder()
    le.fit(np.concatenate([y_train, y_test]))
    y_train = le.transform(y_train)
    y_test = le.transform(y_test)

    return x_train, y_train, x_test, y_test, le


def dataset_meta(x_train, x_test, y_train) -> dict:
    return {
        "n_train": len(x_train),
        "n_test": len(x_test),
        "series_length": x_train.shape[-1],
        "n_channels": 1,
        "n_classes": len(np.unique(y_train)),
    }

def adapt_hydra_input_shape(X):
    '''dataset_io artifacts are (N, 1, T) -- HydraModelExplainable expects
    (N, T). This is the only shape adaptation needed; labels need no
    conversion (bool casts cleanly to 0/1, spiky's int slot indices are
    already RidgeClassifierCV-multiclass-compatible).'''
    if X.ndim == 3 and X.shape[1] == 1:
        return X[:, 0, :]
    raise ValueError(f"expected (N, 1, T), got {X.shape}")
