import torch
import numpy as np

from torch.profiler import profile
from torch._C._profiler import ProfilerActivity
from sklearn.linear_model import RidgeClassifierCV
from torch.autograd.profiler import record_function

# from experiments.hydra import Hydra, SparseScaler
# from experiments.hydra.code.optimised_hydra import UpdatedHydra
from classes.models.hydra_original import Hydra, SparseScaler
from classes.models.optimised_hydra import UpdatedHydra


def _to_numpy(t):
    ''' Move torch tensors back to NumPy
    '''
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return np.asarray(t)


def _get_safe_device():
    ''' Get available device 
    '''
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class HydraModelExplainable:
    def __init__(self, input_dim: int, device: torch.device | None = None, k: int = 8, g: int = 64, seed: int = 42, use_latency_optimisation: bool = True, print_profile: bool = False):
        self.device = device or _get_safe_device()
        self.print_profile = print_profile
        self.use_latency_optimisation = use_latency_optimisation
        self.classifier = None
        self.is_fitted = False

        print(f"[HydraModel] Using device: {self.device}")

        # Use the optimised version for MPS or when explicitly requested
        if use_latency_optimisation or self.device.type == "mps":
            print("[HydraModel] Using latency optimisation")
            self.transform = UpdatedHydra(input_dim, k=k, g=g, seed=seed).to(self.device)
        else:
            self.transform = Hydra(input_dim, k=k, g=g, seed=seed).to(self.device)
            
        self.transform.eval()
        self.scaler = SparseScaler()

        # Set seeds for reproducibility
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


    def _to_tensor(self, x: np.ndarray):
        ''' Convert (N, T) NumPy input to HYDRA tensor format 
        '''
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError(f"Expected shape (N, T), got {x.shape}")
        return torch.from_numpy(x).unsqueeze(-2).to(self.device)


    def _transform_features(self, x_t: torch.Tensor):
        ''' Use batched transform when supported
        '''
        if self.use_latency_optimisation or self.device.type == "mps":
            return self.transform.batch(x_t)
        return self.transform(x_t)


    def _transform_and_scale(self, x: np.ndarray):
        ''' Transform raw series into HYDRA features and scale them
        '''
        x_t = self._to_tensor(x)
        with torch.inference_mode():
            z = self._transform_features(x_t)
        z = self.scaler.transform(z)
        return _to_numpy(z)


    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        ''' Splitting the original call function into seperate functions so that I can add and explain()
            Fit HYDRA features, scaler, and ridge classifier
        '''
        x_train = np.asarray(x_train, dtype=np.float32)
        y_train = np.asarray(y_train).astype(np.int32)
        x_train_t = self._to_tensor(x_train)

        try:
            if self.print_profile:
                z_train = self._print_profile(x_train_t)
            else:
                with torch.inference_mode():
                    z_train = self._transform_features(x_train_t)
        except Exception as e:
            print(f"[HydraModel] Transformation error during fit: {e}")
            raise

        try:
            z_train = self.scaler.fit_transform(z_train)
        except Exception as e:
            print(f"[HydraModel] Scaling error during fit: {e}")
            raise

        z_train_np = _to_numpy(z_train)
        # NOTE - Fitting classifier
        try:
            self.classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
            self.classifier.fit(z_train_np, y_train)
        except Exception as e:
            print(f"[HydraModel] Classification error during fit: {e}")
            raise

        self.is_fitted = True
        return self



    def predict(self, x_test: np.ndarray):
        ''' Predict labels from raw time series
        '''
        if not self.is_fitted:
            raise RuntimeError("HydraModel must be fitted before calling predict().")
        z_test_np = self._transform_and_scale(x_test)
        return self.classifier.predict(z_test_np)
    


    def decision_function(self, x: np.ndarray):
        ''' Return classifier decision scores
        '''
        z_np = self._transform_and_scale(x)
        return self.classifier.decision_function(z_np)



    def __call__(self, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray | None = None):
        ''' Wrapper for fit and predict
        '''
        self.fit(x_train, y_train)
        return self.predict(x_test)



    def explain(self, x_single: np.ndarray, class_index: int | None = None, verbose: bool = False):
        ''' Main funbction for generating explanations for a time series
            Generate a saliency map for one time series
        '''
        x_single = np.asarray(x_single, dtype=np.float32)

        if verbose:
            print("\n[EXPLAIN] Starting explanation...")
            print(f"[EXPLAIN] Input shape: {x_single.shape}")

        x_batched = x_single[None, :]
        x_single_t = self._to_tensor(x_batched)

        # Default to explaining the predicted class
        if class_index is None:
            pred = self.predict(x_batched)[0]

            if len(self.classifier.classes_) > 2:
                class_index = int(np.where(self.classifier.classes_ == pred)[0][0])
            else:
                # Binary RidgeClassifierCV uses one score direction
                class_index = 0

        with torch.inference_mode():
            saliency = self.transform.get_saliency_map(x_single_t, self.classifier, self.scaler, class_index=class_index)

        saliency = np.asarray(saliency, dtype=np.float32)

        # Remove batch dimension
        if saliency.ndim == 2 and saliency.shape[0] == 1:
            saliency = saliency[0]

        if verbose:
            print(f"[EXPLAIN] Saliency shape: {saliency.shape}")
            print(f"[EXPLAIN] Min: {saliency.min()}, Max: {saliency.max()}")
            print("[EXPLAIN] Done.\n")

        return saliency



    def _print_profile(self, x_t: torch.Tensor):
        ''' Profile feature extraction
        '''
        activities = [ProfilerActivity.CPU]

        if self.device.type == "cuda":
            activities.append(ProfilerActivity.CUDA)

        with profile(activities=activities, record_shapes=True) as prof:
            with record_function("model_inference"):
                with torch.inference_mode():
                    z = self._transform_features(x_t)
        try:
            print(prof.key_averages().table(sort_by="self_cpu_time_total"))
        except Exception:
            print(prof.key_averages())

        return z
    

# if __name__ == "__main__":
#     np.random.seed(42)
#     n_train, n_test = 50, 20
#     series_length = 100
#     X_train = np.random.randn(n_train, series_length).astype(np.float32)
#     y_train = np.random.randint(0, 2, size=n_train)
#     X_test = np.random.randn(n_test, series_length).astype(np.float32)
#     y_test = np.random.randint(0, 2, size=n_test)
#     # Model
#     model = HydraModelExplainable(input_dim=series_length, print_profile=False)
#     model.fit(X_train, y_train)
#     print("[Test HYDRA] Model fitted.")

#     # Predicting
#     preds = model.predict(X_test)
#     acc = (preds == y_test).mean()
#     print(f"[Test HYDRA] Test Accuracy: {acc:.3f}")

#     # NOTE - Testing explanation
#     sample_idx = 0
#     saliency = model.explain(X_test[sample_idx], verbose=True)
#     print(f"[Test HYDRA] Saliency shape: {saliency.shape}")