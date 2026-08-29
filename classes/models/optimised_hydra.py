# Angus Dempster, Daniel F Schmidt, Geoffrey I Webb
import numpy as np
import torch

# from experiments.hydra import Hydra
from classes.models.hydra_original import Hydra
import torch.nn.functional as F

# HYDRA: Competing Convolutional Kernels for Fast and Accurate Time Series Classification
# https://arxiv.org/abs/2203.13652

# Implementation of Hydra which utilises Macs MPS GPU
# TODO: Should we try test it on CUDA?
class UpdatedHydra(Hydra):
    def __init__(self, input_length, k=8, g=64, seed=42):
        # 1. Run the parent initialization (this sets up self.W automatically!)
        super().__init__(input_length=input_length, k=k, g=g, seed=seed)

    def batch(self, X, batch_size: int| None = None, target_elements=3_600):
        if batch_size is None:
            # Our custom Apple Silicon SLC-aware batching
            seq_length = X.shape[-1]

            # We learned that batch sizes that are too big cause cache spilling on MPS
            batch_size = max(1, min(256, target_elements // seq_length))
        num_examples = X.shape[0]
        if num_examples <= batch_size:
            return self(X)
        else:
            Z = []
            for batch in X.split(batch_size, dim=0):
                Z.append(self(batch))
            return torch.cat(Z)


    def forward(self, X):
        # Overridden forward pass using our python lists to prevent .item() stalls
        num_examples = X.shape[0]

        if self.divisor > 1:
            diff_X = torch.diff(X)

        Z = []

        for dilation_index in range(self.num_dilations):
            # Using our lists instead of tensor.item()!
            d = self.dilations[dilation_index].item()
            p = self.paddings[dilation_index].item()

            for diff_index in range(self.divisor):
                _Z = F.conv1d(X if diff_index == 0 else diff_X,
                              self.W[dilation_index, diff_index],
                              dilation=d, padding=p).view(num_examples, self.h, self.k, -1)

                max_values, max_indices = _Z.max(2)
                count_max = torch.zeros(num_examples, self.h, self.k, device=X.device)
                count_max.scatter_add_(-1, max_indices, max_values)

                min_values, min_indices = _Z.min(2)
                count_min = torch.zeros(num_examples, self.h, self.k, device=X.device)
                count_min.scatter_add_(-1, min_indices, torch.ones_like(min_values))

                Z.append(count_max)
                Z.append(count_min)

        Z = torch.cat(Z, 1).view(num_examples, -1)
        return Z
    


    def get_saliency_map(
        self,
        X_single,
        ridge_classifier,
        scaler,
        class_index=0,
        saliency_variant="combined",
    ):
        """
        Vectorised saliency projection.

        Variants:
        - "combined": proposed max + min rule
        - "max_only": only max-competition features
        - "min_only": only min-competition/count features
        - "min_activation_scaled": max rule plus min features scaled by min activation
        """
        self.eval()

        if saliency_variant not in {
            "combined",
            "max_only",
            "min_only",
            "min_activation_scaled",
        }:
            raise ValueError(f"Unknown saliency_variant: {saliency_variant}")

        # Ridge weights for the target class.
        # See the matching comment in get_saliency_map() above: coef_ is 1D
        # (n_features,) for binary RidgeClassifierCV and must be used as-is,
        # not indexed with [0] (which would collapse it to a single scalar).
        if ridge_classifier.coef_.ndim == 1:
            w_ridge = ridge_classifier.coef_
        elif ridge_classifier.coef_.shape[0] == 1:
            w_ridge = ridge_classifier.coef_[0]
        else:
            w_ridge = ridge_classifier.coef_[class_index]

        # Map classifier weights back through the scaler
        sigma = scaler.sigma.detach().cpu().numpy()
        w_adjusted = w_ridge / sigma

        seq_length = X_single.shape[-1]
        saliency_map = np.zeros(seq_length, dtype=np.float32)

        if self.divisor > 1:
            diff_X = torch.diff(X_single)

        feature_offset = 0
        g_idx = np.arange(self.h)[:, None]

        for dilation_index in range(self.num_dilations):
            d = int(self.dilations[dilation_index].item())
            p = int(self.paddings[dilation_index].item())

            for diff_index in range(self.divisor):
                with torch.inference_mode():
                    _Z = F.conv1d(
                        X_single if diff_index == 0 else diff_X,
                        self.W[dilation_index, diff_index],
                        dilation=d,
                        padding=p,
                    ).view(1, self.h, self.k, -1)

                    max_values, max_indices = _Z.max(2)
                    min_values, min_indices = _Z.min(2)

                max_vals_np = max_values.squeeze(0).detach().cpu().numpy()
                max_idx_np = max_indices.squeeze(0).detach().cpu().numpy()

                min_vals_np = min_values.squeeze(0).detach().cpu().numpy()
                min_idx_np = min_indices.squeeze(0).detach().cpu().numpy()

                T_out = max_vals_np.shape[1]
                block_size = self.h * self.k

                # Classifier weights for max-count features
                weights_max = w_adjusted[
                    feature_offset: feature_offset + block_size
                ].reshape(self.h, self.k)
                feature_offset += block_size

                # Classifier weights for min-count features
                weights_min = w_adjusted[
                    feature_offset: feature_offset + block_size
                ].reshape(self.h, self.k)
                feature_offset += block_size

                # ---------------------------------------------------------
                # Max-feature contribution:
                # HYDRA's max branch stores positive activation magnitudes,
                # so attribution uses activation × adjusted classifier weight.
                # ---------------------------------------------------------
                max_weights = weights_max[g_idx, max_idx_np]
                max_contrib = max_vals_np * max_weights
                max_contrib = np.where(max_vals_np > 0, max_contrib, 0.0)

                # ---------------------------------------------------------
                # Min-feature contribution:
                # HYDRA's min branch stores occurrence counts, so the proposed
                # rule uses the adjusted classifier weight alone.
                # ---------------------------------------------------------
                min_weights = weights_min[g_idx, min_idx_np]
                min_contrib = min_weights

                # ---------------------------------------------------------
                # Ablation variants
                # ---------------------------------------------------------
                if saliency_variant == "combined":
                    pass

                elif saliency_variant == "max_only":
                    min_contrib = np.zeros_like(min_contrib)

                elif saliency_variant == "min_only":
                    max_contrib = np.zeros_like(max_contrib)

                elif saliency_variant == "min_activation_scaled":
                    # Tests the alternative design: treating min features as if
                    # they were activation-valued rather than count-valued.
                    min_contrib = min_vals_np * min_weights

                t_idx = np.arange(T_out, dtype=np.int64)

                # Accumulate onto the original time axis.
                # Only loop over the 9 receptive-field offsets; group/time work is vectorised.
                for kernel_step in range(9):
                    orig_t = t_idx - p + kernel_step * d
                    valid = (orig_t >= 0) & (orig_t < seq_length)

                    if not np.any(valid):
                        continue

                    target_idx = orig_t[valid]

                    np.add.at(
                        saliency_map,
                        np.broadcast_to(
                            target_idx,
                            (self.h, target_idx.shape[0]),
                        ).ravel(),
                        max_contrib[:, valid].ravel(),
                    )

                    np.add.at(
                        saliency_map,
                        np.broadcast_to(
                            target_idx,
                            (self.h, target_idx.shape[0]),
                        ).ravel(),
                        min_contrib[:, valid].ravel(),
                    )

        return saliency_map.astype(np.float32)
    

    

    # def get_saliency_map(self, X_single, ridge_classifier, scaler, class_index=0):
    #     """
    #     Generates a saliency map for a single time series.
    #     X_single: Tensor of shape (1, 1, sequence_length)
    #     ridge_classifier: Trained sklearn RidgeClassifierCV
    #     scaler: Fitted SparseScaler
    #     class_index: Which class's weights to use for the explanation.
    #     """
    #     self.eval()

    #     # 1. Get the Ridge weights for the target class
    #     #
    #     # IMPORTANT: scikit-learn's RidgeClassifierCV returns coef_ with shape
    #     # (n_features,) -- a 1D array -- for binary classification, and shape
    #     # (n_classes, n_features) for multiclass. A 1D array indexed with [0]
    #     # (i.e. coef_[0]) returns a single SCALAR (the weight of feature 0),
    #     # not the weight vector, which silently collapses the saliency
    #     # projection to (scalar / sigma) for every feature -- still runs
    #     # without error, but the resulting "saliency" no longer reflects the
    #     # classifier's actual learned per-feature weights. This previously
    #     # mis-handled the binary case; some sklearn versions/configurations
    #     # may still return a (1, n_features) shape for binary, which is
    #     # handled separately below.
    #     if ridge_classifier.coef_.ndim == 1:
    #         w_ridge = ridge_classifier.coef_
    #     elif ridge_classifier.coef_.shape[0] == 1:
    #         w_ridge = ridge_classifier.coef_[0]
    #     else:
    #         w_ridge = ridge_classifier.coef_[class_index]

    #     # 2. Adjust weights by the scaler's standard deviation
    #     # (Avoid division by zero using the scaler's epsilon)
    #     sigma = scaler.sigma.cpu().numpy()
    #     w_adjusted = w_ridge / sigma

    #     # 3. Initialize the Saliency Map
    #     seq_length = X_single.shape[-1]
    #     saliency_map = np.zeros(seq_length, dtype=np.float32)

    #     if self.divisor > 1:
    #         diff_X = torch.diff(X_single)

    #     feature_offset = 0  # Keeps track of which feature column we are looking at


    #     for dilation_index in range(self.num_dilations):
    #         d = self.dilations[dilation_index].item()
    #         p = self.paddings[dilation_index].item()

    #         for diff_index in range(self.divisor):
    #             _Z = F.conv1d(X_single if diff_index == 0 else diff_X,
    #                           self.W[dilation_index, diff_index],
    #                           dilation=d, padding=p).view(1, self.h, self.k, -1)

    #             # Get where the kernels activated
    #             max_values, max_indices = _Z.max(2)
    #             min_values, min_indices = _Z.min(2)

    #             # Both are shape: (1, h, seq_len)
    #             max_vals_np = max_values.squeeze(0).cpu().numpy()
    #             max_idx_np = max_indices.squeeze(0).cpu().numpy()

    #             min_vals_np = min_values.squeeze(0).cpu().numpy()
    #             min_idx_np = min_indices.squeeze(0).cpu().numpy()

    #             # Number of features per block (max or min)
    #             block_size = self.h * self.k

    #             # --- PROCESS MAX COUNTS ---
    #             # weights_max shape: (h, k)
    #             weights_max = w_adjusted[feature_offset: feature_offset + block_size].reshape(self.h, self.k)
    #             feature_offset += block_size

    #             # --- PROCESS MIN COUNTS ---
    #             weights_min = w_adjusted[feature_offset: feature_offset + block_size].reshape(self.h, self.k)
    #             feature_offset += block_size

    #             # Smear the weights back onto the original time series
    #             # For each group 'g_idx' and each timestep 't'
    #             for g_idx in range(self.h):
    #                 for t in range(_Z.shape[-1]):
    #                     # Which kernel won max?
    #                     k_max = max_idx_np[g_idx, t]
    #                     val_max = max_vals_np[g_idx, t]

    #                     k_min = min_idx_np[g_idx, t]
    #                     val_min = min_vals_np[g_idx, t]
    #                     if val_max > 0:  # clamp(0) logic from scaler
    #                         weight_m = weights_max[g_idx, k_max]
    #                         # Smear across the 9 points of the receptive field
    #                         for kernel_step in range(9):
    #                             orig_t = t - p + (kernel_step * d)
    #                             if 0 <= orig_t < seq_length:
    #                                 saliency_map[orig_t] += val_max * weight_m

    #                     # Which kernel won min?
    #                     k_min = min_idx_np[g_idx, t]
    #                     # min is just counted (value = 1), not summed!
    #                     weight_min = weights_min[g_idx, k_min]
    #                     for kernel_step in range(9):
    #                         orig_t = t - p + (kernel_step * d)
    #                         if 0 <= orig_t < seq_length:
    #                             saliency_map[orig_t] += 1.0 * weight_min

    #     return saliency_map