import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class LRRawExplainableModel:
    '''Logistic Regression on raw time series features with simple saliency.
    '''
    def __init__(self):
        # Standardise features before training
        self.scaler = StandardScaler()

        # NOTE - LR setup
        self.classifier = LogisticRegression(solver="lbfgs", class_weight="balanced", max_iter=1000, multi_class="auto", C=1.0,)
        self.classes_ = None



    def _reshape_X(self, X):
        X = np.asarray(X)
        # Handle (n_samples, series_length, 1) becomes flatten
        if X.ndim == 3:
            return X.reshape(X.shape[0], -1)
        # Handle single sample (series_length,) becomes (1, series_length)
        if X.ndim == 1:
            return X.reshape(1, -1)
        if X.ndim == 2:
            # Handle single sample (series_length, 1)
            if X.shape[1] == 1:
                return X.reshape(1, -1)
            # Already in correct 2D shape
            return X
        raise ValueError(f"Unexpected X shape: {X.shape}")



    def fit(self, X, y):
        ''' Main fitting function 
        '''
        # Flatten & scale input
        X_2d = self._reshape_X(X)
        X_scaled = self.scaler.fit_transform(X_2d)

        # NOTE - Train classifier
        self.classifier.fit(X_scaled, y)
        self.classes_ = self.classifier.classes_
        return self



    def predict(self, X):
        ''' Main function for predicting classifications
        '''
        # Apply same reshape + scaling at inference
        X_2d = self._reshape_X(X)
        X_scaled = self.scaler.transform(X_2d)
        return self.classifier.predict(X_scaled)



    def decision_function(self, X):
        ''' Main function for generating decision function scores
        '''
        # Return raw decision scores (used for saliency eval)
        X_2d = self._reshape_X(X)
        X_scaled = self.scaler.transform(X_2d)
        return self.classifier.decision_function(X_scaled)


    def explain(self, X, y=None):
        ''' Return saliency using classifier coefficients.
            Basically importance = absolute weight per time step.
        '''
        X_arr = np.asarray(X)
        X_2d = self._reshape_X(X_arr)
        n_samples, series_length = X_2d.shape
        # NOTE - generate predictions to get access to coefficients
        preds = self.predict(X)
        coef = self.classifier.coef_
        # Store per sample saliency
        saliency = np.zeros((n_samples, series_length), dtype=float)

        # Binary LR stores only one weight vector
        binary_case = (coef.shape[0] == 1 and len(self.classes_) == 2)
        for i in range(n_samples):
            # Use labels if available otherwise predictions
            class_label = y[i] if y is not None else preds[i]
            if binary_case:
                # Flip sign depending on class
                if class_label == self.classes_[1]:
                    w = coef[0]
                else:
                    w = -coef[0]
            else:
                # Select weights for pred/true class
                class_idx = np.where(self.classes_ == class_label)[0][0]
                w = coef[class_idx]

            # NOTE - Saliency = magnitude of weights
            saliency[i] = np.abs(w)

        # Return shape same as input format
        if X_arr.ndim == 1:
            return saliency[0]

        if X_arr.ndim == 2 and X_arr.shape[1] == 1:
            return saliency[0].reshape(-1, 1)

        return saliency