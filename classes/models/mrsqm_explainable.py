import numpy as np
from mrsqm import MrSQMClassifier


class MrSQMExplainableModel:
    ''' Wrapper around native MrSQMClassifier for fair fit/predict/explain comparison.
            fit(X, y)
            predict(X)
            get_saliency_map(x_single)
    '''

    def __init__(self, **kwargs):
        # Init model
        self.model = MrSQMClassifier(**kwargs)
        self.is_fitted = False


    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        ''' Train model on full dataset
        '''
        self.model.fit(np.asarray(x_train), np.asarray(y_train))
        self.is_fitted = True
        return self


    def predict(self, x_test: np.ndarray):
        ''' Predict labels for input batch
        '''
        return np.asarray(self.model.predict(np.asarray(x_test)))


    def explain(self, x_single: np.ndarray):
        ''' Return saliency for a single time series
        '''
        x_single = np.asarray(x_single)

        # Expect (T,) shape
        if x_single.ndim != 1:
            raise ValueError(f"Expected shape (T,), got {x_single.shape}")

        sal = self.model.get_saliency_map(x_single)
        sal = np.asarray(sal)

        # Flatten if extra dimension returned
        if sal.ndim == 2:
            sal = sal[0]
        return sal.astype(np.float32)


    def decision_function(self, x: np.ndarray):
        ''' Return decision scores if supported by backend
        '''
        return np.asarray(self.model.decision_function(np.asarray(x)))