import numpy as np
from dtaidistance import dtw
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


class DTWKNNModel:
    '''1-Nearest Neighbour classifier using Dynamic Time Warping distance'''

    def __init__(self):
        self.x_train = None
        self.y_train = None

    def _dtw_distance(self, s1: np.ndarray, s2: np.ndarray) -> float:
        return dtw.distance_fast(s1.astype(np.double), s2.astype(np.double))

    def __call__(self,
                 x_train: np.ndarray,
                 y_train: np.ndarray,
                 x_test: np.ndarray,
                 y_test: np.ndarray):

        # Flatten to 2D if needed: (n_samples, n_timesteps)
        x_train_2d = x_train.reshape(x_train.shape[0], -1)
        x_test_2d = x_test.reshape(x_test.shape[0], -1)

        try:
            self.x_train = x_train_2d
            self.y_train = y_train
        except Exception as e:
            print(f"Unexpected error during fitting: {e}")
            raise e

        try:
            predictions = []
            for i, test_sample in enumerate(x_test_2d):
                distances = np.array([
                    self._dtw_distance(test_sample, train_sample)
                    for train_sample in self.x_train
                ])
                nearest_idx = np.argmin(distances)
                predictions.append(self.y_train[nearest_idx])
            predictions = np.array(predictions)
        except Exception as e:
            print(f"Unexpected error during prediction: {e}")
            raise e

        print(classification_report(y_test, predictions))
        print(confusion_matrix(y_test, predictions))
        try:
            accuracy = accuracy_score(y_test, predictions)
            print(f"Accuracy: {accuracy:.4f}")
        except Exception as e:
            print(f"Unexpected error during accuracy calculation: {e}, {y_test.shape}, {predictions.shape}")
            raise e

        return predictions
