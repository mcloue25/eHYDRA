import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
#from xutil import GTRegressor, GTClassifier

class GTRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, predict_fnc):        
        self.is_fitted = True
        self.predict_fnc = predict_fnc

    def fit(self, X=None, y=None):
        self.is_fitted_ = True
        return self       

    def predict(self, X):
        return self.predict_fnc(X)
    

    
class GTClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, regressor, threshold):        
        self.is_fitted = True
        self.threshold = threshold
        self.regressor = regressor

    def fit(self, X=None, y=None):
        self.is_fitted_ = True
        return self
    
    def predict(self, X):
        return self.regressor.predict(X) > self.threshold
    
    def decision_function(self,X):
        return self.regressor.predict(X) - self.threshold

    def predict_proba(self, X):
        y_pred = (self.regressor.predict(X) - self.threshold) * 4 / np.min((np.abs(self.threshold - 10), np.abs(self.threshold - 50)))
        y_prob_pos = 1/(1 + np.exp(-y_pred))
        y_prob = np.vstack((y_prob_pos, 1 - y_prob_pos)).T
        return y_prob

def _generate_feature(n_points = 200, n_support = 40, start_idx = 50, wave = 'sine', f_support = 20, f_base = 0.0):
    """
    Generate a given feature of a sample with a support of a given type and frequency
    overlaid over a sine wave of a given frequency
    Parameters
    ----------
    wave: str
        type of the support wave
    f_support: int
        frequency of the support wave
    f_base: float
        frequency of the base wave
    Returns
    -------
    x_feature: np.array
        feature of the sample
    start_idx: int
        idx where the support starts
    """

    
    x_feature = np.sin(np.linspace(0, 2 * np.pi * f_base, n_points)).reshape(
        -1, 1
    )
    x_feature *= 0.5
    #start_idx = np.random.randint(0, n_points - n_support)

    if wave == "sine":
        x_tmp = np.sin(
            np.linspace(0, 2 * np.pi * f_support, n_points)
        ).reshape(-1, 1)
        start_tmp = 0
        x_feature[start_idx : start_idx + n_support, 0] += x_tmp[
            start_tmp : start_tmp + n_support, 0
        ]

    elif wave == "square":
        x_tmp = np.sign(
            np.sin(np.linspace(0, 2 * np.pi * f_support, n_points))
        ).reshape(-1, 1)
        start_tmp = 0
        x_feature[start_idx : start_idx + n_support, 0] += x_tmp[
            start_tmp : start_tmp + n_support, 0
        ]

    elif wave == "line":
        x_feature[start_idx : start_idx + n_support, 0] += [0] * n_support
    else:
        raise ValueError("wave must be one of sine, square, sawtooth, line")
    return x_feature.reshape(n_points)

def _window_frequency(window, ts_length):
    window_lengths = np.zeros(window.shape[0])
    positive_zcr = np.diff(np.sign(window)) > 0
    x,_,z = np.where(positive_zcr)
    
    for xi in range(window.shape[0]):
        zcr = z[x==xi]
        if len(zcr) == 0:
            window_lengths[xi] = 1 
        else:
            window_lengths[xi] = np.max(zcr) - np.min(zcr) + 1
    positive_zcr_counts = np.sum(positive_zcr > 0, axis = 2).reshape(window.shape[0])
    positive_zcr_counts[positive_zcr_counts == 0] += 1 # to make sure the frequency is 0 when no cycle found

    return ((positive_zcr_counts - 1) * ts_length) / window_lengths


    



class DoubleFreqTest:

    def __init__(self, clf_threshold = 60):
        self.clf_threshold = clf_threshold
        self.tslength = 200
        self.splength = 40
        self.sp_idx = [30,130]
        self.min_f = 10
        self.max_f = 50

    def generate_regression_data_and_attribs(self, n_samples = 100, random_seed = 42):
        np.random.seed(random_seed)
        frequencies = np.random.randint(low = self.min_f, high = self.max_f+1, size=(n_samples,2))
        
        y = np.sum(frequencies, axis=1)    
        X = np.zeros((n_samples, 1 , self.tslength))
        attribs = np.zeros(X.shape)
        for i in range(n_samples):
            X[i,0,:] = _generate_feature(n_points = self.tslength, n_support=self.splength, start_idx = self.sp_idx[0], wave='sine', f_support=frequencies[i,0], f_base=0)
            X[i,0,:] += _generate_feature(n_points = self.tslength, n_support=self.splength, start_idx = self.sp_idx[1], wave='sine', f_support=frequencies[i,1], f_base=0)
            attribs[i,0,self.sp_idx[0]: self.sp_idx[0] + self.splength] = frequencies[i,0]
            attribs[i,0,self.sp_idx[1]: self.sp_idx[1] + self.splength] = frequencies[i,1]
        return X, y, attribs
    
   
    
    def generate_classification_background_sample(self):
        bg = np.zeros((1, 1 , self.tslength))
        bg[0,0,:] = _generate_feature(n_points = self.tslength, n_support=self.splength, start_idx = self.sp_idx[0], wave='sine', f_support=self.clf_threshold/2, f_base=0)
        bg[0,0,:] += _generate_feature(n_points = self.tslength, n_support=self.splength, start_idx = self.sp_idx[1], wave='sine', f_support=self.clf_threshold/2, f_base=0)
        return bg
    
    def generate_classification_data_and_attribs(self, n_samples, random_seed = 42):
        X, y, attribs = self.generate_regression_data_and_attribs(n_samples = n_samples, random_seed=random_seed)        
        attribs[...,self.sp_idx[0]: self.sp_idx[0] + self.splength] -= self.clf_threshold / 2
        attribs[...,self.sp_idx[1]: self.sp_idx[1] + self.splength] -= self.clf_threshold / 2
        return X, y > self.clf_threshold, attribs
    
    def _predict(self, X):
        if len(X.shape) == 2:
            X = X.reshape((1,X.shape[0],X.shape[1]))
        window1 = X[...,self.sp_idx[0]:self.sp_idx[0] + self.splength] 
        window2 = X[...,self.sp_idx[1]:self.sp_idx[1] + self.splength] 

        return _window_frequency(window1, X.shape[-1]) + _window_frequency(window2, X.shape[-1])
    
    def get_regression_model(self):
        return GTRegressor(predict_fnc=self._predict)
    
    def get_classification_model(self):
        return GTClassifier(regressor=self.get_regression_model(), threshold=self.clf_threshold)
    
