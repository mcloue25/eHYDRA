'''
Abstract interface for the synthetic ground truth saliency benchmark generators.
'''
from abc import ABC, abstractmethod
import numpy as np


class SyntheticGroundTruthGenerator(ABC):
    '''Common interface for all cluster type generators
    '''

    cluster_key: str = None
    cluster_name: str = None

    @abstractmethod
    def generate_data_and_attribs(self, n_samples: int, seed: int = 42):
        ''' Generates a classification dataset with ground-truth attributions
        Returns
            X (np.ndarray) : shape (n_samples, 1, tslength)
            y (np.ndarray) : shape (n_samples,) bool for binary archetypes, int class index for multiclass.
            attribs : np.ndarray, shape (n_samples, 1, tslength)
                Ground-truth signed attribution Phi. Zero outside injected
                regions, graded and signed inside them.
        '''
        raise NotImplementedError



    @abstractmethod
    def generate_background_sample(self) -> np.ndarray:
        ''' Returns a (1, 1, tslength) sample on the hypothetical model's decision boundary
        '''
        raise NotImplementedError

    @abstractmethod
    def get_classifier_model(self):
        ''' Returns an sklearn-compatible estimator wrapping the HYPOTHETICAL
            model (zero-crossing counter / level estimator / spike detector), NOT
            a trained HYDRA -- used only for the Tier A harness-validation checks
            in validate_reproduction.py. The main Priority 2 evaluation (Tier B)
            trains a real HYDRA on this generator's data and evaluates *that*
            model's saliency against `attribs` instead, outside this package
        '''
        raise NotImplementedError

    def metadata(self) -> dict:
        ''' Params + provenance recorded alongside every generated artifact.
            Subclasses should extend the dict returned by super().metadata()
            with their own config.
        '''
        return {
            "cluster_key": self.cluster_key,
            "cluster_name": self.cluster_name,
            "generator_class": type(self).__name__,
        }