import numpy as np
import torch


class HydraMarginExplainable:
    ''' Predicted class minus runner up saliency view of a fitted HydraModelExplainable
    '''

    def __init__(self, base_model):
        if not getattr(base_model, "is_fitted", False):
            raise ValueError("base_model must already be fitted (call base_model.fit(...) first).")
        self.base_model = base_model
        self.classifier = base_model.classifier  # exposed for code that inspects .classifier directly

    def predict(self, X):
        return self.base_model.predict(X)

    def decision_function(self, X):
        return self.base_model.decision_function(X)

    def _saliency_for_class_index(self, x_single_t, class_index):
        with torch.inference_mode():
            saliency = self.base_model.transform.get_saliency_map(x_single_t, self.base_model.classifier, self.base_model.scaler, class_index=class_index)

        saliency = np.asarray(saliency, dtype=np.float32)
        if saliency.ndim == 2 and saliency.shape[0] == 1:
            saliency = saliency[0]
        return saliency



    def explain(self, x_single, verbose=False):
        ''' Explain function for winnder vs runner up saliency
        '''
        x_single = np.asarray(x_single, dtype=np.float32)
        x_batched = x_single[None, :]
        x_single_t = self.base_model._to_tensor(x_batched)

        classes = self.base_model.classifier.classes_

        if len(classes) == 2:
            # NOTE - Binary: one decision direction. 
            # The predicted-class-aligned projection already *is* the margin between the two classes.
            pred = self.base_model.predict(x_batched)[0]
            saliency = self._saliency_for_class_index(x_single_t, class_index=0)

            # classifier.coef_ points toward classes_[1], flip sign if the predicted class is classes_[0] so the map is aligned with "evidence for the class that was actually predicted".
            if pred == classes[0]:
                saliency = -saliency

            if verbose:
                print(f"[MarginExplain] Binary case: predicted={pred}, classes={classes}")

            return saliency

        # NOTE - Multiclass: 
        # two term margin between predicted and runner up class.
        decision = np.asarray(self.base_model.decision_function(x_batched))
        scores = decision[0] if decision.ndim == 2 else decision

        order = np.argsort(scores)[::-1]
        pred_index, runnerup_index = int(order[0]), int(order[1])

        saliency_pred = self._saliency_for_class_index(x_single_t, class_index=pred_index)
        saliency_runnerup = self._saliency_for_class_index(x_single_t, class_index=runnerup_index)

        if verbose:
            print(
                f"[MarginExplain] Multiclass case: predicted_index={pred_index} "
                f"(class {classes[pred_index]}), runner_up_index={runnerup_index} "
                f"(class {classes[runnerup_index]})"
            )

        return saliency_pred - saliency_runnerup
