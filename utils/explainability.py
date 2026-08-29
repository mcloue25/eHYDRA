import numpy as np
import pandas as pd

from utils.perturbation import get_perturbation


def saliency_to_importance(saliency: np.ndarray, use_absolute: bool = True):
    '''Convert raw saliency values into importance values.'''
    saliency = np.asarray(saliency, dtype=np.float32)
    return np.abs(saliency) if use_absolute else saliency


def select_contiguous_window(importance: np.ndarray, fraction: float, mode: str = "top", rng: np.random.Generator | None = None):
    ''' Select a contiguous window using top, bottom, or random importance
    '''
    importance = np.asarray(importance, dtype=np.float32)
    T = len(importance)

    if T == 0:
        raise ValueError("importance must contain at least one value")

    if mode not in {"top", "bottom", "random"}:
        raise ValueError(f"Unknown masking mode: {mode}")

    window = max(1, int(round(T * fraction)))
    mask = np.zeros(T, dtype=bool)

    if window >= T:
        mask[:] = True
        return mask

    if mode == "random":
        rng = np.random.default_rng() if rng is None else rng
        start = int(rng.integers(0, T - window + 1))
    else:
        scores = np.convolve(
            importance,
            np.ones(window, dtype=np.float32),
            mode="valid",
        )

        if mode == "top":
            start = int(np.argmax(scores))
        else:
            start = int(np.argmin(scores))

    mask[start:start + window] = True
    return mask


def apply_mask(x: np.ndarray, mask: np.ndarray):
    ''' Replace the selected region with the mean value of the original time series.
        Kept as the default/backward-compatible perturbation operator
    '''
    return get_perturbation("global_mean")(x, mask)


def predicted_class_score(decision_output: np.ndarray, pred_label: int):
    ''' Return the decision score for the predicted class
    '''
    decision_output = np.asarray(decision_output)

    if decision_output.ndim == 0:
        return float(decision_output)

    if decision_output.ndim == 1:
        margin = float(decision_output[0])
        return margin if pred_label == 1 else -margin

    return float(decision_output[0, pred_label])


def bounded_relative_score_drop(score_before, score_after, eps=1e-6):
    ''' Compute clipped relative score drop
    '''
    score_drop = score_before - score_after
    return float(np.clip(score_drop / (abs(score_before) + eps), -1.0, 1.0))


def explain_single_cached(model, x, pred_before, use_absolute=True):
    ''' Compute saliency, importance, and baseline score once per sample
    '''
    x = np.asarray(x, dtype=np.float32)
    x_batch = x[None, :]

    score_before = predicted_class_score(model.decision_function(x_batch), pred_before)
    saliency = np.asarray(model.explain(x), dtype=np.float32)
    importance = saliency_to_importance(saliency, use_absolute=use_absolute)
    return {"importance": importance, "score_before": score_before}


def mask_from_cached(model, x, importance, pred_before, score_before, fraction, mode="top", rng=None, perturbation_fn=None):
    ''' Apply a mask selected from cached importance values and evaluate the change.
        `perturbation_fn` selects how the masked window is replaced. 
    '''
    if perturbation_fn is None:
        perturbation_fn = get_perturbation("global_mean")

    mask = select_contiguous_window(importance=importance, fraction=fraction, mode=mode, rng=rng)
    x_masked = perturbation_fn(x, mask, rng=rng)
    x_masked_batch = x_masked[None, :]

    pred_after = int(model.predict(x_masked_batch)[0])
    score_after = predicted_class_score(model.decision_function(x_masked_batch), pred_before)
    # NOTE - Clauclate the actual score drop
    score_drop = score_before - score_after

    return {
        "mask": mask,
        "x_masked": x_masked,
        "pred_after": pred_after,
        "score_after": score_after,
        "score_drop": score_drop,
        "bounded_relative_score_drop": bounded_relative_score_drop(score_before, score_after),
        "flipped": int(pred_before != pred_after),
    }


def pred_to_hydra_class_index(model, pred_label):
    '''  Match the prediction to the class index
    '''
    if hasattr(model, "classes_"):
        matches = np.where(model.classes_ == pred_label)[0]
        if len(matches):
            return int(matches[0])

    if hasattr(model, "classifier") and hasattr(model.classifier, "classes_"):
        matches = np.where(model.classifier.classes_ == pred_label)[0]
        if len(matches):
            return int(matches[0])

    return int(pred_label)


def evaluate_masking_dataset(model, X_test: np.ndarray, y_test: np.ndarray, fractions=(0.05, 0.10, 0.20), use_absolute: bool = True, only_correct: bool = True, random_repeats: int = 5, seed: int = 42, max_samples: int | None = None, perturbation_name: str = "global_mean"):
    ''' Evaluate top, random, and bottom masking across one dataset.
    '''
    rng = np.random.default_rng(seed)
    perturbation_fn = get_perturbation(perturbation_name)

    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test)

    base_preds = model.predict(X_test)
    base_acc = float(np.mean(base_preds == y_test))

    candidate_indices = np.arange(len(X_test))

    if only_correct:
        candidate_indices = candidate_indices[base_preds == y_test]

    if max_samples is not None:
        candidate_indices = candidate_indices[:max_samples]

    rows = []

    for count, i in enumerate(candidate_indices, start=1):
        x = X_test[i]
        pred_before = int(base_preds[i])

        cached = explain_single_cached(model=model, x=x, pred_before=pred_before, use_absolute=use_absolute)

        importance = cached["importance"]
        score_before = cached["score_before"]

        if count % 10 == 0:
            print(f"  processed {count}/{len(candidate_indices)} samples")

        for frac in fractions:
            for mode in ("top", "bottom"):
                out = mask_from_cached(model=model, x=x, importance=importance, pred_before=pred_before, score_before=score_before, fraction=frac, mode=mode, rng=rng, perturbation_fn=perturbation_fn)

                rows.append({
                    "sample_idx": int(i),
                    "fraction": frac,
                    "mode": mode,
                    "perturbation": perturbation_name,
                    "true_label": int(y_test[i]),
                    "base_pred": pred_before,
                    "masked_pred": out["pred_after"],
                    "score_before": score_before,
                    "score_after": out["score_after"],
                    "score_drop": out["score_drop"],
                    "bounded_relative_score_drop": out["bounded_relative_score_drop"],
                    "flipped": out["flipped"],
                })

            random_drops = []
            random_bounded_drops = []
            random_flips = []
            random_preds = []

            for _ in range(random_repeats):
                out = mask_from_cached(model=model, x=x, importance=importance, pred_before=pred_before, score_before=score_before, fraction=frac, mode="random", rng=rng, perturbation_fn=perturbation_fn)
                random_drops.append(out["score_drop"])
                random_bounded_drops.append(out["bounded_relative_score_drop"])
                random_flips.append(out["flipped"])
                random_preds.append(out["pred_after"])

            rows.append({
                "sample_idx": int(i),
                "fraction": frac,
                "mode": "random",
                "perturbation": perturbation_name,
                "true_label": int(y_test[i]),
                "base_pred": pred_before,
                "masked_pred": int(round(np.mean(random_preds))),
                "score_before": score_before,
                "score_after": np.nan,
                "score_drop": float(np.mean(random_drops)),
                "bounded_relative_score_drop": float(np.mean(random_bounded_drops)),
                "flipped": float(np.mean(random_flips)),
            })

    sample_df = pd.DataFrame(rows)

    if sample_df.empty:
        summary_df = pd.DataFrame(
            columns=[
                "fraction",
                "mode",
                "perturbation",
                "n_samples",
                "mean_score_drop",
                "mean_bounded_relative_score_drop",
                "flip_rate",
                "base_accuracy",
            ]
        )
        return sample_df, summary_df

    summary_df = (
        sample_df.groupby(["fraction", "mode", "perturbation"], as_index=False)
        .agg(
            n_samples=("sample_idx", "count"),
            mean_score_drop=("score_drop", "mean"),
            mean_bounded_relative_score_drop=("bounded_relative_score_drop", "mean"),
            flip_rate=("flipped", "mean"),
        )
    )

    summary_df["base_accuracy"] = base_acc
    return sample_df, summary_df