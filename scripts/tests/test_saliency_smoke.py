from pathlib import Path
import sys
import traceback

import numpy as np

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


from classes.models.hydra_explainable import HydraModelExplainable
from classes.models.lr_explainable import LRRawExplainableModel
from classes.models.mrsqm_explainable import MrSQMExplainableModel
from utils.data_utils import load_dataset
from utils.explainability import evaluate_masking_dataset


DATASET = "GunPoint"
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



def test_model(name, model, X_train, y_train, X_test, y_test):
    ''' Double check saliency working as its supposed to  
    '''
    print(f"\n{'=' * 80}")
    print(f"Testing {name}")
    print(f"{'=' * 80}")

    print("Fitting model...")
    model.fit(X_train, y_train)

    print("Predicting...")
    preds = model.predict(X_test[:5])
    print("Predictions:", preds)

    print("Decision function...")
    scores = model.decision_function(X_test[:1])
    print("Decision output shape:", np.asarray(scores).shape)

    print("Explaining one sample...")
    saliency = model.explain(X_test[0])
    saliency = np.asarray(saliency)
    print("Saliency shape:", saliency.shape)
    print("Input shape:", X_test[0].shape)

    if saliency.shape[-1] != X_test[0].shape[-1]:
        print("WARNING: saliency length does not match input length")

    print("Running masking evaluation...")
    sample_df, summary_df = evaluate_masking_dataset(
        model=model,
        X_test=X_test,
        y_test=y_test,
        fractions=(0.10,),
        only_correct=True,
        random_repeats=2,
        seed=SEED,
        max_samples=5,
    )

    print("\nSample results:")
    print(sample_df.head())

    print("\nSummary results:")
    print(summary_df)

    required_sample_cols = {
        "score_drop",
        "bounded_relative_score_drop",
        "flipped",
    }

    missing_sample_cols = required_sample_cols - set(sample_df.columns)
    if missing_sample_cols:
        raise ValueError(f"Missing expected sample columns: {missing_sample_cols}")

    required_summary_cols = {
        "mean_score_drop",
        "mean_bounded_relative_score_drop",
        "flip_rate",
        "base_accuracy",
    }

    missing_summary_cols = required_summary_cols - set(summary_df.columns)
    if missing_summary_cols:
        raise ValueError(f"Missing expected summary columns: {missing_summary_cols}")

    print(f"\n{name} saliency smoke test passed.")


def main():
    ''' Main function for testing LR, eHYDRA & MrSQm saliency on one example 
    '''
    X_train, y_train, X_test, y_test, _ = load_dataset(DATASET)

    print(f"Dataset: {DATASET}")
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)

    models = [
        ("LR", LRRawExplainableModel()),
        (
            "HYDRA",
            HydraModelExplainable(
                input_dim=X_train.shape[-1],
                seed=SEED,
                device=DEVICE,
            ),
        ),
        (
            "MrSQM",
            MrSQMExplainableModel(
                nsax=5,
                nsfa=1,
            ),
        ),
    ]

    failed = []

    for name, model in models:
        try:
            test_model(name, model, X_train, y_train, X_test, y_test)
        except Exception:
            failed.append(name)
            print(f"\n{name} failed:")
            traceback.print_exc()

    print("\nSmoke test complete.")

    if failed:
        print("Failed models:", failed)
        raise SystemExit(1)

    print("All saliency smoke tests passed.")


if __name__ == "__main__":
    main()