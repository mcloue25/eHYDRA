''' Predicted-class vs. margin saliency perturbation comparison.
    Fits one HYDRA model per dataset, then runs top/random/bottom masking protocol twice against the *same* fitted model, once using the
    current predicted class saliency (`HydraModelExplainable.explain`) and then again  using margin saliency (`HydraMarginExplainable.explain`, 
'''

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import gc

import numpy as np
import pandas as pd
import torch

from classes.models.hydra_explainable import HydraModelExplainable
from classes.models.hydra_margin_explainable import HydraMarginExplainable
from utils.data_utils import load_dataset
from utils.explainability import evaluate_masking_dataset

VARIANT_LABELS = {
    "predicted": "Predicted-class saliency",
    "margin": "Margin saliency",
}


@dataclass
class MarginSaliencyEvaluator:
    ''' Compare predicted class saliency against margin saliency under perturbation
    '''
    datasets: Sequence[str]
    output_dir: Path | str = Path("outputs/saliency/margin_saliency")
    fraction: float = 0.10
    random_repeats: int = 5
    max_samples_per_dataset: int | None = 20
    seed: int = 42
    device: str | None = None

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


    def run_dataset(self, dataset):
        out_path = self.output_dir / f"margin_saliency_{dataset}_samples.csv"
        if out_path.exists():
            print(f"[SKIP] Loading existing margin-saliency results for {dataset}")
            return pd.read_csv(out_path)

        print(f"\nPredicted vs. margin saliency: {dataset}:")
        X_train, y_train, X_test, y_test, _ = load_dataset(dataset)

        base_model = HydraModelExplainable(input_dim=X_train.shape[-1], seed=self.seed, device=self.device)
        base_model.fit(X_train, y_train)

        margin_model = HydraMarginExplainable(base_model)

        frames = []
        for variant, model in [("predicted", base_model), ("margin", margin_model)]:
            print(f"Evaluating variant: {variant}")
            sample_df, _ = evaluate_masking_dataset(
                model=model,
                X_test=X_test,
                y_test=y_test,
                fractions=(self.fraction,),
                use_absolute=True,
                only_correct=True,
                random_repeats=self.random_repeats,
                seed=self.seed,
                max_samples=self.max_samples_per_dataset,
            )
            sample_df["dataset"] = dataset
            sample_df["variant"] = variant
            sample_df["variant_label"] = VARIANT_LABELS[variant]
            frames.append(sample_df)

        dataset_df = pd.concat(frames, ignore_index=True)
        dataset_df.to_csv(out_path, index=False)

        del base_model, margin_model, X_train, y_train, X_test, y_test
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return dataset_df

    def run(self):
        ''' Main run fucntion 
        '''
        frames = []
        for i, dataset in enumerate(self.datasets, start=1):
            print(f"\n[{i}/{len(self.datasets)}] Margin saliency comparison on {dataset}")
            frames.append(self.run_dataset(dataset))
        # Generate samples and summary DF's
        samples = pd.concat(frames, ignore_index=True)
        samples_path = self.output_dir / "margin_saliency_samples.csv"
        samples.to_csv(samples_path, index=False)
        summary = self.summarise(samples)
        summary_path = self.output_dir / "margin_saliency_summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"\nSaved samples: {samples_path}")
        print(f"Saved summary: {summary_path}")
        return {"samples": samples, "summary": summary}


    def summarise(self, samples=None):
        if samples is None:
            samples = pd.read_csv(self.output_dir / "margin_saliency_samples.csv")
        summary = (
            samples.groupby(["variant", "variant_label", "mode"], as_index=False)
            .agg(
                n_samples=("sample_idx", "count"),
                n_datasets=("dataset", "nunique"),
                mean_score_drop=("score_drop", "mean"),
                mean_bounded_relative_score_drop=("bounded_relative_score_drop", "mean"),
                flip_rate=("flipped", "mean"),
            )
        )
        return summary
