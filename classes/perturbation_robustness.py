'''
Perturbation operator robustness testing
'''
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence
import gc

import numpy as np
import pandas as pd
import torch

from classes.models.hydra_explainable import HydraModelExplainable
from utils.data_utils import load_dataset
from utils.explainability import evaluate_masking_dataset
from utils.perturbation import CORE_PERTURBATIONS, PERTURBATION_LABELS


@dataclass
class PerturbationRobustnessEvaluator:
    ''' Run top/random/bottom masking under multiple perturbation operators for HYDRA.
    '''
    datasets: Sequence[str]
    output_dir: Path | str = Path("outputs/saliency/perturbation_operators")
    operators: Sequence[str] = CORE_PERTURBATIONS
    fractions: Sequence[float] = (0.05, 0.10, 0.20)
    random_repeats: int = 5
    only_correct: bool = True
    max_samples: int | None = 20
    seed: int = 42
    device: str | None = None
    summaries: dict[str, pd.DataFrame] = field(default_factory=dict)

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"



    def run_dataset(self, dataset):
        ''' Fit HYDRA once on this dataset, then evaluate masking under every operator
        '''
        out_path = self.output_dir / f"perturbation_operators_{dataset}_samples.csv"

        # Reuse existing per-dataset output to make interrupted runs resumable.
        if out_path.exists():
            print(f"[SKIP] Loading existing perturbation-operator results for {dataset}")
            return pd.read_csv(out_path)

        print(f"\nPerturbation-operator robustness: {dataset}:")

        X_train, y_train, X_test, y_test, _ = load_dataset(dataset)

        model = HydraModelExplainable(input_dim=X_train.shape[-1], seed=self.seed, device=self.device)
        print("Fitting HYDRA...")
        model.fit(X_train, y_train)

        frames = []
        for operator in self.operators:
            print(f"Evaluating operator: {operator}")
            sample_df, _ = evaluate_masking_dataset(
                model=model,
                X_test=X_test,
                y_test=y_test,
                fractions=self.fractions,
                use_absolute=True,
                only_correct=self.only_correct,
                random_repeats=self.random_repeats,
                seed=self.seed,
                max_samples=self.max_samples,
                perturbation_name=operator,
            )
            sample_df["dataset"] = dataset
            sample_df["model"] = "HYDRA"
            frames.append(sample_df)

        dataset_df = pd.concat(frames, ignore_index=True)
        dataset_df.to_csv(out_path, index=False)

        # Remove old models and data to save on memory
        del model, X_train, y_train, X_test, y_test
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return dataset_df


    def run(self):
        ''' Run all datasets and write the combined samples/summary CSVs
        '''
        frames = []
        skipped = []
        for i, dataset in enumerate(self.datasets, start=1):
            print(f"\n[{i}/{len(self.datasets)}] Perturbation-operator robustness on {dataset}")
            try:
                frames.append(self.run_dataset(dataset))
            except Exception as e:
                print(f"[SKIP] {dataset} failed with {type(e).__name__}: {e}")
                skipped.append({"dataset": dataset, "error": str(e)})
                continue

        if skipped:
            import json
            skip_path = self.output_dir / "skipped_datasets.json"
            skip_path.write_text(json.dumps(skipped, indent=2))
            print(f"\nSkipped {len(skipped)} datasets (see {skip_path}):")
            for s in skipped:
                print(f"  - {s['dataset']}: {s['error'][:80]}")

        if not frames:
            raise RuntimeError("No datasets completed successfully.")

        samples = pd.concat(frames, ignore_index=True)
        samples_path = self.output_dir / "perturbation_operator_samples.csv"
        samples.to_csv(samples_path, index=False)
        summary = self.summarise(samples)
        summary_path = self.output_dir / "perturbation_operator_summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"\nSaved samples: {samples_path}")
        print(f"Saved summary: {summary_path}")
        return {"samples": samples, "summary": summary}


    def summarise(self, samples=None):
        ''' Aggregate per-sample rows into operator x fraction x mode summary metrics
        '''
        if samples is None:
            samples = pd.read_csv(self.output_dir / "perturbation_operator_samples.csv")

        summary = (
            samples.groupby(["perturbation", "fraction", "mode"], as_index=False)
            .agg(
                n_samples=("sample_idx", "count"),
                n_datasets=("dataset", "nunique"),
                mean_score_drop=("score_drop", "mean"),
                mean_bounded_relative_score_drop=("bounded_relative_score_drop", "mean"),
                flip_rate=("flipped", "mean"),
            )
        )
        summary["perturbation_label"] = summary["perturbation"].map(PERTURBATION_LABELS)
        return summary
