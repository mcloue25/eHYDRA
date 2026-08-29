from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence
import gc

import pandas as pd
import torch

from classes.models.hydra_explainable import HydraModelExplainable
from classes.models.lr_explainable import LRRawExplainableModel
from classes.models.mrsqm_explainable import MrSQMExplainableModel
from utils.data_utils import load_dataset
from utils.explainability import evaluate_masking_dataset


# Mapping from CLI/internal model names to report-friendly names.
MODEL_NAMES = {
    "lr": "LR",
    "hydra": "HYDRA",
    "mrsqm": "MrSQM",
}


@dataclass
class SaliencyEvaluator:
    ''' Run saliency masking evaluation for one or more models over datasets
    '''
    datasets: Sequence[str]
    output_dir: Path | str = Path("outputs/saliency/masking")
    fractions: Sequence[float] = (0.05, 0.10, 0.20)
    random_repeats: int = 5
    only_correct: bool = True
    max_samples: int | None = None
    seed: int = 42
    device: str | None = None
    # Stores model-level summary DataFrames after each run.
    summaries: dict[str, pd.DataFrame] = field(default_factory=dict)


    def __post_init__(self):
        '''Prepare output directory and choose CPU/GPU device'''
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"



    def build_model(self, model_name, input_dim):
        '''Construct the requested explainable model wrapper
        '''
        model_name = model_name.lower()
        if model_name == "lr":
            return LRRawExplainableModel()
        if model_name == "hydra":
            return HydraModelExplainable(input_dim=input_dim, seed=self.seed, device=self.device)
        if model_name == "mrsqm":
            return MrSQMExplainableModel(nsax=5, nsfa=1)
        raise ValueError(f"Unknown model name: {model_name}")



    def evaluate_dataset(self, dataset, model_name):
        ''' Fit one model on one dataset and run masking-based saliency evaluation
        '''
        print(f"\nDataset: {dataset}")
        print(f"Model: {MODEL_NAMES[model_name]}")

        # NOTE - Fite eHYDRA model
        X_train, y_train, X_test, y_test, _ = load_dataset(dataset)
        model = self.build_model(model_name=model_name, input_dim=X_train.shape[-1])
        print("Fitting model...")
        model.fit(X_train, y_train)

        # Evaluate
        print("Evaluating masking...")
        sample_df, summary_df = evaluate_masking_dataset(
            model=model,
            X_test=X_test,
            y_test=y_test,
            fractions=self.fractions,
            use_absolute=True,
            only_correct=self.only_correct,
            random_repeats=self.random_repeats,
            seed=self.seed,
            max_samples=self.max_samples,
        )

        # Add identifiers needed when combining results across datasets/models.
        pretty_name = MODEL_NAMES[model_name]
        sample_df["dataset"] = dataset
        sample_df["model"] = pretty_name
        summary_df["dataset"] = dataset
        summary_df["model"] = pretty_name

        # Save memory space
        del model
        del X_train, y_train, X_test, y_test
        return sample_df, summary_df



    def run_model(self, model_name):
        ''' Run one model across all datasets and append outputs to CSV files
        '''
        model_name = model_name.lower()
        samples_path = self.output_dir / f"{model_name}_samples.csv"
        summary_path = self.output_dir / f"{model_name}_summary.csv"

        # Fresh start for every model level run
        if samples_path.exists():
            samples_path.unlink()
        if summary_path.exists():
            summary_path.unlink()

        summary_rows = []
        for i, dataset in enumerate(self.datasets, start=1):
            print(f"\n[{i}/{len(self.datasets)}] {MODEL_NAMES[model_name]} on {dataset}")
            sample_df, summary_df = self.evaluate_dataset(dataset=dataset, model_name=model_name)

            # Append per-sample and per-dataset summary rows immediately.
            sample_df.to_csv(samples_path, mode="a", header=not samples_path.exists(), index=False)
            summary_df.to_csv(summary_path, mode="a", header=not summary_path.exists(), index=False)
            summary_rows.append(summary_df)
            print(f"Saved dataset outputs for {dataset}")
            # Clean memory between datasets
            del sample_df
            del summary_df
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        summary = pd.concat(summary_rows, ignore_index=True)
        self.summaries[model_name] = summary
        print(f"\nSaved samples to: {samples_path}")
        print(f"Saved summary to: {summary_path}")
        return {
            "samples_path": samples_path,
            "summary_path": summary_path,
            "summary": summary,
        }



    def run(self, model_names=("lr", "hydra", "mrsqm")):
        ''' Run the requested models and return their output paths/summaries
        '''
        results = {}
        print(f"Using device: {self.device}")
        print(f"Writing saliency outputs to: {self.output_dir.resolve()}")

        for model_name in model_names:
            results[model_name] = self.run_model(model_name)

        return results