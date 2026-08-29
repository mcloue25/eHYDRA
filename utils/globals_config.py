from __future__ import annotations
from pathlib import Path

# NOTE - FUTURE WORK 
''' Finsihed the plotting gloabls config and now moving onto this. 
    Theres a lot of duplicated definitions of paths and things ive just generally used and technical depbt ive let build up particularly
    in the scrips/ folder, I started moving things to here and will get round to cleaning it up at some point but most likely
    its not going to be done by the 30th because i dont want to risk loudly or silently breaking anything at this stage
'''
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
REPO_ROOT: str = str(PROJECT_ROOT)


# NOTE - Cluster identifiers
CLUSTER_KEYS: list[str] = ["high_frequency", "short_rough", "smooth", "spiky"]

CLUSTER_NAMES: dict[int, str] = {
    0: "High-frequency / high-curvature",
    1: "Smooth / low-complexity",
    2: "Short / moderately rough",
    3: "Spiky / multi-class",
}

CLUSTER_CSV: Path = PROJECT_ROOT / "outputs" / "clustering" / "csv" / "ucr_dataset_clusters_k4_with_types.csv"




# NOTE - Synthetic ground-truth evaluation paths
SYNTH_DIR: str = str(PROJECT_ROOT / "scripts" / "synthetic_dataset_generation")
DEFAULT_DATA_DIR: str = str(PROJECT_ROOT / "data" / "synthetic_dataset")
DEFAULT_MODEL_DIR: str = str(Path(DEFAULT_DATA_DIR) / "trained_models")

GROUND_TRUTH_EVAL_DIR: str = str(Path(DEFAULT_DATA_DIR) / "ground_truth_evaluation")
DEFAULT_EVAL_DIR: str = GROUND_TRUTH_EVAL_DIR

TOPK_FRACTIONS: tuple[float, ...] = (0.05, 0.10, 0.20)




# NOTE - Shared ordering constants
MODE_ORDER: list[str] = ["top", "random", "bottom"]



# NOTE - tsCaptum analysis output
TSCAPTUM_PLOT_DIR: Path = PROJECT_ROOT / "outputs" / "plots" / "tscaptum"