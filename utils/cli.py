'''Shared command-line helpers used across scripts/robustness_testing and
scripts/tscaptum entry points.
'''

from pathlib import Path

import pandas as pd


def select_datasets_from_closest(closest_csv: Path, datasets_per_cluster: int | None = None) -> list[str]:
    ''' Read the closest-to-centroid CSV and return dataset names.

        If datasets_per_cluster is None, all datasets in the file are returned.
        Otherwise the top N closest datasets per cluster are selected, sorted by
        rank_within_cluster if available, otherwise by distance_to_centroid.
    '''
    closest = pd.read_csv(closest_csv)

    if datasets_per_cluster is None:
        return closest["dataset"].dropna().unique().tolist()

    sort_cols = ["cluster"]
    if "rank_within_cluster" in closest.columns:
        sort_cols.append("rank_within_cluster")
    elif "distance_to_centroid" in closest.columns:
        sort_cols.append("distance_to_centroid")

    selected = (
        closest
        .sort_values(sort_cols)
        .groupby("cluster", group_keys=False)
        .head(datasets_per_cluster)
    )
    return selected["dataset"].dropna().unique().tolist()
