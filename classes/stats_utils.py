'''Shared statistical tests (paired Wilcoxon, gap, consistency, Kruskal) reusedacross every module that reports a paired top-vs-baseline test
'''

import numpy as np
import pandas as pd
from scipy.stats import kruskal, wilcoxon


def paired_wilcoxon(df: pd.DataFrame, group_col: str, condition_col: str, value_col: str, candidate: str = "top", baseline: str = "random", alternative: str = "greater"):
    ''' Paired one-sided Wilcoxon signed-rank test of candidate > baseline, grouped by group_col
    '''
    pivot = df.pivot_table(index=group_col, columns=condition_col, values=value_col, aggfunc="mean")

    if candidate not in pivot.columns or baseline not in pivot.columns:
        return {
            "mean_diff": np.nan,
            "candidate_greater": 0,
            "n_non_tied": 0,
            "candidate_greater_count": "0/0",
            "wilcoxon_p": np.nan,
        }

    valid = pivot.dropna(subset=[candidate, baseline])
    diff = valid[candidate] - valid[baseline]
    non_tied = diff[diff != 0]  # drop ties before testing, matches zero_method="wilcox"

    if len(non_tied) == 0:
        return {
            "mean_diff": float(diff.mean()) if len(diff) else np.nan,
            "candidate_greater": 0,
            "n_non_tied": 0,
            "candidate_greater_count": "0/0",
            "wilcoxon_p": np.nan,
        }
    _, p_value = wilcoxon(non_tied, alternative=alternative, zero_method="wilcox")
    candidate_greater = int((non_tied > 0).sum())
    total = int(len(non_tied))

    return {
        "mean_diff": float(diff.mean()),
        "candidate_greater": candidate_greater,
        "n_non_tied": total,
        "candidate_greater_count": f"{candidate_greater}/{total}",
        "wilcoxon_p": float(p_value),
    }


def dataset_level_gap(df: pd.DataFrame, group_col: str, condition_col: str, value_col: str, candidate: str = "top", baseline: str = "random"):
    ''' Mean candidate-minus-baseline gap per group_col
    '''
    pivot = df.pivot_table(index=group_col, columns=condition_col, values=value_col, aggfunc="mean")
    if candidate not in pivot.columns or baseline not in pivot.columns:
        return pd.DataFrame(columns=[group_col, "gap"])
    gap = (pivot[candidate] - pivot[baseline]).rename("gap").reset_index()
    return gap


def consistency_rate(df: pd.DataFrame, group_col: str, condition_col: str, value_col: str, candidate: str = "top", baseline: str = "random"):
    '''Fraction of groups where candidate's mean value exceeds baseline's
    '''
    gap = dataset_level_gap(df, group_col, condition_col, value_col, candidate, baseline)
    if gap.empty:
        return np.nan, 0
    rate = float((gap["gap"] > 0).mean())
    return rate, int(len(gap))


def kruskal_across(df: pd.DataFrame, split_col: str, value_col: str):
    ''' Kruskal-Wallis test of whether value_col differs across split_col groups
    '''
    groups = [g[value_col].dropna().values for _, g in df.groupby(split_col)]
    groups = [g for g in groups if len(g) > 0]  # drop empty groups, kruskal needs >=1 sample each
    if len(groups) < 2:
        return np.nan, np.nan, len(groups)
    stat, p_value = kruskal(*groups)
    return float(stat), float(p_value), len(groups)