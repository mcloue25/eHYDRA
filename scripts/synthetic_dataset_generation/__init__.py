"""
Synthetic ground-truth saliency benchmark generation (Priority 2).

Generates univariate synthetic time series with injected discriminative
subsequences at known locations, for four archetypes matching the
UCR morphology clusters used elsewhere in the thesis pipeline:

    high_frequency  -- frequency-burst archetype, raised frequency band
    short_rough     -- frequency-burst archetype, short series + noise
    smooth          -- level-shift archetype (slow trend + level injection)
    spiky           -- multiclass spike-localisation archetype

See generators/base.py for the shared interface, calibration.py for how
per-cluster parameters are derived from
ucr_dataset_clusters_k4_with_types.csv, and dataset_io.py for the
on-disk artifact format consumed by downstream training/evaluation
scripts (outside this package).
"""
