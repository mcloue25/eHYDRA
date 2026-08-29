import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from saliency_ground_truth_metrics import cosine_similarity, confusion_matrix_metrics, topk_overlap, evaluate_one_sample


def _make_ground_truth(T=250, region=(50, 100), value=10.0):
    Phi = np.zeros(T)
    Phi[region[0]:region[1]] = value
    return Phi


def test_cosine_similarity_perfect_match():
    Phi = _make_ground_truth()
    assert abs(cosine_similarity(Phi, Phi) - 1.0) < 1e-9


def test_cosine_similarity_perfect_opposite():
    Phi = _make_ground_truth()
    assert abs(cosine_similarity(-Phi, Phi) - (-1.0)) < 1e-9


def test_cosine_similarity_orthogonal():
    T = 250
    Phi = np.zeros(T); Phi[50:100] = 10.0
    phi = np.zeros(T); phi[150:200] = 10.0  # disjoint support -> zero dot product
    assert abs(cosine_similarity(phi, Phi)) < 1e-9


def test_confusion_matrix_perfect_match():
    Phi = _make_ground_truth()
    cm = confusion_matrix_metrics(Phi, Phi, magnitude_tol=0.05)
    assert cm["precision"] == 1.0
    assert cm["recall"] == 1.0
    assert cm["f1"] == 1.0
    assert cm["n_false_relevant"] == 0
    assert cm["n_false_irrelevant"] == 0


def test_confusion_matrix_wrong_sign_is_false_relevant():
    Phi = _make_ground_truth()
    phi = -Phi.copy()  # relevant region correctly located, but wrong sign everywhere
    cm = confusion_matrix_metrics(phi, Phi, magnitude_tol=0.05)
    assert cm["precision"] == 0.0
    assert cm["n_false_relevant"] == 50
    assert cm["n_true_relevant"] == 0


def test_confusion_matrix_no_signal_gives_zero_recall():
    Phi = _make_ground_truth()
    phi = np.random.RandomState(0).normal(0, 0.01, size=len(Phi))  # tiny noise everywhere, no real signal
    cm = confusion_matrix_metrics(phi, Phi, magnitude_tol=0.9)  # strict threshold -> almost nothing "relevant"
    assert cm["recall"] < 0.2  # true region almost entirely missed


def test_topk_overlap_perfect_match():
    Phi = _make_ground_truth(T=250, region=(50, 100))  # 50/250 = 20% of series
    tk = topk_overlap(Phi, Phi, q=0.20)
    assert abs(tk["precision"] - 1.0) < 1e-9
    assert abs(tk["recall"] - 1.0) < 1e-9
    assert abs(tk["iou"] - 1.0) < 1e-9
    assert tk["window_size"] == 50


def test_topk_overlap_disjoint_gives_zero():
    T = 250
    Phi = np.zeros(T); Phi[50:100] = 10.0
    phi = np.zeros(T); phi[150:200] = 10.0
    tk = topk_overlap(phi, Phi, q=0.20)
    assert tk["precision"] == 0.0
    assert tk["recall"] == 0.0
    assert tk["iou"] == 0.0


def test_topk_overlap_partial_overlap():
    T = 250
    Phi = np.zeros(T); Phi[50:100] = 10.0 # true region: 50 points
    phi = np.zeros(T); phi[75:125] = 10.0 # top-20% region: 50 points, overlaps [75,100) = 25 points
    tk = topk_overlap(phi, Phi, q=0.20)
    assert tk["window_size"] == 50
    assert abs(tk["precision"] - 0.5) < 1e-9 # 25/50
    assert abs(tk["recall"] - 0.5) < 1e-9  # 25/50
    assert abs(tk["iou"] - (25 / 75)) < 1e-9  # 25 / (50+50-25)


def test_evaluate_one_sample_runs_and_returns_all_keys():
    Phi = _make_ground_truth()
    result = evaluate_one_sample(Phi, Phi)
    expected_keys = ["cosine_similarity", "confmat_precision", "confmat_recall", "confmat_f1", "topk5_precision", "topk10_precision", "topk20_precision"]
    for k in expected_keys:
        assert k in result, f"missing key: {k}"
    assert abs(result["cosine_similarity"] - 1.0) < 1e-9


if __name__ == "__main__":
    tests = [f for name, f in globals().items() if name.startswith("test_") and callable(f)]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print(f"\n{len(tests)} tests passed")
