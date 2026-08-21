"""Pure score-fusion head — the engine shared by the live metric path
(``models/metric_and_visualization.py``) and the offline replay tool
(``tools/replay_score_fusion.py``).

Spec: docs/superpowers/specs/2026-04-19-source-calibrated-score-fusion-design.md
  - §2.1 current pipeline (the recipe being mirrored)
  - §3.2 calibration objective (per-class normalize → per-class AUROC → mean)
  - §5.1 stat function definitions
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


# Mirrors metric_and_visualization.py:181-184 exactly. Do not extend or shrink
# without updating the live metric path in lockstep.
_DEFAULT_TOP20_CLASSES = frozenset({
    "capsules", "macaroni1", "macaroni2",
    "pipe_fryum", "screw", "cashew", "chewinggum",
})


def default_can_k_for_class(obj: str) -> int:
    """Return the negative-K used by the live default scoring head.

    Negative because it's the index passed to np.partition(..., kth=can_k).
    """
    return -20 if obj in _DEFAULT_TOP20_CLASSES else -2000


def topk_mean(v: np.ndarray) -> float:
    return float(np.mean(v))


def topk_max(v: np.ndarray) -> float:
    return float(np.max(v))


def percentile_95(v: np.ndarray) -> float:
    return float(np.percentile(v, 95))


def topk_mean_times_one_plus_std(v: np.ndarray) -> float:
    return float(np.mean(v)) * (1.0 + float(np.std(v)))


_STAT_FUNCS = {
    "topk_mean": topk_mean,
    "topk_max": topk_max,
    "percentile_95": percentile_95,
    "topk_mean_times_one_plus_std": topk_mean_times_one_plus_std,
}


def _safe_min_max(x: np.ndarray) -> np.ndarray:
    """Per-class min-max normalization mirroring metric_and_visualization.py:207-208."""
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def apply_score_fusion(
    results: Dict[str, Any],
    obj_list: List[str],
    score_fusion_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute fused image-level scores per class.

    Two modes:
      - ``score_fusion_config is None``: default mode. Per-class hardcoded
        ``can_k`` (mirrors :181-184), stat=topk_mean, alpha=0.5. Output is
        bit-equal to the live default path.
      - dict ``{"alpha": float, "k_ratio": float, "stat": str}``: calibrated.
        ``k = round(k_ratio * n_pixels)``, applied class-agnostically.

    Returns a dict with these keys (one entry per class in ``obj_list``):
      - ``fused_per_class[obj]``: ndarray of fused scores (already min-max normalized then alpha-blended)
      - ``gt_per_class[obj]``: ndarray of ground-truth labels
      - ``k_used_per_class[obj]``: int — the K actually applied for this class

    Inputs read from ``results-pro`` (must match shape produced by ``test_universal.py``):
      - ``anomaly_maps``: list of (H, W) float32 ndarrays
      - ``cls_names``: list of str (per image)
      - ``gt_sp``: list of int (per image)
      - ``pr_sp``: list of float (image_score_input — see spec §2.4)
    """
    if score_fusion_config is not None:
        for required in ("alpha", "k_ratio", "stat"):
            if required not in score_fusion_config:
                raise ValueError(f"score_fusion_config missing required key: {required!r}")
        if score_fusion_config["stat"] not in _STAT_FUNCS:
            raise ValueError(
                f"unknown stat: {score_fusion_config['stat']!r} "
                f"(allowed: {sorted(_STAT_FUNCS)})"
            )

    fused_per_class: Dict[str, np.ndarray] = {}
    gt_per_class: Dict[str, np.ndarray] = {}
    k_used_per_class: Dict[str, int] = {}

    for obj in obj_list:
        idxes = [i for i, c in enumerate(results["cls_names"]) if c == obj]
        if not idxes:
            continue

        # Compute per-image map_stat
        if score_fusion_config is None:
            can_k = default_can_k_for_class(obj)
            k_used = abs(can_k)
        else:
            n_pixels = results["anomaly_maps"][idxes[0]].size
            k_used = max(1, int(round(score_fusion_config["k_ratio"] * n_pixels)))
            can_k = -k_used
            stat_fn = _STAT_FUNCS[score_fusion_config["stat"]]

        map_stat_vals = []
        gt_vals: List[int] = []
        pr_inp_vals: List[float] = []
        for i in idxes:
            arr = results["anomaly_maps"][i].reshape(-1)
            # Defensive: if |can_k| exceeds arr size, partition would raise. Clamp.
            ck = max(can_k, -arr.size)
            topk = np.partition(arr, kth=ck)[ck:]
            if score_fusion_config is None:
                # Default mode: keep np.mean's return value as-is (np.float32 scalar) to mirror live.
                map_stat_vals.append(np.mean(topk))
            else:
                map_stat_vals.append(float(stat_fn(topk)))
            gt_vals.append(int(results["gt_sp"][i]))
            pr_inp_vals.append(float(results["pr_sp"][i]))

        # Default mode intentionally preserves the live dtype path: map_stat remains float32 until final fusion with float64 image scores.
        if score_fusion_config is None:
            map_stat_arr = np.array(map_stat_vals)
        else:
            # Calibrated mode: stat funcs other than topk_mean already return python floats.
            # Standardize to float64 for the calibrated path.
            map_stat_arr = np.asarray(map_stat_vals, dtype=np.float64)

        # pr_inp side: list of python floats → np.array gives float64 in both modes,
        # mirroring the live `pr_sp = np.array(pr_sp)`.
        pr_inp_arr = np.array(pr_inp_vals)

        map_stat_arr = _safe_min_max(map_stat_arr)
        pr_inp_arr = _safe_min_max(pr_inp_arr)

        alpha = 0.5 if score_fusion_config is None else float(score_fusion_config["alpha"])
        fused = alpha * pr_inp_arr + (1.0 - alpha) * map_stat_arr

        fused_per_class[obj] = fused
        gt_per_class[obj] = np.asarray(gt_vals, dtype=np.int64)
        k_used_per_class[obj] = k_used

    return {
        "fused_per_class": fused_per_class,
        "gt_per_class": gt_per_class,
        "k_used_per_class": k_used_per_class,
    }
