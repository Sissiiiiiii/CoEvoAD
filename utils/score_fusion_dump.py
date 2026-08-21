"""Dump top-K raw anomaly values + image-level fusion inputs for offline
score-fusion calibration.

Spec: docs/superpowers/specs/2026-04-19-source-calibrated-score-fusion-design.md
  - §3.3 inputs
  - §7.1 schema (float32, top-65,536 per image, descending)
  - §7.3 pre-gaussian semantic
"""
from __future__ import annotations

import os
from typing import Any, Dict

import numpy as np


def dump_score_fusion_inputs(
    path: str,
    results: Dict[str, Any],
    k_topk: int = 65536,
) -> None:
    """Write a compressed npz with the spec §7.1 schema. No-op if path is empty.

    Pre-conditions:
      - results-pro['anomaly_maps'][i] is float32 ndarray of shape (H, W).
      - results-pro['pr_sp'][i] is castable to float32.
      - All per-image lists in results-pro have the same length.
    """
    if not path:
        return

    n = len(results["cls_names"])
    if not (
        n == len(results["anomaly_maps"])
        == len(results["gt_sp"])
        == len(results["pr_sp"])
        == len(results["path"])
    ):
        raise ValueError(
            "results-pro lists have mismatched lengths; cannot dump score-fusion inputs."
        )

    topk_arr = np.full((n, k_topk), -np.inf, dtype=np.float32)
    n_pixels = np.zeros(n, dtype=np.int32)

    for i in range(n):
        amap = results["anomaly_maps"][i]
        if amap.dtype != np.float32:
            raise ValueError(
                f"anomaly_maps[{i}] dtype must be float32 (spec §7.1); got {amap.dtype}. "
                f"This dtype check exists to prevent fp16 quantization breaking R4 matched replay."
            )
        flat = amap.reshape(-1)
        n_pixels[i] = flat.size
        k_eff = min(k_topk, flat.size)
        topk_unsorted = np.partition(flat, kth=-k_eff)[-k_eff:]
        topk_sorted = np.sort(topk_unsorted)[::-1]
        topk_arr[i, :k_eff] = topk_sorted

    image_score_input = np.asarray(
        [float(v) for v in results["pr_sp"]], dtype=np.float32
    )
    gt_sp = np.asarray([int(v) for v in results["gt_sp"]], dtype=np.int8)
    cls_name = np.asarray([str(v) for v in results["cls_names"]])
    path_arr = np.asarray([str(v) for v in results["path"]])

    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent or ".", exist_ok=True)
    np.savez_compressed(
        path,
        topk_anomaly=topk_arr,
        image_score_input=image_score_input,
        n_pixels=n_pixels,
        gt_sp=gt_sp,
        cls_name=cls_name,
        path=path_arr,
    )
