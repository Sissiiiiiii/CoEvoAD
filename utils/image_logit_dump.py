"""Dump prompt-bank image logits for offline scoring-geometry replay."""
from __future__ import annotations

import os
from typing import Any, Dict

import numpy as np


def dump_image_logits(
    path: str,
    results: Dict[str, Any],
    *,
    prompt_num: int,
    scorer_temperature: float = 1.0,
) -> None:
    """Write a compressed npz of per-image prompt-bank image logits."""
    if not path:
        return

    n = len(results["cls_names"])
    if not (
        n == len(results["gt_sp"])
        == len(results["pr_sp"])
        == len(results["path"])
        == len(results.get("image_logits", []))
    ):
        raise ValueError("results-pro lists have mismatched lengths; cannot dump image logits.")

    expected = 2 * int(prompt_num)
    logits = np.asarray(results["image_logits"], dtype=np.float32)
    if logits.shape != (n, expected):
        raise ValueError(
            f"image_logits must have shape {(n, expected)}; got {logits.shape}."
        )

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
        image_logits=logits,
        image_score_input=image_score_input,
        gt_sp=gt_sp,
        cls_name=cls_name,
        path=path_arr,
        prompt_num=np.asarray(int(prompt_num), dtype=np.int32),
        scorer_temperature=np.asarray(float(scorer_temperature), dtype=np.float32),
    )
