"""Pure helpers for the ICSR soft-mixture gate.

Kept in a dedicated module so unit tests can exercise this logic without
pulling in the full ``test_universal`` runtime (which drags in
``datasets``/``imgaug``/``albumentations`` and friends).
"""
from __future__ import annotations

import json
import logging
import math
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


_SUPPORTED_RANGE_SOURCES: Tuple[str, ...] = ("fixed",)


@dataclass(frozen=True)
class SoftMixOptions:
    """Options driving the soft-mixture gate. Frozen so the ranges can be
    treated as read-only constants for the life of a run."""
    formula: str
    range_source: str
    top1_sim_range: Tuple[float, float] = (0.15, 0.25)
    margin_range: Tuple[float, float] = (0.00, 0.05)
    h_norm_range: Tuple[float, float] = (0.95, 1.00)

    def __post_init__(self) -> None:
        if self.range_source not in _SUPPORTED_RANGE_SOURCES:
            raise NotImplementedError(
                f"range_source={self.range_source!r} not implemented "
                f"(supported: {_SUPPORTED_RANGE_SOURCES})"
            )


def _clip_normalize(v: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (v - lo) / (hi - lo)))


def compute_soft_mix_alpha(
    meta: Optional[Dict[str, Any]],
    opts: SoftMixOptions,
) -> float:
    """Return scalar alpha in [0, 1]. Engineering-safe: None / NaN / missing key -> 0.0."""
    if meta is None:
        return 0.0
    try:
        top1_sim = float(meta["top1_sim"])
        margin = float(meta["top1_top2_margin"])
        h_norm = float(meta["H_norm"])
    except (KeyError, TypeError, ValueError):
        return 0.0
    if any(math.isnan(v) for v in (top1_sim, margin, h_norm)):
        return 0.0

    a_sim = _clip_normalize(top1_sim, *opts.top1_sim_range)
    a_margin = _clip_normalize(margin, *opts.margin_range)
    h_width = opts.h_norm_range[1] - opts.h_norm_range[0]
    a_H = _clip_normalize(1.0 - h_norm, 0.0, h_width)

    if opts.formula == "F_A":
        return (a_sim + a_margin + a_H) / 3.0
    raise NotImplementedError(f"formula {opts.formula!r} is not implemented")


def _blend_score_and_map(
    alpha: float,
    icsr_score: Optional[float],
    icsr_map: Optional[np.ndarray],
    semantic_score: float,
    semantic_map: Optional[np.ndarray],
) -> Tuple[float, Optional[np.ndarray]]:
    """Blend ICSR and semantic (score, pixel_map) by alpha.

    Safety falls to pure-semantic when icsr side is unusable or shapes differ.
    Endpoint identities: alpha=0 -> pure semantic; alpha=1 -> pure icsr.
    """
    if semantic_map is None:
        return float(semantic_score), None
    if icsr_score is None or icsr_map is None:
        return float(semantic_score), semantic_map
    if tuple(icsr_map.shape) != tuple(semantic_map.shape):
        logging.getLogger(__name__).error(
            "soft_mix pixel-map shape mismatch: icsr=%s semantic=%s; falling back to semantic.",
            tuple(icsr_map.shape), tuple(semantic_map.shape),
        )
        return float(semantic_score), semantic_map

    a = float(alpha)
    if a <= 0.0:
        return float(semantic_score), semantic_map
    if a >= 1.0:
        return float(icsr_score), icsr_map
    blended_score = a * float(icsr_score) + (1.0 - a) * float(semantic_score)
    blended_map = a * icsr_map.astype(np.float32) + (1.0 - a) * semantic_map.astype(np.float32)
    return blended_score, blended_map


def _dump_debug_scores(path, results) -> None:
    """Write per-image scores/classes/gt/path to JSON for regression checks.

    No-op if ``path`` is empty or None. Only the four fixed keys are serialized,
    so the payload stays schema-stable.
    """
    if not path:
        return
    payload = {
        "pr_sp": [float(v) for v in results.get("pr_sp", [])],
        "cls_names": [str(v) for v in results.get("cls_names", [])],
        "gt_sp": [int(v) for v in results.get("gt_sp", [])],
        "path": [str(v) for v in results.get("path", [])],
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh)


def _flag_explicitly_passed(flag_name: str) -> bool:
    for arg in sys.argv[1:]:
        if arg == flag_name or arg.startswith(f"{flag_name}="):
            return True
    return False


_SOFT_MIX_IGNORED_HARD_GATE_FLAGS: Tuple[str, ...] = (
    "--icsr_gate_entropy_threshold",
    "--icsr_min_sim",
    "--icsr_min_margin",
)


def _compute_ignored_soft_mix_flags(soft_mix_enabled: bool) -> List[str]:
    """Return the subset of hard-gate flags the user explicitly passed but
    that soft_mix silently ignores. Empty when soft_mix is off."""
    if not soft_mix_enabled:
        return []
    return [f for f in _SOFT_MIX_IGNORED_HARD_GATE_FLAGS if _flag_explicitly_passed(f)]
