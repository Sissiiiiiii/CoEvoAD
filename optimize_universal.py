"""
Universal Stage 2 - checkpoint-aware, model-agnostic prompt evolution.

Uses the AnomalyScorer abstraction, supporting a deterministic Prompt Bank and arbitrary external models.
Stage2 optimizes a combined image+pixel objective by default and supports checkpoint-aware custom scorers.

Usage:
    # Stage2 mainline: pure CLIP transfer scorer
    python optimize_universal.py --scorer_type clip_transfer --dataset visa \
        --use_coevo_prompt --evo_dual_branch

    # Prompt bank scorer
    python optimize_universal.py --scorer_type prompt_bank --dataset visa \
        --checkpoint_path ./my_exps/.../stage1_final.pth \
        --use_coevo_prompt --evo_dual_branch --stage2_objective image_pixel

    # Custom scorer
    python optimize_universal.py --scorer_type custom \
        --scorer_module models.scorer_template --scorer_class VanillaCLIPScorer \
        --dataset visa --checkpoint_path ./my_exps/.../custom_stage1.pth \
        --evo_dual_branch --stage2_objective image_pixel

:author: PromptBank Universal Stage 2
:date: 2026
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from utils.common import setup_seed, setup_logger, _transform_test
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from tqdm import tqdm

from datasets import Makedataset
from models.evoprompt import EvoPromptOptimizer
from models.scorer import build_scorer
from utils.cuda_memory import log_cuda_peak_memory, reset_cuda_peak_memory

# Module-level logger for functions outside run_stage2_universal() scope.
# Points to the same named logger that setup_logger() configures, so handlers
# attached there will pick up messages from here once setup_logger() has run.
_module_logger = logging.getLogger("optimize_universal")


def _enforce_transfer_mainline(args, logger=None):
    """Record the Stage2 mainline semantics: the core is model-agnostic and the checkpoint is interpreted by the scorer itself."""
    scorer_type = getattr(args, "scorer_type", "prompt_bank")
    msg = (
        f"Stage2 mainline: scorer='{scorer_type}' "
        "with model-agnostic optimizer core; checkpoint semantics are delegated to scorer"
    )
    if logger:
        logger.info(msg)
    else:
        print(f"[info] {msg}")

    if scorer_type == "prompt_bank" and not getattr(args, "checkpoint_path", ""):
        msg = "prompt_bank scorer requires --checkpoint_path for Stage2."
        if logger:
            logger.warning(msg)
        else:
            print(f"[warn] {msg}")


def _normalize_stage2_category_name(name: str) -> str:
    return str(name).strip().lower()


def _filter_stage2_categories(
    obj_list: List[str],
    requested: Optional[List[str]],
) -> List[str]:
    if not requested:
        return list(obj_list)
    wanted: set[str] = set()
    for item in requested:
        if item is None:
            continue
        for part in str(item).split(","):
            token = _normalize_stage2_category_name(part)
            if token:
                wanted.add(token)
    if not wanted:
        return list(obj_list)
    return [name for name in obj_list if _normalize_stage2_category_name(name) in wanted]


def _parse_float_or_none(text: str) -> Optional[float]:
    text = str(text).strip()
    if text.lower() == "none":
        return None
    return float(text)


def _parse_stage2_resume_log_text(log_text: str) -> Dict[str, Any]:
    optimize_re = re.compile(r"Optimize:\s*([A-Za-z0-9_.-]+)")
    normal_re = re.compile(r"\bnormal=(.+)$")
    abnormal_re = re.compile(r"\babnormal=(.+)$")
    shared_re = re.compile(r"\bshared=(.+)$")
    safe_meta_re = re.compile(
        r"safe_meta:\s*src\s+([-+]?\d*\.?\d+)→([-+]?\d*\.?\d+)\s+\(gain=([-+]?\d*\.?\d+)\),\s*"
        r"cross\s+(None|[-+]?\d*\.?\d+)→(None|[-+]?\d*\.?\d+)\s+\(gain=(None|[-+]?\d*\.?\d+)\),\s*"
        r"score_std=([-+]?\d*\.?\d+)"
    )

    recovered_rules: Dict[str, Dict[str, str]] = {"normal": {}, "abnormal": {}, "shared": {}}
    recovered_meta: Dict[Tuple[str, str], Dict[str, Any]] = {}
    completed_categories: List[str] = []
    completed_seen = set()
    category_state: Dict[str, Dict[str, Any]] = {}
    current_category: Optional[str] = None

    for raw_line in str(log_text).splitlines():
        line = raw_line.strip()
        if not line:
            continue

        match = optimize_re.search(line)
        if match:
            current_category = _normalize_stage2_category_name(match.group(1))
            category_state.setdefault(current_category, {})
            continue

        if current_category is None:
            continue

        state = category_state.setdefault(current_category, {})

        match = normal_re.search(line)
        if match:
            state["normal"] = match.group(1).strip()
            continue

        match = abnormal_re.search(line)
        if match:
            state["abnormal"] = match.group(1).strip()
            continue

        match = shared_re.search(line)
        if match:
            state["shared"] = match.group(1).strip()
            if current_category not in completed_seen:
                recovered_rules["shared"][current_category] = state["shared"]
                completed_categories.append(current_category)
                completed_seen.add(current_category)
            continue

        match = safe_meta_re.search(line)
        if match:
            state["safe_meta"] = {
                "category": current_category,
                "default_src": float(match.group(1)),
                "best_src": float(match.group(2)),
                "gain_src": float(match.group(3)),
                "default_cross": _parse_float_or_none(match.group(4)),
                "best_cross": _parse_float_or_none(match.group(5)),
                "gain_cross": _parse_float_or_none(match.group(6)),
                "score_std": float(match.group(7)),
                "recovered_from_log": True,
            }

        if (
            current_category not in completed_seen
            and "normal" in state
            and "abnormal" in state
            and "safe_meta" in state
        ):
            recovered_rules["normal"][current_category] = state["normal"]
            recovered_rules["abnormal"][current_category] = state["abnormal"]
            recovered_meta[("pair", current_category)] = dict(state["safe_meta"])
            completed_categories.append(current_category)
            completed_seen.add(current_category)

    return {
        "optimized_rules": recovered_rules,
        "rule_metadata": recovered_meta,
        "completed_categories": completed_categories,
    }


def _serialize_evo_optimizer_state(evo_optimizer) -> Dict[str, Any]:
    state = {
        "cache": {
            f"{role}_{name}": prompt
            for (role, name), prompt in evo_optimizer.cache.items()
        },
        "metadata": {
            f"{role}_{name}": meta
            for (role, name), meta in evo_optimizer.rule_metadata.items()
        },
        "templates": evo_optimizer.templates,
        "normal_templates": evo_optimizer.normal_templates,
        "abnormal_templates": evo_optimizer.abnormal_templates,
        "adjectives": evo_optimizer.adjectives,
        "normal_adjectives": evo_optimizer.normal_adjectives,
        "abnormal_adjectives": evo_optimizer.abnormal_adjectives,
        "population_size": evo_optimizer.population_size,
        "generations": evo_optimizer.generations,
        "topk": evo_optimizer.topk,
        "lambda_diversity": evo_optimizer.lambda_diversity,
    }
    if getattr(evo_optimizer, "record_population_trace", False):
        state["population_trace_schema"] = "coevo_population_trace_v1"
        state["population_trace"] = list(getattr(evo_optimizer, "population_trace", []))
    return state


def _write_stage2_partial_state(
    save_path: str,
    optimized_rules: Dict[str, Dict[str, str]],
    evo_optimizer,
) -> None:
    os.makedirs(save_path, exist_ok=True)
    rules_path = os.path.join(save_path, "optimized_prompt_rules.partial.json")
    cache_path = os.path.join(save_path, "evo_prompt_cache.partial.json")
    with open(rules_path, "w", encoding="utf-8") as f:
        json.dump(optimized_rules, f, indent=2, ensure_ascii=False)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(_serialize_evo_optimizer_state(evo_optimizer), f, indent=2, ensure_ascii=False)


def _load_stage2_partial_state(save_path: str) -> Dict[str, Any]:
    recovered = {
        "optimized_rules": {"normal": {}, "abnormal": {}, "shared": {}},
        "cache": {},
        "rule_metadata": {},
    }
    rules_path = os.path.join(save_path, "optimized_prompt_rules.partial.json")
    cache_path = os.path.join(save_path, "evo_prompt_cache.partial.json")
    if os.path.exists(rules_path):
        with open(rules_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        for role in ("normal", "abnormal", "shared"):
            role_payload = payload.get(role, {})
            if isinstance(role_payload, dict):
                recovered["optimized_rules"][role] = dict(role_payload)
    if os.path.exists(cache_path):
        tmp_optimizer = EvoPromptOptimizer()
        tmp_optimizer.load_optimized_rules(cache_path)
        recovered["cache"] = dict(tmp_optimizer.cache)
        recovered["rule_metadata"] = dict(tmp_optimizer.rule_metadata)
    return recovered


def _merge_stage2_recovered_state(
    optimized_rules: Dict[str, Dict[str, str]],
    evo_optimizer,
    recovered: Dict[str, Any],
    *,
    overwrite: bool = False,
) -> None:
    for role in ("normal", "abnormal", "shared"):
        for category, prompt in recovered.get("optimized_rules", {}).get(role, {}).items():
            if overwrite or category not in optimized_rules[role]:
                optimized_rules[role][category] = prompt
            cache_key = (role, evo_optimizer._normalize_name(category))
            if overwrite or cache_key not in evo_optimizer.cache:
                evo_optimizer.cache[cache_key] = prompt

    for (role, category), meta in recovered.get("rule_metadata", {}).items():
        meta_key = (role, evo_optimizer._normalize_name(category))
        if overwrite or meta_key not in evo_optimizer.rule_metadata:
            evo_optimizer.rule_metadata[meta_key] = dict(meta)


def _completed_stage2_categories(optimized_rules: Dict[str, Dict[str, str]]) -> List[str]:
    completed = set(optimized_rules.get("shared", {}).keys())
    normal_done = set(optimized_rules.get("normal", {}).keys())
    abnormal_done = set(optimized_rules.get("abnormal", {}).keys())
    completed |= (normal_done & abnormal_done)
    return sorted(completed)


def _validate_transfer_regularizer_args(args, parser=None):
    """Validate CDACE / CCTO mutual exclusion and the CCTO preconditions."""
    if getattr(args, "adaptive_ccto_alpha", False) and not getattr(args, "ccto", False):
        msg = "--adaptive_ccto_alpha requires --ccto."
        if parser is not None:
            parser.error(msg)
        raise ValueError(msg)

    min_alpha = float(getattr(args, "ccto_alpha_min", 0.3))
    max_alpha = float(getattr(args, "ccto_alpha_max", 0.9))
    if min_alpha > max_alpha:
        msg = (
            f"Invalid CCTO alpha bounds: ccto_alpha_min ({min_alpha}) "
            f"> ccto_alpha_max ({max_alpha})."
        )
        if parser is not None:
            parser.error(msg)
        raise ValueError(msg)

    std_threshold = float(getattr(args, "ccto_std_threshold", 0.5))
    if std_threshold <= 0.0:
        msg = f"Invalid ccto_std_threshold ({std_threshold}). Must be > 0."
        if parser is not None:
            parser.error(msg)
        raise ValueError(msg)

    if getattr(args, "adaptive_ccto_alpha", False):
        collapse_min_unique = int(getattr(args, "adaptive_ccto_collapse_min_unique", 2))
        collapse_min_std = float(getattr(args, "adaptive_ccto_collapse_min_std", 0.01))
        collapse_min_spread = float(getattr(args, "adaptive_ccto_collapse_min_spread", 0.05))
        collapse_mode = str(getattr(args, "adaptive_ccto_collapse_mode", "warn"))
        reliability_clip = float(getattr(args, "adaptive_ccto_reliability_clip", 2.5))
        reliability_eps = float(getattr(args, "adaptive_ccto_reliability_eps", 1e-6))
        if collapse_min_unique < 1:
            msg = (
                f"Invalid adaptive_ccto_collapse_min_unique ({collapse_min_unique}). "
                "Must be >= 1."
            )
            if parser is not None:
                parser.error(msg)
            raise ValueError(msg)
        if collapse_min_std < 0.0:
            msg = (
                f"Invalid adaptive_ccto_collapse_min_std ({collapse_min_std}). "
                "Must be >= 0."
            )
            if parser is not None:
                parser.error(msg)
            raise ValueError(msg)
        if collapse_min_spread < 0.0:
            msg = (
                f"Invalid adaptive_ccto_collapse_min_spread ({collapse_min_spread}). "
                "Must be >= 0."
            )
            if parser is not None:
                parser.error(msg)
            raise ValueError(msg)
        if reliability_clip < 0.0:
            msg = (
                f"Invalid adaptive_ccto_reliability_clip ({reliability_clip}). "
                "Must be >= 0."
            )
            if parser is not None:
                parser.error(msg)
            raise ValueError(msg)
        if reliability_eps <= 0.0:
            msg = (
                f"Invalid adaptive_ccto_reliability_eps ({reliability_eps}). "
                "Must be > 0."
            )
            if parser is not None:
                parser.error(msg)
            raise ValueError(msg)
        if collapse_mode == "remap" and abs(max_alpha - min_alpha) < 1e-12:
            msg = (
                "adaptive_ccto_collapse_mode=remap requires ccto_alpha_min < ccto_alpha_max."
            )
            if parser is not None:
                parser.error(msg)
            raise ValueError(msg)

    ccto_cross_agg = getattr(args, "ccto_cross_agg", "mean")
    if ccto_cross_agg not in {"mean", "min", "bottomk"}:
        msg = f"Invalid --ccto_cross_agg '{ccto_cross_agg}'. Must be mean/min/bottomk."
        if parser is not None:
            parser.error(msg)
        raise ValueError(msg)
    ccto_bottomk = int(getattr(args, "ccto_bottomk", 3))
    if ccto_bottomk < 1:
        msg = f"Invalid --ccto_bottomk ({ccto_bottomk}). Must be >= 1."
        if parser is not None:
            parser.error(msg)
        raise ValueError(msg)

    evo_fitness_agg = getattr(args, "evo_fitness_agg", "mean")
    if evo_fitness_agg not in {"mean", "cvar"}:
        msg = f"Invalid --evo_fitness_agg '{evo_fitness_agg}'. Must be mean/cvar."
        if parser is not None:
            parser.error(msg)
        raise ValueError(msg)
    evo_cvar_k = int(getattr(args, "evo_cvar_k", 3))
    if evo_cvar_k < 1:
        msg = f"Invalid --evo_cvar_k ({evo_cvar_k}). Must be >= 1."
        if parser is not None:
            parser.error(msg)
        raise ValueError(msg)

    if getattr(args, "asym_b_enable", False):
        if not getattr(args, "ccto", False):
            msg = "--asym_b_enable requires --ccto."
            if parser is not None:
                parser.error(msg)
            raise ValueError(msg)
        if not getattr(args, "evo_dual_branch", False):
            msg = "--asym_b_enable requires --evo_dual_branch."
            if parser is not None:
                parser.error(msg)
            raise ValueError(msg)

        lambda_ng = float(getattr(args, "asym_b_lambda_normal_gen", 0.35))
        lambda_as = float(getattr(args, "asym_b_lambda_abn_spec", 0.20))
        if lambda_ng < 0.0 or lambda_ng > 1.0:
            msg = (
                f"Invalid asym_b_lambda_normal_gen ({lambda_ng}). "
                "Expected range is [0, 1]."
            )
            if parser is not None:
                parser.error(msg)
            raise ValueError(msg)
        if lambda_as < 0.0:
            msg = (
                f"Invalid asym_b_lambda_abn_spec ({lambda_as}). "
                "Expected value is >= 0."
            )
            if parser is not None:
                parser.error(msg)
            raise ValueError(msg)
        policy = str(getattr(args, "asym_b_policy", "fixed")).strip().lower()
        if policy not in {"fixed", "manual_kappa"}:
            msg = f"Invalid asym_b_policy ({policy}). Expected fixed/manual_kappa."
            if parser is not None:
                parser.error(msg)
            raise ValueError(msg)
        if policy == "manual_kappa":
            for name in ("asym_b_kappa_normal", "asym_b_kappa_abnormal"):
                value = float(getattr(args, name, 0.0))
                if value < -1.0 or value > 1.0:
                    msg = f"Invalid {name} ({value}). Expected range is [-1, 1]."
                    if parser is not None:
                        parser.error(msg)
                    raise ValueError(msg)

    if getattr(args, "ccto", False):
        if getattr(args, "cdace_target_dataset", "") or getattr(args, "cdace_target_categories", None):
            msg = "--ccto is mutually exclusive with --cdace_target_dataset/--cdace_target_categories."
            if parser is not None:
                parser.error(msg)
            raise ValueError(msg)
        if not getattr(args, "evo_dual_branch", False):
            msg = "--ccto requires --evo_dual_branch because CCTO only regularizes the abnormal branch."
            if parser is not None:
                parser.error(msg)
            raise ValueError(msg)


def _resolve_effective_ccto_scope(args, requested_scope: Optional[str] = None) -> str:
    """Resolve runtime CCTO scope with asymmetric-objective guardrail."""
    scope = (requested_scope or "abnormal_only").strip().lower()
    if getattr(args, "asym_b_enable", False):
        return "symmetric"
    if scope not in {"abnormal_only", "symmetric"}:
        return "abnormal_only"
    return scope


def _resolve_asym_b_policy(args) -> str:
    policy = str(getattr(args, "asym_b_policy", "fixed")).strip().lower()
    if policy not in {"fixed", "manual_kappa"}:
        return "fixed"
    return policy


def _resolve_asym_b_kappa(role: Optional[str], args) -> float:
    safe_role = (role or "shared").strip().lower()
    policy = _resolve_asym_b_policy(args)
    if policy == "manual_kappa":
        if safe_role == "normal":
            return float(getattr(args, "asym_b_kappa_normal", 0.35))
        if safe_role == "abnormal":
            return float(getattr(args, "asym_b_kappa_abnormal", -0.20))
        return 0.0

    lambda_ng = float(getattr(args, "asym_b_lambda_normal_gen", 0.35))
    lambda_as = float(getattr(args, "asym_b_lambda_abn_spec", 0.20))
    if safe_role == "normal":
        return lambda_ng
    if safe_role == "abnormal":
        return -lambda_as
    return 0.0


def _asym_b_config_dict(args, effective_ccto_scope: Optional[str] = None) -> Dict[str, Any]:
    return {
        "enabled": True,
        "policy": _resolve_asym_b_policy(args),
        "lambda_normal_gen": float(getattr(args, "asym_b_lambda_normal_gen", 0.35)),
        "lambda_abn_spec": float(getattr(args, "asym_b_lambda_abn_spec", 0.20)),
        "kappa_normal": float(_resolve_asym_b_kappa("normal", args)),
        "kappa_abnormal": float(_resolve_asym_b_kappa("abnormal", args)),
        "effective_ccto_scope": effective_ccto_scope,
    }


def _compute_asym_b_final_score(
    role: Optional[str],
    src_score: float,
    cross_score: float,
    args,
) -> Dict[str, float]:
    """Compute role-conditioned asymmetric objective score with bounded range."""
    safe_role = (role or "shared").strip().lower()
    # Keep source/cross scores in their original scale for formula fidelity.
    # Only clip the final score that enters optimizer selection.
    src = float(src_score)
    cross = float(cross_score)
    policy = _resolve_asym_b_policy(args)
    kappa = _resolve_asym_b_kappa(safe_role, args)
    raw = src + kappa * (cross - src)
    final = float(np.clip(raw, 0.0, 1.0))

    return {
        "role": safe_role,
        "policy": policy,
        "kappa": float(kappa),
        "src_score": src,
        "cross_score": cross,
        "raw_score": float(raw),
        "final_score": float(final),
    }


def _aggregate_cross_scores(scores: List[float], agg: str, bottomk: int = 3) -> float:
    """Aggregate per-category cross scores into a single scalar.

    Args:
        scores: Per-category scores (one float per cross-category).
        agg: Aggregation mode -- "mean", "min", or "bottomk".
        bottomk: Number of lowest scores to average when agg="bottomk".

    Returns:
        Aggregated scalar score.
    """
    if not scores:
        return 0.0
    if agg == "mean":
        return float(np.mean(scores))
    if agg == "min":
        return float(np.min(scores))
    if agg == "bottomk":
        k = max(1, min(bottomk, len(scores)))
        return float(np.mean(sorted(scores)[:k]))
    raise ValueError(f"Unknown ccto_cross_agg: {agg!r}")


def _compute_ccto_diag_stats(
    src_scores: List[float],
    cross_scores: List[float],
    blended_scores: List[float],
    evo_topk: int,
) -> Optional[Dict[str, float]]:
    """Compute CCTO diagnostic stats for logging."""
    n = len(blended_scores)
    if n <= 1:
        return None

    diag: Dict[str, float] = {
        "n": float(n),
        "cross_spread": float("nan"),
        "blended_spread": float(max(blended_scores) - min(blended_scores)),
        "rank_corr_src_blended": float("nan"),
        "topk_overlap": float("nan"),
    }

    if len(cross_scores) == n:
        diag["cross_spread"] = float(max(cross_scores) - min(cross_scores))

    if len(src_scores) == n and n >= 3:
        try:
            from scipy.stats import spearmanr

            corr, _ = spearmanr(src_scores, blended_scores)
            diag["rank_corr_src_blended"] = float(corr) if not np.isnan(corr) else 0.0
        except ImportError:
            pass

    if len(src_scores) == n:
        k = max(1, min(int(evo_topk), n))
        topk_by_src = set(np.argsort(src_scores)[-k:])
        topk_by_blended = set(np.argsort(blended_scores)[-k:])
        diag["topk_overlap"] = float(len(topk_by_src & topk_by_blended) / k)

    return diag


def _load_model_config(args):
    """Load the model configuration."""
    if getattr(args, "scorer_type", "prompt_bank") == "clip_transfer":
        return
    config_path = getattr(args, "config_path", "")
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            model_configs = json.load(f)
        args.vision_width = model_configs["vision_cfg"]["width"]
        args.text_width = model_configs["text_cfg"]["width"]
        args.embed_dim = model_configs["embed_dim"]


def _safe_roc_auc(labels: np.ndarray, preds: np.ndarray) -> float:
    labels = np.asarray(labels).reshape(-1)
    preds = np.asarray(preds).reshape(-1)
    if labels.size == 0 or len(np.unique(labels)) < 2:
        return 0.5
    try:
        return float(roc_auc_score(labels, preds))
    except Exception:
        return 0.5


def _safe_average_precision(labels: np.ndarray, preds: np.ndarray) -> float:
    labels = np.asarray(labels).reshape(-1)
    preds = np.asarray(preds).reshape(-1)
    if labels.size == 0 or labels.max() <= 0:
        return 0.0
    try:
        return float(average_precision_score(labels, preds))
    except Exception:
        return 0.0


def _best_f1_and_threshold(labels: np.ndarray, preds: np.ndarray) -> tuple[float, float]:
    labels = np.asarray(labels).reshape(-1)
    preds = np.asarray(preds).reshape(-1)
    if labels.size == 0 or labels.max() <= 0:
        return 0.0, 0.5
    try:
        precisions, recalls, thresholds = precision_recall_curve(labels, preds)
    except Exception:
        return 0.0, 0.5
    if thresholds.size == 0:
        return 0.0, 0.5
    f1_scores = (2.0 * precisions[:-1] * recalls[:-1]) / (
        precisions[:-1] + recalls[:-1] + 1e-8
    )
    if f1_scores.size == 0:
        return 0.0, 0.5
    best_idx = int(np.nanargmax(f1_scores))
    return float(np.nanmax(f1_scores)), float(thresholds[best_idx])


def _binary_iou(labels: np.ndarray, binary_preds: np.ndarray) -> float:
    labels = np.asarray(labels).astype(bool)
    binary_preds = np.asarray(binary_preds).astype(bool)
    union = np.logical_or(labels, binary_preds).sum()
    if union <= 0:
        return 0.0
    inter = np.logical_and(labels, binary_preds).sum()
    return float(inter / union)


def _resolve_stage2_metric_weights(args) -> tuple[float, float, float]:
    weight_image = max(0.0, float(getattr(args, "stage2_weight_image", 0.3)))
    weight_pixel_ap = max(0.0, float(getattr(args, "stage2_weight_pixel_ap", 0.5)))
    weight_pixel_f1 = max(0.0, float(getattr(args, "stage2_weight_pixel_f1", 0.2)))
    weight_sum = weight_image + weight_pixel_ap + weight_pixel_f1
    if weight_sum <= 0:
        return 0.3, 0.5, 0.2
    return (
        weight_image / weight_sum,
        weight_pixel_ap / weight_sum,
        weight_pixel_f1 / weight_sum,
    )


def _weighted_harmonic_mean(
    values: Sequence[float],
    weights: Sequence[float],
) -> float:
    denom = 0.0
    for value, weight in zip(values, weights):
        w = float(weight)
        if w <= 0.0:
            continue
        v = float(value)
        if v <= 0.0:
            return 0.0
        denom += w / v
    if denom <= 0.0:
        return 0.0
    return float(1.0 / denom)


def _resize_metric_tensor(
    tensor: Optional[torch.Tensor],
    resolution: int,
    mode: str,
) -> Optional[np.ndarray]:
    if tensor is None:
        return None
    t = tensor.detach().float()
    if t.ndim == 3:
        t = t.unsqueeze(1)
    elif t.ndim == 2:
        t = t.unsqueeze(0).unsqueeze(0)
    if t.ndim != 4:
        raise ValueError(f"Unsupported metric tensor shape: {tuple(t.shape)}")
    t = F.interpolate(
        t,
        size=(resolution, resolution),
        mode=mode,
        align_corners=False if mode in {"bilinear", "bicubic"} else None,
    )
    return t.squeeze(1).detach().cpu().numpy().astype(np.float32)


def _resize_metric_mask(mask_tensor: Any, resolution: int) -> Optional[np.ndarray]:
    if mask_tensor is None:
        return None
    if isinstance(mask_tensor, torch.Tensor):
        t = mask_tensor.detach().float()
    else:
        t = torch.as_tensor(mask_tensor, dtype=torch.float32)
    if t.ndim == 3:
        t = t.unsqueeze(1)
    elif t.ndim == 2:
        t = t.unsqueeze(0).unsqueeze(0)
    if t.ndim != 4:
        raise ValueError(f"Unsupported mask tensor shape: {tuple(t.shape)}")
    t = F.interpolate(t, size=(resolution, resolution), mode="nearest")
    return (t.squeeze(1).cpu().numpy() > 0.5).astype(np.uint8)


def _normalize_eval_result(
    result: Dict[str, Any],
    batch: Dict[str, Any],
    args,
) -> Dict[str, Any]:
    if not isinstance(result, dict):
        raise TypeError(f"scorer.evaluate_candidate must return dict, got {type(result)}")

    image_scores = result.get("image_scores", None)
    if image_scores is None:
        raise ValueError("scorer.evaluate_candidate result missing 'image_scores'")
    if isinstance(image_scores, torch.Tensor):
        image_scores = image_scores.detach().float().cpu().numpy()
    else:
        image_scores = np.asarray(image_scores, dtype=np.float32)
    image_scores = image_scores.reshape(-1).astype(np.float32)

    pixel_maps = _resize_metric_tensor(
        result.get("pixel_maps", None),
        resolution=int(getattr(args, "stage2_metric_resolution", 256)),
        mode="bilinear",
    )
    teacher_pixel_maps = _resize_metric_tensor(
        result.get("teacher_pixel_maps", None),
        resolution=int(getattr(args, "stage2_metric_resolution", 256)),
        mode="bilinear",
    )

    teacher_image_scores = result.get("teacher_image_scores", None)
    if teacher_image_scores is not None:
        if isinstance(teacher_image_scores, torch.Tensor):
            teacher_image_scores = teacher_image_scores.detach().float().cpu().numpy()
        else:
            teacher_image_scores = np.asarray(teacher_image_scores, dtype=np.float32)
        teacher_image_scores = teacher_image_scores.reshape(-1).astype(np.float32)

    return {
        "labels": batch["labels"],
        "pixel_masks": batch["pixel_masks"],
        "image_scores": image_scores,
        "pixel_maps": pixel_maps,
        "teacher_image_scores": teacher_image_scores,
        "teacher_pixel_maps": teacher_pixel_maps,
    }


def _compute_objective_metrics(
    batch_outputs: List[Dict[str, Any]],
    objective: str,
    args,
) -> Dict[str, float]:
    labels = np.concatenate([x["labels"] for x in batch_outputs], axis=0)
    image_scores = np.concatenate([x["image_scores"] for x in batch_outputs], axis=0)
    image_auroc = _safe_roc_auc(labels, image_scores)
    metrics: Dict[str, float] = {
        "image_auroc": image_auroc,
        "score": image_auroc,
    }

    pixel_masks = [x["pixel_masks"] for x in batch_outputs if x.get("pixel_masks") is not None]
    pixel_maps = [x["pixel_maps"] for x in batch_outputs if x.get("pixel_maps") is not None]

    if objective == "image_only":
        return metrics

    if objective in {"image_pixel", "image_pixel_hmean"}:
        if len(pixel_masks) != len(batch_outputs) or len(pixel_maps) != len(batch_outputs):
            raise ValueError("image_pixel objective requires scorer pixel_maps and source pixel masks")
        gt_px = np.concatenate(pixel_masks, axis=0).astype(np.uint8)
        pr_px = np.concatenate(pixel_maps, axis=0).astype(np.float32)
        pixel_ap = _safe_average_precision(gt_px.reshape(-1), pr_px.reshape(-1))
        pixel_f1, best_threshold = _best_f1_and_threshold(gt_px.reshape(-1), pr_px.reshape(-1))
        pixel_iou = _binary_iou(gt_px.reshape(-1), pr_px.reshape(-1) > best_threshold)
        weight_image, weight_pixel_ap, weight_pixel_f1 = _resolve_stage2_metric_weights(args)
        if objective == "image_pixel_hmean":
            score = _weighted_harmonic_mean(
                [image_auroc, pixel_ap, pixel_f1],
                [weight_image, weight_pixel_ap, weight_pixel_f1],
            )
        else:
            score = (
                weight_image * image_auroc
                + weight_pixel_ap * pixel_ap
                + weight_pixel_f1 * pixel_f1
            )
        metrics.update(
            {
                "pixel_ap": pixel_ap,
                "pixel_f1": pixel_f1,
                "pixel_iou": pixel_iou,
                "pixel_threshold": best_threshold,
                "score": score,
            }
        )
        return metrics

    if objective == "image_teacher":
        teacher_image = [x["teacher_image_scores"] for x in batch_outputs if x.get("teacher_image_scores") is not None]
        teacher_pixel = [x["teacher_pixel_maps"] for x in batch_outputs if x.get("teacher_pixel_maps") is not None]
        teacher_terms = []
        if teacher_image:
            t_img = np.concatenate(teacher_image, axis=0).astype(np.float32)
            teacher_terms.append(float(1.0 - np.mean(np.abs(image_scores - t_img))))
        if teacher_pixel and pixel_maps:
            t_px = np.concatenate(teacher_pixel, axis=0).astype(np.float32)
            pr_px = np.concatenate(pixel_maps, axis=0).astype(np.float32)
            teacher_terms.append(float(1.0 - np.mean(np.abs(pr_px - t_px))))
        if not teacher_terms:
            raise ValueError(
                "image_teacher objective requires teacher_image_scores or teacher_pixel_maps "
                "from scorer.evaluate_candidate"
            )
        teacher_score = float(np.clip(np.mean(teacher_terms), 0.0, 1.0))
        metrics.update(
            {
                "teacher_score": teacher_score,
                "score": 0.5 * image_auroc + 0.5 * teacher_score,
            }
        )
        return metrics

    raise ValueError(f"Unknown stage2_objective: {objective}")


def _bootstrap_regularized_score(
    batch_outputs: List[Dict[str, Any]],
    candidate: str,
    role: str,
    args,
    objective_override: Optional[str] = None,
) -> Dict[str, float]:
    objective = objective_override or getattr(args, "stage2_objective", "image_pixel")
    metrics = _compute_objective_metrics(batch_outputs, objective, args)

    n_bootstrap = max(1, int(getattr(args, "stage2_stability_bootstrap", 3)))
    weight = float(getattr(args, "stage2_stability_weight", 0.1))
    if n_bootstrap <= 1 or len(batch_outputs) <= 1:
        metrics["score_mean"] = metrics["score"]
        metrics["score_std"] = 0.0
        return metrics

    seed_payload = f"{getattr(args, 'seed', 0)}::{role}::{candidate}".encode("utf-8")
    seed = int(hashlib.sha1(seed_payload).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    sample_scores = [metrics["score"]]
    for _ in range(n_bootstrap - 1):
        idxs = rng.integers(0, len(batch_outputs), size=len(batch_outputs))
        subset = [batch_outputs[int(i)] for i in idxs]
        sample_scores.append(_compute_objective_metrics(subset, objective, args)["score"])
    score_mean = float(np.mean(sample_scores))
    score_std = float(np.std(sample_scores))
    metrics["score_mean"] = score_mean
    metrics["score_std"] = score_std
    metrics["score"] = score_mean - weight * score_std
    return metrics


def _iter_candidate_chunks(candidates, chunk_size: int):
    chunk_size = max(1, int(chunk_size))
    for idx in range(0, len(candidates), chunk_size):
        yield candidates[idx: idx + chunk_size]


def _cache_text_features(
    scorer,
    optimizer,
    candidates: List[str],
    role: str,
) -> None:
    if optimizer is None or not hasattr(optimizer, "text_feat_cache"):
        return
    for candidate in candidates:
        feat = scorer.get_text_embedding(candidate, role)
        if feat is not None:
            optimizer.text_feat_cache[(role or "shared", candidate)] = feat


def _collect_candidate_outputs_on_cache(
    scorer,
    optimizer,
    eval_cache: List[Dict[str, Any]],
    candidates: List[str],
    role: str,
    baseline: str,
    args,
    use_coevo: bool,
) -> List[List[Dict[str, Any]]]:
    if not candidates:
        return []

    batch_outputs_by_candidate = [[] for _ in candidates]
    for batch_idx, batch in enumerate(eval_cache):
        results = scorer.evaluate_candidates(
            prepared=batch["prepared"],
            candidates=candidates,
            role=role,
            baseline=baseline,
        )
        if len(results) != len(candidates):
            raise ValueError(
                f"scorer.evaluate_candidates returned {len(results)} results "
                f"for {len(candidates)} candidates"
            )
        for idx, result in enumerate(results):
            batch_outputs_by_candidate[idx].append(
                _normalize_eval_result(result, batch, args)
            )
        if batch_idx == 0 and use_coevo:
            _cache_text_features(scorer, optimizer, candidates, role)

    return batch_outputs_by_candidate


def _evaluate_candidates_on_cache(
    scorer,
    optimizer,
    eval_cache: List[Dict[str, Any]],
    candidates: List[str],
    role: str,
    baseline: str,
    args,
    use_coevo: bool,
    objective_override: Optional[str] = None,
) -> List[Dict[str, float]]:
    if not candidates:
        return []

    batch_outputs_by_candidate = _collect_candidate_outputs_on_cache(
        scorer=scorer,
        optimizer=optimizer,
        eval_cache=eval_cache,
        candidates=candidates,
        role=role,
        baseline=baseline,
        args=args,
        use_coevo=use_coevo,
    )

    return [
        _bootstrap_regularized_score(
            batch_outputs_by_candidate[idx],
            candidate,
            role or "shared",
            args,
            objective_override=objective_override,
        )
        for idx, candidate in enumerate(candidates)
    ]


def _evaluate_candidate_on_cache(
    scorer,
    optimizer,
    eval_cache: List[Dict[str, Any]],
    candidate: str,
    role: str,
    baseline: str,
    args,
    use_coevo: bool,
    objective_override: Optional[str] = None,
) -> Dict[str, float]:
    return _evaluate_candidates_on_cache(
        scorer=scorer,
        optimizer=optimizer,
        eval_cache=eval_cache,
        candidates=[candidate],
        role=role,
        baseline=baseline,
        args=args,
        use_coevo=use_coevo,
        objective_override=objective_override,
    )[0]


def _evaluate_candidates_on_cache_map_macro(
    scorer,
    optimizer,
    cache_by_category: Dict[str, List[Dict[str, Any]]],
    candidates: List[str],
    role: str,
    baseline: str,
    args,
    use_coevo: bool,
    objective_override: Optional[str] = None,
) -> Tuple[List[float], int]:
    macro_metrics, total_batches = _evaluate_candidates_on_cache_map_macro_metrics(
        scorer=scorer,
        optimizer=optimizer,
        cache_by_category=cache_by_category,
        candidates=candidates,
        role=role,
        baseline=baseline,
        args=args,
        use_coevo=use_coevo,
        objective_override=objective_override,
    )
    return [float(metric["score"]) for metric in macro_metrics], total_batches


def _evaluate_candidates_on_cache_map_macro_metrics(
    scorer,
    optimizer,
    cache_by_category: Dict[str, List[Dict[str, Any]]],
    candidates: List[str],
    role: str,
    baseline: str,
    args,
    use_coevo: bool,
    objective_override: Optional[str] = None,
) -> Tuple[List[Dict[str, float]], int]:
    """Evaluate candidates per category and macro-average the scores. Returns a list of per-candidate metrics dicts."""
    if not candidates:
        return [], 0

    device = getattr(scorer, "device", None)
    if device is None and hasattr(scorer, "base_scorer"):
        device = getattr(scorer.base_scorer, "device", None)
    if device is None:
        raise ValueError("CCTO requires scorer.device to reload offloaded eval caches")

    per_candidate_metrics = [[] for _ in candidates]
    total_batches = 0

    for _, cat_cache in cache_by_category.items():
        _reload_eval_cache(cat_cache, device)
        try:
            cat_metrics = _evaluate_candidates_on_cache(
                scorer=scorer,
                optimizer=optimizer,
                eval_cache=cat_cache,
                candidates=candidates,
                role=role,
                baseline=baseline,
                args=args,
                use_coevo=use_coevo,
                objective_override=objective_override,
            )
        finally:
            _offload_eval_cache(cat_cache)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        total_batches += len(cat_cache)
        for idx, metric in enumerate(cat_metrics):
            per_candidate_metrics[idx].append(metric)

    macro_metrics: List[Dict[str, float]] = []
    _fitness_agg = getattr(args, "evo_fitness_agg", "mean")
    if _fitness_agg == "cvar":
        _agg = "bottomk"
        _bk = int(getattr(args, "evo_cvar_k", 3))
    else:
        _agg = getattr(args, "ccto_cross_agg", "mean")
        _bk = int(getattr(args, "ccto_bottomk", 3))
    for metrics_list in per_candidate_metrics:
        if not metrics_list:
            macro_metrics.append({"score": 0.0, "score_mean": 0.0, "score_std": 0.0})
            continue
        macro_metrics.append(
            {
                "score": _aggregate_cross_scores(
                    [float(m.get("score", 0.0)) for m in metrics_list], _agg, _bk
                ),
                "score_mean": _aggregate_cross_scores(
                    [float(m.get("score_mean", m.get("score", 0.0))) for m in metrics_list], _agg, _bk
                ),
                "score_std": _aggregate_cross_scores(
                    [float(m.get("score_std", 0.0)) for m in metrics_list],
                    "mean" if _fitness_agg == "cvar" else _agg, _bk,
                ),
            }
        )
    return macro_metrics, total_batches


def _default_prompt_for_role(base_prompt: str, role: Optional[str]) -> str:
    if not isinstance(base_prompt, str):
        return str(base_prompt)
    name = base_prompt[2:] if base_prompt.startswith("X ") else base_prompt
    if role == "normal":
        return f"X normal {name}".strip()
    if role == "abnormal":
        return f"X abnormal {name}".strip()
    return base_prompt


def _evaluate_prompt_pair_on_cache(
    scorer,
    eval_cache: List[Dict[str, Any]],
    normal_prompt: str,
    abnormal_prompt: str,
    args,
    objective_override: Optional[str] = None,
) -> Dict[str, float]:
    batch_outputs: List[Dict[str, Any]] = []
    for batch in eval_cache:
        result = scorer.evaluate_prompt_pair(
            prepared=batch["prepared"],
            normal_prompt=normal_prompt,
            abnormal_prompt=abnormal_prompt,
            stage=2,
        )
        batch_outputs.append(_normalize_eval_result(result, batch, args))
    return _bootstrap_regularized_score(
        batch_outputs=batch_outputs,
        candidate=f"{normal_prompt} || {abnormal_prompt}",
        role="pair",
        args=args,
        objective_override=objective_override,
    )


def _evaluate_prompt_pair_on_cache_map_macro(
    scorer,
    cache_by_category: Dict[str, List[Dict[str, Any]]],
    normal_prompt: str,
    abnormal_prompt: str,
    args,
    objective_override: Optional[str] = None,
) -> Tuple[Dict[str, float], int]:
    if not cache_by_category:
        return {"score": 0.0, "score_mean": 0.0, "score_std": 0.0}, 0

    device = getattr(scorer, "device", None)
    if device is None and hasattr(scorer, "base_scorer"):
        device = getattr(scorer.base_scorer, "device", None)
    if device is None:
        raise ValueError("Prompt-pair macro eval requires scorer.device to reload offloaded eval caches")

    metrics_list: List[Dict[str, float]] = []
    total_batches = 0
    for cat_cache in cache_by_category.values():
        _reload_eval_cache(cat_cache, device)
        try:
            metrics = _evaluate_prompt_pair_on_cache(
                scorer=scorer,
                eval_cache=cat_cache,
                normal_prompt=normal_prompt,
                abnormal_prompt=abnormal_prompt,
                args=args,
                objective_override=objective_override,
            )
        finally:
            _offload_eval_cache(cat_cache)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        metrics_list.append(metrics)
        total_batches += len(cat_cache)

    _agg = getattr(args, "ccto_cross_agg", "mean")
    _bk = int(getattr(args, "ccto_bottomk", 3))
    return {
        "score": _aggregate_cross_scores(
            [float(m.get("score", 0.0)) for m in metrics_list], _agg, _bk
        ),
        "score_mean": _aggregate_cross_scores(
            [float(m.get("score_mean", m.get("score", 0.0))) for m in metrics_list], _agg, _bk
        ),
        "score_std": _aggregate_cross_scores(
            [float(m.get("score_std", 0.0)) for m in metrics_list], _agg, _bk
        ),
    }, total_batches


def _compute_pair_safe_metadata(
    scorer,
    eval_cache: List[Dict[str, Any]],
    cross_cache_by_category: Optional[Dict[str, List[Dict[str, Any]]]],
    category: str,
    best_normal_prompt: str,
    best_abnormal_prompt: str,
    args,
) -> Dict[str, Any]:
    default_normal_prompt = f"X normal {category}"
    default_abnormal_prompt = f"X abnormal {category}"

    default_src = _evaluate_prompt_pair_on_cache(
        scorer=scorer,
        eval_cache=eval_cache,
        normal_prompt=default_normal_prompt,
        abnormal_prompt=default_abnormal_prompt,
        args=args,
    )
    best_src = _evaluate_prompt_pair_on_cache(
        scorer=scorer,
        eval_cache=eval_cache,
        normal_prompt=best_normal_prompt,
        abnormal_prompt=best_abnormal_prompt,
        args=args,
    )

    default_cross = None
    best_cross = None
    cross_batches = 0
    if cross_cache_by_category:
        default_cross, cross_batches = _evaluate_prompt_pair_on_cache_map_macro(
            scorer=scorer,
            cache_by_category=cross_cache_by_category,
            normal_prompt=default_normal_prompt,
            abnormal_prompt=default_abnormal_prompt,
            args=args,
            objective_override="image_only",
        )
        best_cross, _ = _evaluate_prompt_pair_on_cache_map_macro(
            scorer=scorer,
            cache_by_category=cross_cache_by_category,
            normal_prompt=best_normal_prompt,
            abnormal_prompt=best_abnormal_prompt,
            args=args,
            objective_override="image_only",
        )

    best_src_score = float(best_src.get("score", 0.0))
    default_src_score = float(default_src.get("score", 0.0))
    best_cross_score = float(best_cross.get("score", 0.0)) if best_cross is not None else None
    default_cross_score = float(default_cross.get("score", 0.0)) if default_cross is not None else None
    best_src_std = float(best_src.get("score_std", 0.0))
    best_cross_std = float(best_cross.get("score_std", 0.0)) if best_cross is not None else 0.0

    return {
        "scope": "pair",
        "category": category,
        "default_normal_prompt": default_normal_prompt,
        "default_abnormal_prompt": default_abnormal_prompt,
        "best_normal_prompt": best_normal_prompt,
        "best_abnormal_prompt": best_abnormal_prompt,
        "default_src": default_src_score,
        "best_src": best_src_score,
        "default_cross": default_cross_score,
        "best_cross": best_cross_score,
        "gain_src": best_src_score - default_src_score,
        "gain_cross": (
            best_cross_score - default_cross_score
            if best_cross_score is not None and default_cross_score is not None
            else None
        ),
        "default_src_std": float(default_src.get("score_std", 0.0)),
        "best_src_std": best_src_std,
        "default_cross_std": float(default_cross.get("score_std", 0.0)) if default_cross is not None else None,
        "best_cross_std": float(best_cross.get("score_std", 0.0)) if best_cross is not None else None,
        "score_std": max(best_src_std, best_cross_std),
        "cross_batches": int(cross_batches),
    }


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _safe_sigmoid(x: float) -> float:
    x = float(np.clip(x, -60.0, 60.0))
    return float(1.0 / (1.0 + np.exp(-x)))


def _compute_alpha_diversity_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {
            "count": 0.0,
            "unique_count": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "spread": 0.0,
        }
    arr = np.asarray(values, dtype=np.float64)
    unique_count = len(np.unique(np.round(arr, 6)))
    return {
        "count": float(arr.size),
        "unique_count": float(unique_count),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "spread": float(np.max(arr) - np.min(arr)),
    }


def _is_alpha_collapsed(diversity_stats: Dict[str, float], args) -> bool:
    if diversity_stats.get("count", 0.0) <= 1.0:
        return True
    min_unique = int(getattr(args, "adaptive_ccto_collapse_min_unique", 2))
    min_std = float(getattr(args, "adaptive_ccto_collapse_min_std", 0.01))
    min_spread = float(getattr(args, "adaptive_ccto_collapse_min_spread", 0.05))
    return (
        diversity_stats.get("unique_count", 0.0) < float(min_unique)
        or diversity_stats.get("std", 0.0) < min_std
        or diversity_stats.get("spread", 0.0) < min_spread
    )


def _remap_alpha_plan_by_reliability(alpha_plan: Dict[str, Dict[str, Any]], args) -> None:
    if not alpha_plan:
        return
    min_alpha = float(getattr(args, "ccto_alpha_min", 0.3))
    max_alpha = float(getattr(args, "ccto_alpha_max", 0.9))
    if len(alpha_plan) == 1 or abs(max_alpha - min_alpha) < 1e-12:
        for _, trace in alpha_plan.items():
            trace["effective_ccto_alpha"] = float(_clamp(float(trace["effective_ccto_alpha"]), min_alpha, max_alpha))
            trace["decision_reason"] = "collapse_remap_degenerate"
            trace["collapse_mode_applied"] = "remap"
        return

    ranked = sorted(
        alpha_plan.items(),
        key=lambda kv: (-float(kv[1].get("signals", {}).get("reliability", 0.0)), kv[0]),
    )
    n = len(ranked)
    for rank, (_, trace) in enumerate(ranked):
        frac = float(rank) / float(n - 1)
        remapped = min_alpha + frac * (max_alpha - min_alpha)
        trace["raw_alpha_pre_remap"] = float(trace.get("effective_ccto_alpha", remapped))
        trace["effective_ccto_alpha"] = float(_clamp(remapped, min_alpha, max_alpha))
        trace["decision_reason"] = "collapse_remap_rank"
        trace["collapse_mode_applied"] = "remap"
        trace["remap_rank"] = int(rank)
        trace["remap_total"] = int(n)


def _compute_adaptive_ccto_alpha(
    scorer,
    eval_cache: List[Dict[str, Any]],
    cross_cache_by_category: Dict[str, List[Dict[str, Any]]],
    category: str,
    args,
) -> Dict[str, Any]:
    """Compute per-category adaptive CCTO alpha and return decision trace."""
    policy = str(getattr(args, "adaptive_ccto_policy", "continuous_v2"))
    base_alpha = float(getattr(args, "ccto_alpha", 0.6))
    min_alpha = float(getattr(args, "ccto_alpha_min", 0.3))
    max_alpha = float(getattr(args, "ccto_alpha_max", 0.9))
    std_threshold = float(getattr(args, "ccto_std_threshold", 0.5))

    default_normal = f"X normal {category}"
    default_abnormal = f"X abnormal {category}"
    src_metrics = _evaluate_prompt_pair_on_cache(
        scorer=scorer,
        eval_cache=eval_cache,
        normal_prompt=default_normal,
        abnormal_prompt=default_abnormal,
        args=args,
        objective_override="image_only",
    )

    if not cross_cache_by_category:
        alpha = float(_clamp(base_alpha, min_alpha, max_alpha))
        return {
            "policy": policy,
            "effective_ccto_alpha": alpha,
            "raw_alpha": alpha,
            "decision_reason": "no_cross_cache_fallback_base",
            "signals": {
                "src_score_image": float(src_metrics.get("score", 0.0)),
                "src_std": float(src_metrics.get("score_std", 0.0)),
                "cross_score_image": 0.0,
                "cross_std": 0.0,
                "cross_src_ratio": 0.0,
                "cross_minus_src": 0.0,
                "reliability_raw": 0.0,
                "reliability_clipped": 0.0,
                "reliability": 0.0,
            },
        }

    cross_metrics, _ = _evaluate_prompt_pair_on_cache_map_macro(
        scorer=scorer,
        cache_by_category=cross_cache_by_category,
        normal_prompt=default_normal,
        abnormal_prompt=default_abnormal,
        args=args,
        objective_override="image_only",
    )

    src_score = float(src_metrics.get("score", 0.0))
    src_std = float(src_metrics.get("score_std", 0.0))
    cross_score = float(cross_metrics.get("score", 0.0))
    cross_std = float(cross_metrics.get("score_std", 0.0))
    cross_src_ratio = cross_score / max(abs(src_score), 1e-8)
    cross_minus_src = cross_score - src_score
    reliability_eps = float(getattr(args, "adaptive_ccto_reliability_eps", 1e-6))
    reliability_clip = float(getattr(args, "adaptive_ccto_reliability_clip", 2.5))
    std_penalty = float(getattr(args, "adaptive_ccto_std_penalty", 1.0))
    reliability_denom = max(abs(cross_score) + abs(src_score), reliability_eps)
    reliability_raw = (cross_minus_src / reliability_denom) - (
        std_penalty * (cross_std / max(std_threshold, reliability_eps))
    )
    reliability_clipped = float(
        np.clip(reliability_raw, -reliability_clip, reliability_clip)
    )
    reliability = reliability_clipped

    if policy == "legacy_threshold":
        if cross_std > std_threshold:
            raw_alpha = max_alpha
            reason = "legacy_high_std_to_max"
        elif src_score > 0 and cross_score > src_score * 0.8 and cross_std < std_threshold * 0.5:
            raw_alpha = min_alpha
            reason = "legacy_reliable_cross_to_min"
        else:
            raw_alpha = base_alpha
            reason = "legacy_base_alpha"
    else:
        temperature = float(getattr(args, "adaptive_ccto_temperature", 4.0))
        cross_confidence = _safe_sigmoid(temperature * reliability)
        raw_alpha = max_alpha - (max_alpha - min_alpha) * cross_confidence
        reason = "continuous_v2_sigmoid"

    effective_alpha = float(_clamp(raw_alpha, min_alpha, max_alpha))
    return {
        "policy": policy,
        "effective_ccto_alpha": effective_alpha,
        "raw_alpha": float(raw_alpha),
        "decision_reason": reason,
        "signals": {
            "src_score_image": src_score,
            "src_std": src_std,
            "cross_score_image": cross_score,
            "cross_std": cross_std,
            "cross_src_ratio": cross_src_ratio,
            "cross_minus_src": cross_minus_src,
            "reliability_denom": reliability_denom,
            "reliability_raw": reliability_raw,
            "reliability_clipped": reliability_clipped,
            "reliability": reliability,
        },
    }


def _build_adaptive_ccto_alpha_plan(
    make_dataset,
    scorer,
    args,
    categories: List[str],
    ccto_per_cat_cache: Dict[str, List[Dict[str, Any]]],
    device: torch.device,
    logger,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, float]]:
    alpha_plan: Dict[str, Dict[str, Any]] = {}
    if not categories or not ccto_per_cat_cache:
        return alpha_plan, _compute_alpha_diversity_stats([])

    eval_batches = int(getattr(args, "adaptive_ccto_eval_batches", 0))
    if eval_batches <= 0:
        eval_batches = max(1, int(getattr(args, "evo_val_batches", 10)))
    logger.info(
        "Adaptive CCTO precheck: policy=%s eval_batches=%d categories=%d",
        str(getattr(args, "adaptive_ccto_policy", "continuous_v2")),
        eval_batches,
        len(categories),
    )

    for category in categories:
        cache = _build_eval_cache(
            make_dataset=make_dataset,
            scorer=scorer,
            args=args,
            category=category,
            device=device,
            max_batches=eval_batches,
        )
        if len(cache) == 0:
            logger.warning("Adaptive CCTO precheck skipped empty category cache: %s", category)
            continue
        active_cross = {
            c: cc for c, cc in ccto_per_cat_cache.items()
            if c != category
        }
        trace = _compute_adaptive_ccto_alpha(
            scorer=scorer,
            eval_cache=cache,
            cross_cache_by_category=active_cross,
            category=category,
            args=args,
        )
        alpha_plan[category] = trace
        _offload_eval_cache(cache)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    before = _compute_alpha_diversity_stats(
        [float(v.get("effective_ccto_alpha", 0.0)) for v in alpha_plan.values()]
    )
    collapsed = _is_alpha_collapsed(before, args)
    mode = str(getattr(args, "adaptive_ccto_collapse_mode", "warn"))
    logger.info(
        "ADAPTIVE_CCTO_SUMMARY|phase=precheck_before|count=%d|unique=%d|std=%.6f|min=%.6f|max=%.6f|spread=%.6f|collapsed=%d|mode=%s",
        int(before.get("count", 0.0)),
        int(before.get("unique_count", 0.0)),
        float(before.get("std", 0.0)),
        float(before.get("min", 0.0)),
        float(before.get("max", 0.0)),
        float(before.get("spread", 0.0)),
        1 if collapsed else 0,
        mode,
    )

    if collapsed:
        if mode == "strict":
            raise RuntimeError(
                "Adaptive CCTO alpha collapsed in strict mode: "
                f"unique={int(before.get('unique_count', 0.0))}, "
                f"std={before.get('std', 0.0):.6f}, spread={before.get('spread', 0.0):.6f}"
            )
        if mode == "remap":
            _remap_alpha_plan_by_reliability(alpha_plan, args)
        else:
            logger.warning(
                "Adaptive CCTO alpha collapse detected (warn mode): "
                "unique=%d std=%.6f spread=%.6f",
                int(before.get("unique_count", 0.0)),
                float(before.get("std", 0.0)),
                float(before.get("spread", 0.0)),
            )

    after = _compute_alpha_diversity_stats(
        [float(v.get("effective_ccto_alpha", 0.0)) for v in alpha_plan.values()]
    )
    logger.info(
        "ADAPTIVE_CCTO_SUMMARY|phase=precheck_after|count=%d|unique=%d|std=%.6f|min=%.6f|max=%.6f|spread=%.6f|collapsed=%d|mode=%s",
        int(after.get("count", 0.0)),
        int(after.get("unique_count", 0.0)),
        float(after.get("std", 0.0)),
        float(after.get("min", 0.0)),
        float(after.get("max", 0.0)),
        float(after.get("spread", 0.0)),
        1 if _is_alpha_collapsed(after, args) else 0,
        mode,
    )
    return alpha_plan, after


def _rerank_candidates_on_cache(
    scorer,
    eval_cache: List[Dict[str, Any]],
    candidates: List[str],
    role: str,
    baseline: str,
    args,
) -> List[float]:
    if not candidates:
        return []

    batch_outputs_by_candidate = _collect_candidate_outputs_on_cache(
        scorer=scorer,
        optimizer=None,
        eval_cache=eval_cache,
        candidates=candidates,
        role=role,
        baseline=baseline,
        args=args,
        use_coevo=False,
    )

    scores = []
    for batch_outputs in batch_outputs_by_candidate:
        metrics = _compute_objective_metrics(batch_outputs, "image_pixel", args)
        scores.append(float(metrics.get("pixel_iou", 0.0)))
    return scores


def _rerank_iou_score(
    scorer,
    eval_cache: List[Dict[str, Any]],
    candidate: str,
    role: str,
    baseline: str,
    args,
) -> float:
    return _rerank_candidates_on_cache(
        scorer=scorer,
        eval_cache=eval_cache,
        candidates=[candidate],
        role=role,
        baseline=baseline,
        args=args,
    )[0]


def _build_eval_cache(
    make_dataset,
    scorer,
    args,
    category: str,
    device: torch.device,
    max_batches: Optional[int] = None,
    dataset_override: Optional[str] = None,
) -> List[Dict[str, Any]]:
    ds_name = dataset_override or args.dataset
    dataloader, _ = make_dataset.mask_dataset(
        name=ds_name,
        product_list=[category],
        batchsize=args.batch_size,
        shuf=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    cache: List[Dict[str, Any]] = []
    metric_resolution = int(getattr(args, "stage2_metric_resolution", 256))
    for batch_idx, items in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = items["img"].to(device)
        prepared = scorer.prepare_images(images)
        labels = np.asarray(items["anomaly"]).reshape(-1).astype(np.int32)
        pixel_masks = _resize_metric_mask(items.get("img_mask"), metric_resolution)
        cache.append(
            {
                "prepared": prepared,
                "labels": labels,
                "pixel_masks": pixel_masks,
            }
    )
    # ── label coverage check ──
    if cache:
        all_labels = np.concatenate([b["labels"] for b in cache])
        unique_vals, counts = np.unique(all_labels, return_counts=True)
        label_hist = dict(zip(unique_vals.tolist(), counts.tolist()))
        if len(unique_vals) < 2:
            _module_logger.warning(
                "eval cache %s: single-class labels (histogram=%s), "
                "AUROC will be 0.5 — increase batches or check data ordering",
                category, label_hist,
            )
    return cache


def _label_hist_from_eval_cache(eval_cache: List[Dict[str, Any]]) -> Dict[int, int]:
    if not eval_cache:
        return {}
    labels = [
        np.asarray(batch["labels"]).reshape(-1).astype(np.int32)
        for batch in eval_cache
        if batch.get("labels") is not None and np.asarray(batch["labels"]).size > 0
    ]
    if not labels:
        return {}
    all_labels = np.concatenate(labels)
    unique_vals, counts = np.unique(all_labels, return_counts=True)
    return dict(zip(unique_vals.tolist(), counts.tolist()))


def _select_search_eval_cache(
    full_eval_cache: List[Dict[str, Any]],
    args,
    logger=None,
    category: Optional[str] = None,
) -> List[Dict[str, Any]]:
    if not full_eval_cache:
        return []

    requested_batches = max(1, int(getattr(args, "evo_val_batches", 10)))
    configured_max_batches = int(getattr(args, "evo_search_max_batches", 0) or 0)
    effective_max_batches = len(full_eval_cache) if configured_max_batches <= 0 else configured_max_batches
    effective_max_batches = max(1, min(len(full_eval_cache), effective_max_batches))

    initial_batches = min(requested_batches, effective_max_batches)
    selected_batches = initial_batches
    label_hist = _label_hist_from_eval_cache(full_eval_cache[:selected_batches])

    while len(label_hist) < 2 and selected_batches < effective_max_batches:
        selected_batches += 1
        label_hist = _label_hist_from_eval_cache(full_eval_cache[:selected_batches])

    search_cache = full_eval_cache[:selected_batches]
    if logger is not None:
        if selected_batches > initial_batches:
            logger.info(
                "  search_cache auto-expanded: %s %d->%d batches labels=%s (max=%d)",
                category or "-",
                initial_batches,
                selected_batches,
                label_hist,
                effective_max_batches,
            )
        elif len(label_hist) < 2:
            logger.warning(
                "  search_cache remains single-class: %s batches=%d labels=%s (max=%d)",
                category or "-",
                selected_batches,
                label_hist,
                effective_max_batches,
            )
    return search_cache


def _build_shared_streaming_dataloader(
    make_dataset,
    args,
    category: str,
) -> Any:
    """DataLoader dedicated to the shared bank / shared alpha-search.

    Forced single-process, so pt_data_worker does not inflate CPU RAM on the shared path.
    """
    dataloader, _ = make_dataset.mask_dataset(
        name=args.dataset,
        product_list=[category],
        batchsize=args.batch_size,
        shuf=False,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )
    return dataloader


def _build_shared_alpha_eval_cache(
    make_dataset,
    scorer,
    args,
    category: str,
    device: torch.device,
    max_batches: int = 8,
) -> List[Dict[str, Any]]:
    """Build a compact eval cache for the shared alpha-search and offload it to CPU immediately."""
    dataloader = _build_shared_streaming_dataloader(
        make_dataset=make_dataset,
        args=args,
        category=category,
    )
    cache: List[Dict[str, Any]] = []
    metric_resolution = int(getattr(args, "stage2_metric_resolution", 256))
    for batch_idx, items in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = items["img"].to(device)
        with torch.no_grad():
            prepared = scorer.prepare_images(images)
        labels = np.asarray(items["anomaly"]).reshape(-1).astype(np.int32)
        pixel_masks = _resize_metric_mask(items.get("img_mask"), metric_resolution)
        cache.append(
            {
                "prepared": _to_cpu_recursive(prepared),
                "labels": labels,
                "pixel_masks": pixel_masks,
            }
        )
        del prepared
        del images
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return cache


def _count_shared_alpha_cached_batches(
    make_dataset,
    args,
    categories: List[str],
    max_batches: int = 8,
) -> int:
    """Estimate the total number of cached batches for the shared alpha-search, without triggering feature extraction."""
    total_batches = 0
    for category in categories:
        dataloader = _build_shared_streaming_dataloader(
            make_dataset=make_dataset,
            args=args,
            category=category,
        )
        total_batches += min(len(dataloader), max_batches)
        del dataloader
    return total_batches


# ---------------------------------------------------------------------------
# eval_cache GPU memory management
# ---------------------------------------------------------------------------

def _to_cpu_recursive(obj):
    """Recursively move tensors / lists / tuples to CPU."""
    if isinstance(obj, torch.Tensor):
        return obj.cpu()
    if isinstance(obj, list):
        return [_to_cpu_recursive(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(_to_cpu_recursive(x) for x in obj)
    return obj


def _to_device_recursive(obj, device: torch.device):
    """Recursively move tensors / lists / tuples to the given device."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, list):
        return [_to_device_recursive(x, device) for x in obj]
    if isinstance(obj, tuple):
        return tuple(_to_device_recursive(x, device) for x in obj)
    return obj


def _offload_eval_cache(eval_cache: List[Dict[str, Any]]) -> None:
    """Move prepared GPU tensors in eval_cache to CPU to free GPU memory."""
    for entry in eval_cache:
        prepared = entry.get("prepared")
        if prepared is not None:
            entry["prepared"] = _to_cpu_recursive(prepared)


def _reload_eval_cache(eval_cache: List[Dict[str, Any]], device: torch.device) -> None:
    """Move tensors in eval_cache back to GPU."""
    for entry in eval_cache:
        prepared = entry.get("prepared")
        if prepared is not None:
            entry["prepared"] = _to_device_recursive(prepared, device)


# ---------------------------------------------------------------------------
# Optimizer Builder
# ---------------------------------------------------------------------------

def build_optimizer(args, logger):
    """Build the Stage2 prompt optimizer."""
    use_coevo = getattr(args, "use_coevo_prompt", False)

    if use_coevo:
        from models.coevo_evoprompt import build_coevo_optimizer
        _game_flag = getattr(args, "game_metrics_enable", None)
        if _game_flag is None:
            _game_flag = bool(getattr(args, "asym_b_enable", False))
        _llm_enabled = getattr(args, "llm_mutation_enabled", False)
        _llm_model_id = getattr(args, "llm_mutation_model_id", "")
        _llm_max_tokens = getattr(args, "llm_mutation_max_tokens", 32)
        _mutation_ops = getattr(args, "evo_mutation_ops", "")
        optimizer = build_coevo_optimizer(
            population_size=args.evo_population,
            generations=args.evo_generations,
            topk=args.evo_topk,
            lambda_diversity=args.evo_lambda_diversity,
            coevo_pair_k=args.coevo_pair_k,
            coevo_alpha_auroc=args.coevo_alpha_auroc,
            coevo_beta_contrast=args.coevo_beta_contrast,
            coevo_crossover_rate=args.coevo_crossover_rate,
            coevo_final_pair_select=getattr(args, "coevo_final_pair_select", "marginal"),
            record_population_trace=getattr(args, "coevo_record_population_trace", False),
            game_metrics_enable=_game_flag,
            llm_mutation_enabled=_llm_enabled,
            llm_model_id=_llm_model_id,
            llm_mutation_max_tokens=_llm_max_tokens,
            evo_mutation_ops=_mutation_ops,
        )
        logger.info(
            "CoEvo optimizer: K=%d alpha=%.3f beta=%.3f crossover_rate=%.3f final_pair_select=%s population_trace=%s game_metrics=%s llm_mutation=%s mutation_ops=%s",
            args.coevo_pair_k,
            args.coevo_alpha_auroc,
            args.coevo_beta_contrast,
            args.coevo_crossover_rate,
            getattr(args, "coevo_final_pair_select", "marginal"),
            getattr(args, "coevo_record_population_trace", False),
            _game_flag,
            _llm_enabled,
            ",".join(getattr(optimizer, "evo_mutation_ops", [])),
        )
        return optimizer

    if getattr(args, "game_metrics_enable", False):
        logger.warning(
            "--game_metrics_enable is set but --use_coevo_prompt is not; "
            "GAME_METRICS will NOT be collected (requires CoEvo mode)."
        )
    optimizer = EvoPromptOptimizer(
        population_size=args.evo_population,
        generations=args.evo_generations,
        topk=args.evo_topk,
        lambda_diversity=args.evo_lambda_diversity,
        llm_mutation_enabled=getattr(args, "llm_mutation_enabled", False),
        llm_model_id=getattr(args, "llm_mutation_model_id", ""),
        llm_mutation_max_tokens=getattr(args, "llm_mutation_max_tokens", 32),
        evo_crossover_rate=getattr(args, "evo_crossover_rate", 0.0),
        evo_random_search=getattr(args, "evo_random_search", False),
        evo_random_search_seed=getattr(args, "seed", None),
        evo_mutation_ops=getattr(args, "evo_mutation_ops", ""),
    )
    logger.info(
        "Standard EvoPrompt optimizer (llm_mutation=%s, random_search=%s, crossover_rate=%.3f, mutation_ops=%s)",
        getattr(args, "llm_mutation_enabled", False),
        getattr(args, "evo_random_search", False),
        float(getattr(args, "evo_crossover_rate", 0.0)),
        ",".join(getattr(optimizer, "evo_mutation_ops", [])),
    )
    return optimizer


# ---------------------------------------------------------------------------
# Scoring Callback - combined objective (image + pixel)
# ---------------------------------------------------------------------------

def build_scoring_callback(
    scorer,
    optimizer,
    eval_cache,
    base_prompt,
    args,
    role_type,
    use_coevo,
    target_cache=None,
    cdace_alpha=1.0,
    target_objective="image_only",
    ccto_per_cat_cache=None,
    ccto_scope="abnormal_only",
):
    """Build scoring_callback(candidates, role) -> List[float].

    Three mutually exclusive modes (priority CCTO > CDACE > source-only):
    1. CCTO: ccto_per_cat_cache non-empty -> per-category evaluation + macro-average
       fitness = α * source_score + (1-α) * mean(per_cat_scores)
    2. CDACE: target_cache non-empty -> pooled evaluation
       fitness = α * source_score + (1-α) * target_score
    3. source-only: source_score alone

    Asymmetric objective: the source side follows args.stage2_objective (including pixel), while the
    cross-category / target side is forced to target_objective (image_only by default), using
    image-level signal only so poor pixel masks cannot pollute the search direction.
    """

    _asym_b_enable = bool(getattr(args, "asym_b_enable", False))
    _use_ccto = (
        ccto_per_cat_cache is not None
        and len(ccto_per_cat_cache) > 0
        and (cdace_alpha < 1.0 or _asym_b_enable)
    )
    _use_cdace = target_cache is not None and cdace_alpha < 1.0 and not _use_ccto
    _enable_default_regret = (
        _use_ccto
        and bool(getattr(args, "enable_default_regret_ccto", True))
        and not _asym_b_enable
    )
    _regret_src_weight = float(getattr(args, "safe_regret_src_weight", 1.0))
    _regret_cross_weight = float(getattr(args, "safe_regret_cross_weight", 1.0))
    _regret_std_weight = float(getattr(args, "safe_regret_std_weight", 0.0))
    _ccto_scope = ccto_scope
    _default_metrics_cache: Dict[str, Dict[str, Optional[Dict[str, float]]]] = {}

    def _validate_asym_cross_cache(effective_role: Optional[str]) -> None:
        if not _asym_b_enable:
            return
        if not _use_ccto or ccto_per_cat_cache is None or len(ccto_per_cat_cache) == 0:
            raise ValueError(
                "asym_b_enable requires non-empty cross-category CCTO cache. "
                "Current cache map is empty."
            )
        empty_categories = [
            str(cat_name)
            for cat_name, cat_cache in ccto_per_cat_cache.items()
            if not cat_cache
        ]
        if empty_categories:
            preview = ",".join(sorted(empty_categories)[:8])
            if len(empty_categories) > 8:
                preview += ",..."
            raise ValueError(
                "asym_b_enable requires non-empty cross-category CCTO cache per category. "
                f"role={effective_role}, empty_categories={preview}"
            )

    def _get_default_metrics(effective_role: Optional[str]) -> Dict[str, Optional[Dict[str, float]]]:
        role_key = effective_role or "shared"
        if role_key in _default_metrics_cache:
            return _default_metrics_cache[role_key]

        default_prompt = _default_prompt_for_role(base_prompt, effective_role)
        src_default = _evaluate_candidate_on_cache(
            scorer=scorer,
            optimizer=optimizer,
            eval_cache=eval_cache,
            candidate=default_prompt,
            role=effective_role,
            baseline=base_prompt,
            args=args,
            use_coevo=use_coevo,
        )

        cross_default = None
        if _use_ccto and (_ccto_scope == "symmetric" or effective_role == "abnormal"):
            cross_default_metrics, _ = _evaluate_candidates_on_cache_map_macro_metrics(
                scorer=scorer,
                optimizer=optimizer,
                cache_by_category=ccto_per_cat_cache,
                candidates=[default_prompt],
                role=effective_role,
                baseline=base_prompt,
                args=args,
                use_coevo=use_coevo,
                objective_override=target_objective,
            )
            cross_default = cross_default_metrics[0] if cross_default_metrics else None

        _default_metrics_cache[role_key] = {"src": src_default, "cross": cross_default}
        return _default_metrics_cache[role_key]

    def scoring_callback(candidates, role=None):
        effective_role = role if role is not None else role_type
        scores = []
        _cb_src_metrics: List[Dict[str, float]] = []
        _diag_src_scores: List[float] = []
        _diag_cross_scores: List[float] = []
        chunk_size = max(1, int(getattr(args, "candidate_batch_size", 2)))
        apply_ccto = _use_ccto and (_ccto_scope == "symmetric" or effective_role == "abnormal")
        _validate_asym_cross_cache(effective_role)
        if _asym_b_enable and not apply_ccto:
            raise ValueError(
                "asym_b_enable requires cross-category evaluation for the active role. "
                "Ensure CCTO caches are available and effective scope is symmetric."
            )

        progress = tqdm(total=len(candidates), desc=f"Eval ({effective_role})", leave=False)
        try:
            for chunk in _iter_candidate_chunks(candidates, chunk_size):
                chunk_list = list(chunk)
                src_metrics = _evaluate_candidates_on_cache(
                    scorer=scorer,
                    optimizer=optimizer,
                    eval_cache=eval_cache,
                    candidates=chunk_list,
                    role=effective_role,
                    baseline=base_prompt,
                    args=args,
                    use_coevo=use_coevo,
                )
                _cb_src_metrics.extend(src_metrics)

                if apply_ccto:
                    macro_metrics, _ = _evaluate_candidates_on_cache_map_macro_metrics(
                        scorer=scorer,
                        optimizer=optimizer,
                        cache_by_category=ccto_per_cat_cache,
                        candidates=chunk_list,
                        role=effective_role,
                        baseline=base_prompt,
                        args=args,
                        use_coevo=use_coevo,
                        objective_override=target_objective,
                    )
                    default_metrics = _get_default_metrics(effective_role) if _enable_default_regret else None
                    for cand_text, sm, macro_m in zip(chunk_list, src_metrics, macro_metrics):
                        src_s = float(sm["score"])
                        macro_s = float(macro_m["score"])
                        _diag_src_scores.append(src_s)
                        _diag_cross_scores.append(macro_s)
                        if _asym_b_enable:
                            asym_trace = _compute_asym_b_final_score(
                                role=effective_role,
                                src_score=src_s,
                                cross_score=macro_s,
                                args=args,
                            )
                            blended = float(asym_trace["final_score"])
                            _module_logger.info(
                                "ASYM_B_SCORE|role=%s|policy=%s|kappa=%.6f|candidate=%s|src_score=%.6f|cross_score=%.6f|raw_score=%.6f|final_score=%.6f",
                                str(asym_trace.get("role", effective_role)),
                                str(asym_trace.get("policy", _resolve_asym_b_policy(args))),
                                float(asym_trace.get("kappa", 0.0)),
                                json.dumps(cand_text, ensure_ascii=False),
                                float(asym_trace["src_score"]),
                                float(asym_trace["cross_score"]),
                                float(asym_trace["raw_score"]),
                                float(asym_trace["final_score"]),
                            )
                        else:
                            blended = cdace_alpha * src_s + (1 - cdace_alpha) * macro_s
                        if (not _asym_b_enable) and default_metrics is not None:
                            default_src = float(default_metrics["src"]["score"])
                            default_cross = float(default_metrics["cross"]["score"]) if default_metrics["cross"] is not None else 0.0
                            regret_src = max(0.0, default_src - src_s)
                            regret_cross = max(0.0, default_cross - macro_s)
                            std_term = max(float(sm.get("score_std", 0.0)), float(macro_m.get("score_std", 0.0)))
                            blended = (
                                blended
                                - _regret_src_weight * regret_src
                                - _regret_cross_weight * regret_cross
                                - _regret_std_weight * std_term
                            )
                        scores.append(blended)
                elif _use_cdace:
                    # CDACE: pooled evaluation
                    tgt_metrics = _evaluate_candidates_on_cache(
                        scorer=scorer,
                        optimizer=optimizer,
                        eval_cache=target_cache,
                        candidates=chunk_list,
                        role=effective_role,
                        baseline=base_prompt,
                        args=args,
                        use_coevo=use_coevo,
                        objective_override=target_objective,
                    )
                    for sm, tm in zip(src_metrics, tgt_metrics):
                        src_s = float(sm["score"])
                        tgt_s = float(tm["score"])
                        blended = cdace_alpha * src_s + (1 - cdace_alpha) * tgt_s
                        scores.append(blended)
                else:
                    default_metrics = _get_default_metrics(effective_role) if _enable_default_regret else None
                    for sm in src_metrics:
                        src_s = float(sm["score"])
                        if default_metrics is None:
                            scores.append(src_s)
                            continue
                        default_src = float(default_metrics["src"]["score"])
                        regret_src = max(0.0, default_src - src_s)
                        std_term = float(sm.get("score_std", 0.0))
                        scores.append(
                            src_s
                            - _regret_src_weight * regret_src
                            - _regret_std_weight * std_term
                        )

                progress.update(len(chunk_list))
        finally:
            progress.close()

        # --- CCTO cross-aggregation diagnostics (per-generation) ---
        if apply_ccto and len(scores) > 1:
            try:
                diag = _compute_ccto_diag_stats(
                    src_scores=_diag_src_scores,
                    cross_scores=_diag_cross_scores,
                    blended_scores=scores,
                    evo_topk=int(getattr(args, "evo_topk", 4)),
                )
                if diag is not None:
                    _module_logger.info(
                        "CCTO_DIAG|role=%s|cross_agg=%s|n=%d|cross_spread=%.6f"
                        "|blended_spread=%.6f|rank_corr_src_blended=%.4f|topk_overlap=%.4f",
                        effective_role,
                        f"cvar(k={getattr(args, 'evo_cvar_k', 3)})"
                        if getattr(args, "evo_fitness_agg", "mean") == "cvar"
                        else getattr(args, "ccto_cross_agg", "mean"),
                        int(diag["n"]),
                        float(diag["cross_spread"]),
                        float(diag["blended_spread"]),
                        float(diag["rank_corr_src_blended"]),
                        float(diag["topk_overlap"]),
                    )
                    if getattr(args, "asym_b_enable", False) and len(_diag_src_scores) > 0:
                        _module_logger.info(
                            "CCTO_DIAG_ASYM|role=%s|src_mean=%.6f|cross_mean=%.6f",
                            effective_role,
                            float(sum(_diag_src_scores) / len(_diag_src_scores)),
                            float(sum(_diag_cross_scores) / max(1, len(_diag_cross_scores))),
                        )
            except Exception as exc:
                _module_logger.debug("CCTO_DIAG failed: %s", exc)

        return {"scores": scores, "src_metrics": _cb_src_metrics}

    return scoring_callback


def build_rerank_callback(
    scorer,
    rerank_cache,
    base_prompt,
    args,
):
    """Build the final top-k rerank callback, selecting by mean IoU on the source split."""

    def rerank_callback(candidates, role=None):
        rerank_scores = []
        effective_role = role or "shared"
        chunk_size = max(1, int(getattr(args, "candidate_batch_size", 2)))
        for chunk in _iter_candidate_chunks(candidates, chunk_size):
            rerank_scores.extend(
                _rerank_candidates_on_cache(
                    scorer=scorer,
                    eval_cache=rerank_cache,
                    candidates=list(chunk),
                    role=effective_role,
                    baseline=base_prompt,
                    args=args,
                )
            )
        return rerank_scores

    return rerank_callback


# ---------------------------------------------------------------------------
# Shared bank builder (cross-class mean AUROC + MMR)
# ---------------------------------------------------------------------------

def _mmr_select_candidates(candidates, k: int, lambda_diversity: float = 0.2):
    if len(candidates) <= k:
        return candidates
    selected = []
    remaining = sorted(candidates, key=lambda x: x["score"], reverse=True)
    selected.append(remaining.pop(0))

    while len(selected) < k and remaining:
        best_idx = 0
        best_mmr = -float("inf")
        for i, cand in enumerate(remaining):
            feat = cand.get("feat")
            if feat is None:
                mmr = cand["score"]
            else:
                max_sim = -1.0
                for sel in selected:
                    if sel.get("feat") is None:
                        continue
                    sim = torch.nn.functional.cosine_similarity(
                        feat.unsqueeze(0), sel["feat"].unsqueeze(0), dim=1
                    ).item()
                    max_sim = max(max_sim, sim)
                if max_sim < 0:
                    max_sim = 0.0
                mmr = (1.0 - lambda_diversity) * cand["score"] + lambda_diversity * (1.0 - max_sim)
            if mmr > best_mmr:
                best_mmr = mmr
                best_idx = i
        selected.append(remaining.pop(best_idx))
    return selected


def _build_shared_bank(
    scorer,
    evo_optimizer,
    make_dataset,
    obj_list,
    args,
    logger,
    device,
):
    start_t = time.time()
    bank_size = int(getattr(args, "shared_bank_size", 4))
    max_candidates = int(getattr(args, "shared_bank_max_candidates_per_role", 48))
    eval_batches = int(getattr(args, "shared_bank_eval_batches", 2))
    lambda_div = float(getattr(args, "evo_lambda_diversity", 0.2))
    if bank_size <= 0:
        return {"meta": {"enabled": False}, "normal": [], "abnormal": [], "pairs": []}

    role_candidates = {"normal": [], "abnormal": []}
    for key, prompt in evo_optimizer.cache.items():
        if not (isinstance(key, tuple) and len(key) == 2):
            continue
        role, name = key
        if role in role_candidates:
            role_candidates[role].append((str(name), str(prompt)))

    def _dedup(items):
        uniq = []
        seen = set()
        for name, prompt in sorted(items, key=lambda x: (x[0], x[1])):
            if prompt in seen:
                continue
            seen.add(prompt)
            uniq.append((name, prompt))
        return uniq[:max_candidates]

    normal_pool = _dedup(role_candidates["normal"])
    abnormal_pool = _dedup(role_candidates["abnormal"])
    logger.info(
        "Shared bank build: normal_pool=%d abnormal_pool=%d eval_batches=%d",
        len(normal_pool), len(abnormal_pool), eval_batches,
    )

    eval_cache_by_cls: Dict[str, List[Dict[str, Any]]] = {}

    def _get_cls_eval_cache(cls_name: str) -> List[Dict[str, Any]]:
        if cls_name not in eval_cache_by_cls:
            eval_cache_by_cls[cls_name] = _build_eval_cache(
                make_dataset=make_dataset,
                scorer=scorer,
                args=args,
                category=cls_name,
                device=device,
                max_batches=eval_batches,
            )
        return eval_cache_by_cls[cls_name]

    def _score_pool(pool, role):
        scored = []
        chunk_size = max(1, int(getattr(args, "candidate_batch_size", 2)))
        for end_idx, chunk in enumerate(_iter_candidate_chunks(pool, chunk_size), start=1):
            chunk_names = [name for name, _prompt in chunk]
            chunk_prompts = [prompt for _name, prompt in chunk]
            cls_scores_by_prompt = [[] for _ in chunk_prompts]
            for cls_name in obj_list:
                cls_cache = _get_cls_eval_cache(cls_name)
                if len(cls_cache) == 0:
                    for cls_scores in cls_scores_by_prompt:
                        cls_scores.append(0.0)
                    continue
                metrics_list = _evaluate_candidates_on_cache(
                    scorer=scorer,
                    optimizer=None,
                    eval_cache=cls_cache,
                    candidates=chunk_prompts,
                    role=role,
                    baseline=f"X {cls_name}",
                    args=args,
                    use_coevo=False,
                )
                for idx, metrics in enumerate(metrics_list):
                    cls_scores_by_prompt[idx].append(float(metrics["score"]))

            for idx, prompt in enumerate(chunk_prompts):
                feat = scorer.get_text_embedding(prompt, role)
                if isinstance(feat, torch.Tensor):
                    feat = feat.detach().float().cpu()
                elif feat is not None:
                    feat = torch.as_tensor(feat, dtype=torch.float32).flatten().cpu()
                else:
                    feat = None
                avg_score = (
                    float(np.mean(cls_scores_by_prompt[idx]))
                    if cls_scores_by_prompt[idx] else 0.5
                )
                scored.append(
                    {
                        "name": chunk_names[idx],
                        "prompt": prompt,
                        "score": avg_score,
                        "feat": feat,
                    }
                )
            processed = min(end_idx * chunk_size, len(pool))
            if processed % max(1, len(pool) // 4) == 0 or processed == len(pool):
                logger.info(
                    "Shared bank eval [%s] %d/%d avg_score=%.4f",
                    role, processed, len(pool), scored[-1]["score"],
                )
        return scored

    normal_scored = _score_pool(normal_pool, "normal")
    abnormal_scored = _score_pool(abnormal_pool, "abnormal")

    sel_n = _mmr_select_candidates(normal_scored, k=bank_size, lambda_diversity=lambda_div)
    sel_a = _mmr_select_candidates(abnormal_scored, k=bank_size, lambda_diversity=lambda_div)
    pair_cnt = min(len(sel_n), len(sel_a), bank_size)
    pairs = []
    for i in range(pair_cnt):
        n_item = sel_n[i]
        a_item = sel_a[i]
        pairs.append(
            {
                "id": i,
                "name": n_item.get("name", ""),
                "normal_prompt": n_item["prompt"],
                "abnormal_prompt": a_item["prompt"],
                "normal_score": float(n_item["score"]),
                "abnormal_score": float(a_item["score"]),
                "pair_score": float(0.5 * (n_item["score"] + a_item["score"])),
            }
        )

    elapsed = time.time() - start_t
    logger.info("Shared bank build finished in %.1fs (pairs=%d)", elapsed, len(pairs))
    return {
        "meta": {
            "bank_size": bank_size,
            "max_candidates_per_role": max_candidates,
            "eval_batches": eval_batches,
            "lambda_diversity": lambda_div,
            "elapsed_sec": elapsed,
            "source_cache_size": len(evo_optimizer.cache),
        },
        "normal": [{"name": x["name"], "prompt": x["prompt"], "score": float(x["score"])} for x in sel_n],
        "abnormal": [{"name": x["name"], "prompt": x["prompt"], "score": float(x["score"])} for x in sel_a],
        "pairs": pairs,
    }


# ---------------------------------------------------------------------------
# Main optimization loop
# ---------------------------------------------------------------------------

def run_stage2_universal(args):
    os.makedirs(args.save_path, exist_ok=True)
    logger = setup_logger(args.save_path, "result_universal_stage2.txt", "optimize_universal")
    _enforce_transfer_mainline(args, logger)

    _load_model_config(args)

    device = torch.device(
        f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu"
    )
    if torch.cuda.is_available():
        torch.cuda.set_device(args.device_id)
    reset_cuda_peak_memory(args.device_id)

    for arg in vars(args):
        logger.info("%s: %s", arg, getattr(args, arg))

    logger.info("=" * 60)
    logger.info("Universal Stage 2 optimization")
    logger.info("=" * 60)

    # build the scorer
    scorer = build_scorer(args, device)
    logger.info("Scorer: %s", type(scorer).__name__)

    # build the optimizer
    evo_optimizer = build_optimizer(args, logger)

    # prepare evaluation data
    preprocess = _transform_test(args.image_size)
    stage2_split = getattr(args, "stage2_split", "test")
    if stage2_split not in ("train", "val", "test"):
        logger.warning("Unknown stage2_split '%s', falling back to 'test'", stage2_split)
        stage2_split = "test"
    allow_fallback = getattr(args, "allow_split_fallback", False)
    make_dataset = Makedataset(
        train_data_path=args.train_data_path,
        preprocess_test=preprocess,
        mode=stage2_split,
        image_size=args.image_size,
        allow_split_fallback=allow_fallback,
    )
    _, all_obj_list = make_dataset.mask_dataset(
        name=args.dataset,
        batchsize=args.batch_size,
        product_list=None,
        shuf=False,
        num_workers=0,
        pin_memory=False,
    )
    obj_list = _filter_stage2_categories(
        all_obj_list,
        getattr(args, "stage2_categories", None),
    )
    if getattr(args, "stage2_categories", None) and not obj_list:
        raise ValueError(
            f"--stage2_categories did not match any dataset categories: {getattr(args, 'stage2_categories')}"
        )

    logger.info("Stage2 eval split: %s (source-supervised: labels+masks)", stage2_split)
    logger.info("Categories: %s", all_obj_list)
    if obj_list != all_obj_list:
        logger.info("Stage2 target categories: %s", obj_list)
    logger.info(
        "GA params: population=%d generations=%d topk=%d",
        args.evo_population, args.evo_generations, args.evo_topk,
    )

    # per-category optimization loop
    use_coevo = getattr(args, "use_coevo_prompt", False)
    optimized_rules = {"normal": {}, "abnormal": {}, "shared": {}}
    partial_state = _load_stage2_partial_state(args.save_path)
    if partial_state["cache"] or any(partial_state["optimized_rules"].values()):
        _merge_stage2_recovered_state(optimized_rules, evo_optimizer, partial_state)
        logger.info(
            "Recovered Stage2 partial state from save_path=%s (completed=%d)",
            args.save_path,
            len(_completed_stage2_categories(optimized_rules)),
        )

    resume_log = str(getattr(args, "resume_stage2_log", "") or "").strip()
    if resume_log:
        with open(resume_log, "r", encoding="utf-8", errors="ignore") as f:
            recovered_from_log = _parse_stage2_resume_log_text(f.read())
        _merge_stage2_recovered_state(
            optimized_rules,
            evo_optimizer,
            recovered_from_log,
            overwrite=False,
        )
        logger.info(
            "Recovered Stage2 categories from resume log=%s: %s",
            resume_log,
            recovered_from_log.get("completed_categories", []),
        )

    completed_categories = set(_completed_stage2_categories(optimized_rules))
    if completed_categories:
        logger.info("Stage2 recovered completed categories: %s", sorted(completed_categories))
    obj_list = [cat for cat in obj_list if cat not in completed_categories]
    if len(obj_list) == 0:
        logger.info("No pending Stage2 target categories remain after recovery; final artifacts will be rebuilt only.")

    # QD archive setup
    _use_qd = bool(getattr(args, "use_qd_archive", False))
    _qd_bins = int(getattr(args, "qd_bins", 5))
    _qd_bd_names = [s.strip() for s in getattr(args, "qd_bd", "image_auroc,pixel_f1").split(",")]
    _qd_archives: Dict[str, Any] = {}
    if _use_qd:
        from models.qd_archive import QDArchive
        logger.info("QD archive enabled: bins=%d bd=%s", _qd_bins, _qd_bd_names)


    # -- CDACE: build the target-domain eval cache --
    cdace_target_cache = None
    _cdace_target_ds = getattr(args, "cdace_target_dataset", "")
    if _cdace_target_ds:
        _target_cats = getattr(args, "cdace_target_categories", None)
        if not _target_cats:
            _auto_target = {
                "mvtec": ["chewinggum", "cashew", "pipe_fryum", "capsules", "candle"],
                "visa": ["bottle", "hazelnut", "wood", "zipper", "leather"],
            }
            _target_cats = _auto_target.get(args.dataset, [])
        if _target_cats:
            _cdace_batches = int(getattr(args, "cdace_target_batches", 2))
            logger.info("CDACE: building target eval cache (%s, %d classes, %d batches/cls)",
                        _cdace_target_ds, len(_target_cats), _cdace_batches)
            cdace_target_cache = []
            for _tcat in _target_cats:
                _tc = _build_eval_cache(
                    make_dataset=make_dataset,
                    scorer=scorer,
                    args=args,
                    category=_tcat,
                    device=device,
                    max_batches=_cdace_batches,
                    dataset_override=_cdace_target_ds,
                )
                cdace_target_cache.extend(_tc)
            logger.info("CDACE: target cache ready (%d batches total)", len(cdace_target_cache))

    _cdace_alpha = float(getattr(args, "cdace_alpha", 1.0))
    if cdace_target_cache is None:
        _cdace_alpha = 1.0  # without a target-domain cache, fall back to source-only
    _ccto_scope_effective = _resolve_effective_ccto_scope(
        args,
        requested_scope=getattr(args, "ccto_scope", "abnormal_only"),
    )
    if bool(getattr(args, "asym_b_enable", False)):
        _asym_cfg = _asym_b_config_dict(args, _ccto_scope_effective)
        logger.info(
            "ASYM_B_MODE|enabled=1|policy=%s|effective_ccto_scope=%s|lambda_normal_gen=%.6f|lambda_abn_spec=%.6f|kappa_normal=%.6f|kappa_abnormal=%.6f",
            str(_asym_cfg["policy"]),
            _ccto_scope_effective,
            float(_asym_cfg["lambda_normal_gen"]),
            float(_asym_cfg["lambda_abn_spec"]),
            float(_asym_cfg["kappa_normal"]),
            float(_asym_cfg["kappa_abnormal"]),
        )

    # -- CCTO: build the per-category eval cache on the source domain --
    ccto_per_cat_cache: Dict[str, List[Dict[str, Any]]] = {}
    if getattr(args, "ccto", False):
        _ccto_batches = int(getattr(args, "ccto_batches", 2))
        logger.info("CCTO: building source cross-category caches "
                    "(%d categories, %d batches/cat, scope=%s, objective=image_only)",
                    len(all_obj_list), _ccto_batches, _ccto_scope_effective)
        for _cat in all_obj_list:
            _cc = _build_eval_cache(
                make_dataset=make_dataset,
                scorer=scorer,
                args=args,
                category=_cat,
                device=device,
                max_batches=_ccto_batches,
            )
            _offload_eval_cache(_cc)
            ccto_per_cat_cache[_cat] = _cc
            _lbl_hist = {}
            if _cc:
                _all_lbl = np.concatenate([b["labels"] for b in _cc])
                _uv, _uc = np.unique(_all_lbl, return_counts=True)
                _lbl_hist = dict(zip(_uv.tolist(), _uc.tolist()))
            logger.info("  CCTO cache: %s -> %d batches, labels=%s", _cat, len(_cc), _lbl_hist)
        logger.info("CCTO: all cross-category caches ready (%d categories)", len(ccto_per_cat_cache))

    adaptive_alpha_plan: Dict[str, Dict[str, Any]] = {}
    adaptive_alpha_summary: Dict[str, float] = {}
    adaptive_alpha_used: List[float] = []
    if getattr(args, "adaptive_ccto_alpha", False) and ccto_per_cat_cache:
        adaptive_alpha_plan, adaptive_alpha_summary = _build_adaptive_ccto_alpha_plan(
            make_dataset=make_dataset,
            scorer=scorer,
            args=args,
            categories=list(obj_list),
            ccto_per_cat_cache=ccto_per_cat_cache,
            device=device,
            logger=logger,
        )

    _completed_categories: set = set()
    if getattr(args, "resume", False) and args.save_path:
        _ckpt_rules_path = os.path.join(args.save_path, "checkpoint_rules.json")
        _ckpt_cache_path = os.path.join(args.save_path, "checkpoint_evo_cache.json")
        if os.path.exists(_ckpt_rules_path) and os.path.exists(_ckpt_cache_path):
            with open(_ckpt_rules_path, "r", encoding="utf-8") as _f:
                _loaded_rules = json.load(_f)
            for _rk in ("normal", "abnormal", "shared"):
                if _rk in _loaded_rules:
                    optimized_rules[_rk].update(_loaded_rules[_rk])
                    _completed_categories.update(_loaded_rules[_rk].keys())
            evo_optimizer.load_optimized_rules(_ckpt_cache_path)
            logger.info("RESUME: loaded checkpoint — %d categories done: %s",
                        len(_completed_categories), sorted(_completed_categories))
        else:
            logger.warning("RESUME: no checkpoint found in %s, starting fresh", args.save_path)

    for idx, category in enumerate(obj_list):
        logger.info("[%d/%d] Optimize: %s", idx + 1, len(obj_list), category)
        if category in _completed_categories:
            logger.info("  [SKIP] already completed (resume mode)")
            continue
        base_prompt = f"X {category}"

        # clear the baseline embedding cache when switching category
        if hasattr(scorer, "clear_baseline_cache"):
            scorer.clear_baseline_cache()

        # During the actual search, build the full eval cache lazily per category; do not reuse the shared-bank build cache
        full_eval_cache = _build_eval_cache(
            make_dataset=make_dataset,
            scorer=scorer,
            args=args,
            category=category,
            device=device,
            max_batches=None,
        )
        if len(full_eval_cache) == 0:
            logger.warning("Skip category '%s': empty eval cache", category)
            continue
        search_cache = _select_search_eval_cache(
            full_eval_cache,
            args,
            logger=logger,
            category=category,
        )
        rerank_cache = full_eval_cache
        logger.info(
            "  eval_cache: search_batches=%d rerank_batches=%d objective=%s",
            len(search_cache),
            len(rerank_cache),
            getattr(args, "stage2_objective", "image_pixel"),
        )

        # -- CCTO: assemble the per-category cross-cache for the current category (excluding itself) --
        _active_target_cache = cdace_target_cache  # CDACE only
        _active_alpha = _cdace_alpha
        _active_ccto_cache: Optional[Dict[str, List[Dict[str, Any]]]] = None
        _alpha_trace: Optional[Dict[str, Any]] = None
        _ccto_total_batches = 0
        if ccto_per_cat_cache:
            _active_ccto_cache = {
                _other_cat: _other_cache
                for _other_cat, _other_cache in ccto_per_cat_cache.items()
                if _other_cat != category
            }
            if getattr(args, "adaptive_ccto_alpha", False) and _active_ccto_cache:
                _alpha_trace = adaptive_alpha_plan.get(category)
                if _alpha_trace is None:
                    _alpha_trace = _compute_adaptive_ccto_alpha(
                        scorer=scorer,
                        eval_cache=rerank_cache,
                        cross_cache_by_category=_active_ccto_cache,
                        category=category,
                        args=args,
                    )
                _active_alpha = float(_alpha_trace.get("effective_ccto_alpha", getattr(args, "ccto_alpha", 0.6)))
                adaptive_alpha_used.append(_active_alpha)
                logger.info("  Adaptive CCTO alpha for %s: %.3f", category, _active_alpha)
                _signals = _alpha_trace.get("signals", {}) if _alpha_trace else {}
                logger.info(
                    "ADAPTIVE_CCTO_ALPHA|category=%s|alpha=%.6f|raw=%.6f|policy=%s|reason=%s|"
                    "src=%.6f|cross=%.6f|cross_std=%.6f|rel_raw=%.6f|rel_clip=%.6f|reliability=%.6f|ratio=%.6f",
                    category,
                    _active_alpha,
                    float(_alpha_trace.get("raw_alpha", _active_alpha)) if _alpha_trace else _active_alpha,
                    str(_alpha_trace.get("policy", "n/a")) if _alpha_trace else "n/a",
                    str(_alpha_trace.get("decision_reason", "n/a")) if _alpha_trace else "n/a",
                    float(_signals.get("src_score_image", 0.0)),
                    float(_signals.get("cross_score_image", 0.0)),
                    float(_signals.get("cross_std", 0.0)),
                    float(_signals.get("reliability_raw", 0.0)),
                    float(_signals.get("reliability_clipped", 0.0)),
                    float(_signals.get("reliability", 0.0)),
                    float(_signals.get("cross_src_ratio", 0.0)),
                )
            else:
                _active_alpha = float(getattr(args, "ccto_alpha", 0.6))
            _ccto_total_batches = sum(len(_cache) for _cache in _active_ccto_cache.values())
            logger.info(
                "  CCTO: other_categories=%d cross_batches=%d scope=%s objective=image_only (exclude=%s)",
                len(_active_ccto_cache),
                _ccto_total_batches,
                _ccto_scope_effective,
                category,
            )

        scorer_for_search = scorer

        if use_coevo and args.evo_dual_branch:
            # CoEvo joint dual-branch optimization
            callback = build_scoring_callback(
                scorer_for_search, evo_optimizer, search_cache, base_prompt,
                args, role_type="normal", use_coevo=True,
                target_cache=_active_target_cache, cdace_alpha=_active_alpha,
                ccto_per_cat_cache=_active_ccto_cache,
                ccto_scope=_ccto_scope_effective,
            )
            rerank_cb = build_rerank_callback(
                scorer=scorer_for_search,
                rerank_cache=rerank_cache,
                base_prompt=base_prompt,
                args=args,
            )
            # QD archives for this category
            _qd_n, _qd_a = None, None
            if _use_qd:
                _qd_n = QDArchive(bd_names=_qd_bd_names, bins_per_dim=_qd_bins)
                _qd_a = QDArchive(bd_names=_qd_bd_names, bins_per_dim=_qd_bins)
            opt_n_list, opt_a_list = evo_optimizer.optimize_dual(
                [base_prompt], scoring_callback=callback,
                rerank_callback=rerank_cb,
                rerank_topk=int(getattr(args, "stage2_rerank_topk", 8)),
                qd_archive_normal=_qd_n,
                qd_archive_abnormal=_qd_a,
                qd_bd_names=_qd_bd_names if _use_qd else None,
            )
            if _use_qd:
                _qd_archives[category] = {"normal": _qd_n, "abnormal": _qd_a}
                logger.info(
                    "  QD archive: normal=%d/%d cells (%.0f%%), abnormal=%d/%d cells (%.0f%%)",
                    _qd_n.size, _qd_n.max_cells, _qd_n.coverage * 100,
                    _qd_a.size, _qd_a.max_cells, _qd_a.coverage * 100,
                )
            opt_normal, opt_abnormal = opt_n_list[0], opt_a_list[0]
            optimized_rules["normal"][category] = opt_normal
            optimized_rules["abnormal"][category] = opt_abnormal
            safe_meta = _compute_pair_safe_metadata(
                scorer=scorer,
                eval_cache=rerank_cache,
                cross_cache_by_category=_active_ccto_cache,
                category=category,
                best_normal_prompt=opt_normal,
                best_abnormal_prompt=opt_abnormal,
                args=args,
            )
            safe_meta["effective_ccto_alpha"] = _active_alpha
            if bool(getattr(args, "asym_b_enable", False)):
                safe_meta["asym_b_config"] = _asym_b_config_dict(args, _ccto_scope_effective)
            if _alpha_trace:
                safe_meta["adaptive_ccto_trace"] = {
                    "policy": _alpha_trace.get("policy"),
                    "decision_reason": _alpha_trace.get("decision_reason"),
                    "raw_alpha": _alpha_trace.get("raw_alpha"),
                    "signals": _alpha_trace.get("signals", {}),
                }
                if adaptive_alpha_summary:
                    safe_meta["adaptive_ccto_summary"] = dict(adaptive_alpha_summary)
            evo_optimizer.set_rule_metadata(category, "pair", safe_meta)
            logger.info("  normal=%s", opt_normal)
            logger.info("  abnormal=%s", opt_abnormal)
            logger.info(
                "  safe_meta: src %.4f→%.4f (gain=%.4f), cross %s→%s (gain=%s), score_std=%.4f",
                float(safe_meta["default_src"]),
                float(safe_meta["best_src"]),
                float(safe_meta["gain_src"]),
                "None" if safe_meta["default_cross"] is None else f"{float(safe_meta['default_cross']):.4f}",
                "None" if safe_meta["best_cross"] is None else f"{float(safe_meta['best_cross']):.4f}",
                "None" if safe_meta["gain_cross"] is None else f"{float(safe_meta['gain_cross']):.4f}",
                float(safe_meta["score_std"]),
            )
            if _active_ccto_cache and _active_alpha < 1.0:
                logger.info(
                    "  CCTO: alpha=%.2f other_categories=%d cross_batches=%d scope=%s",
                    _active_alpha,
                    len(_active_ccto_cache),
                    _ccto_total_batches,
                    _ccto_scope_effective,
                )
            elif _active_target_cache and _active_alpha < 1.0:
                logger.info("  CDACE: alpha=%.2f, target_batches=%d",
                            _active_alpha, len(_active_target_cache))

        elif args.evo_dual_branch:
            # standard dual-branch (optimized separately)
            cb_n = build_scoring_callback(
                scorer_for_search, evo_optimizer, search_cache, base_prompt,
                args, role_type="normal", use_coevo=False,
                target_cache=_active_target_cache, cdace_alpha=_active_alpha,
                ccto_per_cat_cache=_active_ccto_cache,
                ccto_scope=_ccto_scope_effective,
            )
            rerank_cb = build_rerank_callback(
                scorer=scorer_for_search,
                rerank_cache=rerank_cache,
                base_prompt=base_prompt,
                args=args,
            )
            _qd_n = QDArchive(bd_names=_qd_bd_names, bins_per_dim=_qd_bins) if _use_qd else None
            opt_normal = evo_optimizer.optimize(
                [base_prompt],
                scoring_callback=cb_n,
                role="normal",
                rerank_callback=rerank_cb,
                rerank_topk=int(getattr(args, "stage2_rerank_topk", 8)),
                qd_archive=_qd_n,
                qd_bd_names=_qd_bd_names if _use_qd else None,
            )[0]

            cb_a = build_scoring_callback(
                scorer_for_search, evo_optimizer, search_cache, base_prompt,
                args, role_type="abnormal", use_coevo=False,
                target_cache=_active_target_cache, cdace_alpha=_active_alpha,
                ccto_per_cat_cache=_active_ccto_cache,
                ccto_scope=_ccto_scope_effective,
            )
            _qd_a = QDArchive(bd_names=_qd_bd_names, bins_per_dim=_qd_bins) if _use_qd else None
            opt_abnormal = evo_optimizer.optimize(
                [base_prompt],
                scoring_callback=cb_a,
                role="abnormal",
                rerank_callback=rerank_cb,
                rerank_topk=int(getattr(args, "stage2_rerank_topk", 8)),
                qd_archive=_qd_a,
                qd_bd_names=_qd_bd_names if _use_qd else None,
            )[0]
            if _use_qd:
                _qd_archives[category] = {"normal": _qd_n, "abnormal": _qd_a}
                logger.info(
                    "  QD archive: normal=%d/%d cells (%.0f%%), abnormal=%d/%d cells (%.0f%%)",
                    _qd_n.size, _qd_n.max_cells, _qd_n.coverage * 100,
                    _qd_a.size, _qd_a.max_cells, _qd_a.coverage * 100,
                )

            optimized_rules["normal"][category] = opt_normal
            optimized_rules["abnormal"][category] = opt_abnormal
            safe_meta = _compute_pair_safe_metadata(
                scorer=scorer,
                eval_cache=rerank_cache,
                cross_cache_by_category=_active_ccto_cache,
                category=category,
                best_normal_prompt=opt_normal,
                best_abnormal_prompt=opt_abnormal,
                args=args,
            )
            safe_meta["effective_ccto_alpha"] = _active_alpha
            if bool(getattr(args, "asym_b_enable", False)):
                safe_meta["asym_b_config"] = _asym_b_config_dict(args, _ccto_scope_effective)
            if _alpha_trace:
                safe_meta["adaptive_ccto_trace"] = {
                    "policy": _alpha_trace.get("policy"),
                    "decision_reason": _alpha_trace.get("decision_reason"),
                    "raw_alpha": _alpha_trace.get("raw_alpha"),
                    "signals": _alpha_trace.get("signals", {}),
                }
                if adaptive_alpha_summary:
                    safe_meta["adaptive_ccto_summary"] = dict(adaptive_alpha_summary)
            evo_optimizer.set_rule_metadata(category, "pair", safe_meta)
            logger.info("  normal=%s", opt_normal)
            logger.info("  abnormal=%s", opt_abnormal)
            logger.info(
                "  safe_meta: src %.4f→%.4f (gain=%.4f), cross %s→%s (gain=%s), score_std=%.4f",
                float(safe_meta["default_src"]),
                float(safe_meta["best_src"]),
                float(safe_meta["gain_src"]),
                "None" if safe_meta["default_cross"] is None else f"{float(safe_meta['default_cross']):.4f}",
                "None" if safe_meta["best_cross"] is None else f"{float(safe_meta['best_cross']):.4f}",
                "None" if safe_meta["gain_cross"] is None else f"{float(safe_meta['gain_cross']):.4f}",
                float(safe_meta["score_std"]),
            )

        else:
            # shared optimization
            cb = build_scoring_callback(
                scorer_for_search, evo_optimizer, search_cache, base_prompt,
                args, role_type=None, use_coevo=use_coevo,
                target_cache=_active_target_cache, cdace_alpha=_active_alpha,
                ccto_per_cat_cache=_active_ccto_cache,
                ccto_scope=_ccto_scope_effective,
            )
            rerank_cb = build_rerank_callback(
                scorer=scorer_for_search,
                rerank_cache=rerank_cache,
                base_prompt=base_prompt,
                args=args,
            )
            _qd_s = QDArchive(bd_names=_qd_bd_names, bins_per_dim=_qd_bins) if _use_qd else None
            opt_shared = evo_optimizer.optimize(
                [base_prompt],
                scoring_callback=cb,
                role=None,
                rerank_callback=rerank_cb,
                rerank_topk=int(getattr(args, "stage2_rerank_topk", 8)),
                qd_archive=_qd_s,
                qd_bd_names=_qd_bd_names if _use_qd else None,
            )[0]
            optimized_rules["shared"][category] = opt_shared
            logger.info("  shared=%s", opt_shared)
            if _use_qd and _qd_s is not None:
                _qd_archives[category] = {"shared": _qd_s}
                logger.info("  QD archive: shared=%d/%d cells (%.0f%%)",
                            _qd_s.size, _qd_s.max_cells, _qd_s.coverage * 100)

        _write_stage2_partial_state(args.save_path, optimized_rules, evo_optimizer)
        logger.info(
            "Stage 2 partial checkpoint saved: completed_categories=%d",
            len(_completed_stage2_categories(optimized_rules)),
        )

        _offload_eval_cache(full_eval_cache)
        del full_eval_cache
        del search_cache
        del rerank_cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Per-category checkpoint (enables --resume)
        if args.save_path:
            os.makedirs(args.save_path, exist_ok=True)
            _ckpt_r = os.path.join(args.save_path, "checkpoint_rules.json")
            with open(_ckpt_r, "w", encoding="utf-8") as _f:
                json.dump(optimized_rules, _f, indent=2, ensure_ascii=False)
            _ckpt_c = os.path.join(args.save_path, "checkpoint_evo_cache.json")
            evo_optimizer.save_optimized_rules(_ckpt_c)
            logger.info("Checkpoint saved after '%s' (%d/%d)", category, idx + 1, len(obj_list))

    if adaptive_alpha_used:
        runtime_stats = _compute_alpha_diversity_stats(adaptive_alpha_used)
        logger.info(
            "ADAPTIVE_CCTO_SUMMARY|phase=runtime_used|count=%d|unique=%d|std=%.6f|min=%.6f|max=%.6f|spread=%.6f|collapsed=%d",
            int(runtime_stats.get("count", 0.0)),
            int(runtime_stats.get("unique_count", 0.0)),
            float(runtime_stats.get("std", 0.0)),
            float(runtime_stats.get("min", 0.0)),
            float(runtime_stats.get("max", 0.0)),
            float(runtime_stats.get("spread", 0.0)),
            1 if _is_alpha_collapsed(runtime_stats, args) else 0,
        )

    # free the CCTO cache
    if ccto_per_cat_cache:
        for _cc in ccto_per_cat_cache.values():
            _offload_eval_cache(_cc)
        ccto_per_cat_cache.clear()
        logger.info("CCTO: cross-category caches released")

    # save the rules
    rules_path = os.path.join(args.save_path, "optimized_prompt_rules.json")
    with open(rules_path, "w", encoding="utf-8") as f:
        json.dump(optimized_rules, f, indent=2, ensure_ascii=False)

    cache_path = os.path.join(args.save_path, "evo_prompt_cache.json")
    evo_optimizer.save_optimized_rules(cache_path)

    # Save QD archives
    if _use_qd and _qd_archives:
        qd_dir = os.path.join(args.save_path, "qd_archives")
        os.makedirs(qd_dir, exist_ok=True)
        for cat, archives in _qd_archives.items():
            for role_name, archive in archives.items():
                archive.save(os.path.join(qd_dir, f"{cat}_{role_name}.json"))
        logger.info("QD archives saved: %d categories to %s", len(_qd_archives), qd_dir)

    # build the shared bank (cross-category mean AUROC + MMR)
    shared_bank_path = os.path.join(args.save_path, "shared_prompt_bank.json")
    try:
        shared_bank = _build_shared_bank(
            scorer=scorer,
            evo_optimizer=evo_optimizer,
            make_dataset=make_dataset,
            obj_list=all_obj_list,
            args=args,
            logger=logger,
            device=device,
        )
        with open(shared_bank_path, "w", encoding="utf-8") as f:
            json.dump(shared_bank, f, indent=2, ensure_ascii=False)
        logger.info(
            "Stage 2 done. shared_bank=%s (pairs=%d)",
            shared_bank_path,
            len(shared_bank.get("pairs", [])),
        )
    except Exception as exc:
        logger.warning("Shared bank build skipped due to error: %s", exc)

    logger.info("Stage 2 done. rules=%s", rules_path)
    logger.info("Stage 2 done. cache=%s", cache_path)
    log_cuda_peak_memory(logger, "stage2", args.device_id)
    return optimized_rules


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        "Universal Stage 2 prompt optimization", add_help=True
    )

    # Compatibility flags
    parser.add_argument("--stage2_only", action="store_true")

    # Dataset / model
    parser.add_argument("--dataset", type=str, default="visa")
    parser.add_argument("--model", type=str, default="ViT-L-14-336")
    parser.add_argument("--pretrained", type=str, default="openai")
    parser.add_argument("--features_list", type=int, nargs="+",
                        default=[6, 12, 18, 24])

    # Paths
    parser.add_argument("--train_data_path", type=str,
                        default="./dataset/mvisa/data")
    parser.add_argument("--data_path", dest="train_data_path", type=str,
                        help="Compatibility alias for --train_data_path")
    parser.add_argument("--save_path", type=str,
                        default="./my_exps/universal_stage2")
    parser.add_argument("--config_path", type=str,
                        default="./open_clip_local/model_configs/ViT-L-14-336.json")
    parser.add_argument("--pretrained_path", type=str,
                        default="./pretrained_weight/ViT-L-14-336px.pt")
    parser.add_argument("--checkpoint_path", type=str, default="",
                        help="Model weight path used by Stage2; whether it is required depends on the scorer")

    parser.add_argument("--prompt_context_len", type=int, default=5)
    parser.add_argument("--prompt_num", type=int, default=8)
    parser.add_argument("--prompt_state_len", type=int, default=5)
    parser.add_argument("--per_slot_mapping", action="store_true",
                        help="Use per-slot class_mapping (auto-detected from checkpoint if omitted)")

    # Training config
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--num_workers", type=int, default=4)

    # EvoPrompt
    parser.add_argument("--evo_population", type=int, default=16)
    parser.add_argument("--evo_generations", type=int, default=5)
    parser.add_argument("--evo_topk", type=int, default=4)
    parser.add_argument("--evo_lambda_diversity", type=float, default=0.2)
    parser.add_argument("--evo_random_search", action="store_true",
                        help="Equal-budget random prompt search control: evaluate N*(G+1) random candidates and disable mutation/crossover.")
    parser.add_argument("--evo_crossover_rate", type=float, default=0.0,
                        help="Standard EvoPrompt GA crossover rate. Default 0.0 preserves mutation-only search.")
    parser.add_argument("--evo_mutation_ops", type=str, default="",
                        help="Comma-separated Stage2 mutation operators for ablations. "
                             "Empty preserves the default rule set.")
    parser.add_argument("--evo_dual_branch", action="store_true")
    parser.add_argument("--evo_val_batches", type=int, default=10)
    parser.add_argument("--evo_search_max_batches", type=int, default=0,
                        help="Stage2 search-cache expansion upper bound. <=0 means use full per-category eval cache when auto-expanding single-class search labels.")
    parser.add_argument("--evo_fitness_agg", type=str, default="mean",
                        choices=["mean", "cvar"],
                        help="Cross-category fitness aggregation: mean (all) or cvar (bottom-k worst)")
    parser.add_argument("--use_qd_archive", action="store_true",
                        help="Enable MAP-Elites QD archive for prompt evolution")
    parser.add_argument("--qd_bins", type=int, default=5,
                        help="Number of bins per behavior descriptor dimension for QD archive")
    parser.add_argument("--qd_bd", type=str, default="image_auroc,pixel_f1",
                        help="Comma-separated behavior descriptor names for QD archive")
    parser.add_argument("--evo_cvar_k", type=int, default=3,
                        help="Number of worst categories to average when evo_fitness_agg=cvar (>=1)")
    parser.add_argument("--candidate_batch_size", type=int, default=2,
                        help="Number of candidates evaluated in parallel per Stage2 step")
    parser.add_argument("--zero_shot_scoring", dest="zero_shot_scoring", action="store_true",
                        help="Compatibility mode for legacy PFL-style text encoders/checkpoints: replace latent prompt bias with zeros")
    parser.add_argument("--no_zero_shot_scoring", dest="zero_shot_scoring", action="store_false",
                        help="If a legacy PFL-compatible scorer/model is present, use its learned latent prompt bias")
    parser.set_defaults(zero_shot_scoring=False)
    parser.add_argument("--shared_bank_eval_batches", type=int, default=2,
                        help="Max evaluation batches per category when building the shared bank")
    parser.add_argument("--shared_bank_max_candidates_per_role", type=int, default=48,
                        help="Max candidates per role when building the shared bank")
    parser.add_argument("--shared_bank_size", type=int, default=4,
                        help="Number of candidate pairs finally kept when building the shared bank")
    parser.add_argument("--stage2_objective", type=str, default="image_pixel",
                        choices=["image_pixel", "image_pixel_hmean", "image_only", "image_teacher"],
                        help="Stage2 candidate scoring objective")
    parser.add_argument("--stage2_weight_image", type=float, default=0.3,
                        help="Weight on image AUROC within the image_pixel objective")
    parser.add_argument("--stage2_weight_pixel_ap", type=float, default=0.5,
                        help="Weight on pixel AP within the image_pixel objective")
    parser.add_argument("--stage2_weight_pixel_f1", type=float, default=0.2,
                        help="Weight on pixel F1max within the image_pixel objective")
    parser.add_argument("--stage2_metric_resolution", type=int, default=256,
                        help="Resolution at which Stage2 pixel metrics are evaluated")
    parser.add_argument("--pixel_layer_weights", type=float, nargs=4, default=[1, 1, 1, 1],
                        help="Per-layer pixel fusion weights; length must match the number of patch feature layers")
    parser.add_argument("--stage2_rerank_topk", type=int, default=8,
                        help="Number of candidates reranked by mean IoU at the end")
    parser.add_argument("--stage2_stability_bootstrap", type=int, default=3,
                        help="Number of bootstrap rounds for Stage2 stability")
    parser.add_argument("--stage2_stability_weight", type=float, default=0.1,
                        help="Weight of the Stage2 stability regularizer")
    parser.add_argument("--stage2_split", type=str, default="test",
                        help="Data split for Stage2 prompt search (train/val/test). "
                             "Default 'test' uses source domain test split (source-supervised).")
    parser.add_argument("--allow_split_fallback", action="store_true",
                        help="Allow fallback to 'test' split when requested split not in meta JSON "
                             "(needed for cross-domain where meta JSON only has 'test' key)")
    parser.add_argument("--stage2_categories", type=str, nargs="*", default=None,
                        help="Optional subset of categories to optimize, kept in dataset order. "
                             "Useful for interrupted Stage2 recovery.")
    parser.add_argument("--resume_stage2_log", type=str, default="",
                        help="Path to an interrupted Stage2 log used to recover already-completed categories.")

    # CoEvo
    parser.add_argument("--use_coevo_prompt", action="store_true")
    parser.add_argument("--coevo_pair_k", type=int, default=3)
    parser.add_argument("--coevo_alpha_auroc", type=float, default=0.85)
    parser.add_argument("--coevo_beta_contrast", type=float, default=0.15)
    parser.add_argument("--coevo_crossover_rate", type=float, default=0.0,
                        help="Budget-matched same-role crossover rate for CoEvo offspring generation. "
                             "Default 0.0 preserves mutation-only search.")
    parser.add_argument("--coevo_final_pair_select", type=str, default="marginal",
                        choices=["marginal", "global_argmax"],
                        help="Final CoEvo pair selector. Default 'marginal' preserves the current "
                             "split normal/abnormal selection; 'global_argmax' selects a final "
                             "normal/abnormal pair over the final candidate cross-product.")
    parser.add_argument("--coevo_record_population_trace", action="store_true",
                        help="Persist per-generation CoEvo populations and evaluated pair candidates for source-only replay gates.")

    # CDACE (Cross-Domain Aware Co-Evolution)
    parser.add_argument("--cdace_target_dataset", type=str, default="",
                        help="Target domain dataset for cross-domain fitness "
                             "(e.g. 'visa' when source is 'mvtec'). Empty = disabled.")
    parser.add_argument("--cdace_target_categories", type=str, nargs="*", default=None,
                        help="Target domain categories for fitness eval "
                             "(default: auto-select 5 from opposite domain)")
    parser.add_argument("--cdace_alpha", type=float, default=0.6,
                        help="Source domain weight in CDACE fitness "
                             "(target weight = 1 - alpha). Default 0.6.")
    parser.add_argument("--cdace_target_batches", type=int, default=2,
                        help="Max eval batches per target category (keep small for speed)")

    # Cross-Category Transfer Objective (CCTO)
    parser.add_argument("--ccto", action="store_true",
                        help="Enable Cross-Category Transfer Objective "
                             "(zero-shot: uses source cross-category eval instead of CDACE)")
    parser.add_argument("--ccto_alpha", type=float, default=0.6,
                        help="Own-category weight in CCTO fitness "
                             "(cross-category weight = 1 - alpha). Default 0.6.")
    parser.add_argument("--ccto_batches", type=int, default=20,
                        help="Max eval batches per cross-category (keep small for speed)")
    parser.add_argument("--ccto_scope", type=str, default="abnormal_only",
                        choices=["abnormal_only", "symmetric"],
                        help="CCTO scope: 'abnormal_only' applies CCTO only to abnormal branch; "
                             "'symmetric' applies to both normal and abnormal branches.")
    parser.add_argument("--asym_b_enable", action="store_true",
                        help="Enable role-conditioned asymmetric objective (normal-generalization vs abnormal-specialization).")
    parser.add_argument("--asym_b_lambda_normal_gen", type=float, default=0.35,
                        help="Normal-role cross-generalization weight λ_ng in asym_b mode.")
    parser.add_argument("--asym_b_lambda_abn_spec", type=float, default=0.20,
                        help="Abnormal-role specialization weight λ_as in asym_b mode.")
    parser.add_argument("--asym_b_policy", type=str, default="fixed",
                        choices=["fixed", "manual_kappa"],
                        help="Role-asymmetry policy. fixed preserves legacy lambdas; manual_kappa uses signed kappa flags.")
    parser.add_argument("--asym_b_kappa_normal", type=float, default=0.35,
                        help="Signed normal-role kappa for --asym_b_policy manual_kappa.")
    parser.add_argument("--asym_b_kappa_abnormal", type=float, default=-0.20,
                        help="Signed abnormal-role kappa for --asym_b_policy manual_kappa.")
    parser.add_argument("--game_metrics_enable", action="store_true", default=None,
                        help="Per-generation game-theory metrics logging. Auto-on with --asym_b_enable.")
    parser.add_argument("--no_game_metrics", dest="game_metrics_enable", action="store_false",
                        help="Disable game metrics even with asym_b.")
    parser.add_argument("--llm_mutation_enabled", action="store_true",
                        help="Enable LLM-based mutation operators in prompt evolution.")
    parser.add_argument("--llm_mutation_model_id", type=str, default="",
                        help="Local path or HuggingFace model ID for LLM mutation.")
    parser.add_argument("--llm_mutation_max_tokens", type=int, default=32,
                        help="Max new tokens for LLM mutation generation.")
    parser.add_argument("--enable_default_regret_ccto", action="store_true", default=True,
                        help="Enable source-only default-regret constraint during CCTO scoring")
    parser.add_argument("--disable_default_regret_ccto", dest="enable_default_regret_ccto", action="store_false",
                        help="Disable source-only default-regret constraint during CCTO scoring")
    parser.add_argument("--safe_regret_src_weight", type=float, default=1.0,
                        help="Penalty weight for own-category regret against default prompt")
    parser.add_argument("--safe_regret_cross_weight", type=float, default=1.0,
                        help="Penalty weight for cross-category regret against default prompt")
    parser.add_argument("--safe_regret_std_weight", type=float, default=0.0,
                        help="Extra penalty weight for candidate bootstrap score_std "
                             "(stage2_stability_weight remains unchanged)")
    parser.add_argument("--ccto_cross_agg", type=str, default="mean",
                        choices=["mean", "min", "bottomk"],
                        help="Cross-category score aggregation mode for CCTO. "
                             "'mean' averages all categories (baseline). "
                             "'min' takes the worst category (minimax). "
                             "'bottomk' averages the k lowest category scores. Default: mean.")
    parser.add_argument("--ccto_bottomk", type=int, default=3,
                        help="Number of lowest-scoring categories to average when "
                             "--ccto_cross_agg=bottomk. Clamped to [1, num_categories]. Default: 3.")

    # Adaptive per-category CCTO alpha
    parser.add_argument("--adaptive_ccto_alpha", action="store_true",
                        help="Enable per-category adaptive CCTO alpha based on cross-category reliability")
    parser.add_argument("--adaptive_ccto_policy", type=str, default="continuous_v2",
                        choices=["continuous_v2", "legacy_threshold"],
                        help="Adaptive CCTO alpha policy. Default continuous_v2.")
    parser.add_argument("--ccto_alpha_min", type=float, default=0.3,
                        help="Min alpha (more cross-category influence). Default 0.3.")
    parser.add_argument("--ccto_alpha_max", type=float, default=0.9,
                        help="Max alpha (more source influence). Default 0.9.")
    parser.add_argument("--ccto_std_threshold", type=float, default=0.5,
                        help="Cross score_std above this triggers fallback to max_alpha. Default 0.5.")
    parser.add_argument("--adaptive_ccto_temperature", type=float, default=4.0,
                        help="Temperature for continuous_v2 sigmoid alpha mapping. Default 4.0.")
    parser.add_argument("--adaptive_ccto_std_penalty", type=float, default=1.0,
                        help="Penalty weight of cross score_std in adaptive reliability. Default 1.0.")
    parser.add_argument("--adaptive_ccto_reliability_clip", type=float, default=2.5,
                        help="Clip abs value for bounded reliability before sigmoid mapping. Default 2.5.")
    parser.add_argument("--adaptive_ccto_reliability_eps", type=float, default=1e-6,
                        help="Epsilon used in bounded reliability normalization denominator. Default 1e-6.")
    parser.add_argument("--adaptive_ccto_eval_batches", type=int, default=0,
                        help="Eval batches per category for adaptive alpha precheck. <=0 uses evo_val_batches.")
    parser.add_argument("--adaptive_ccto_collapse_mode", type=str, default="warn",
                        choices=["warn", "remap", "strict"],
                        help="Action when adaptive alphas collapse: warn/remap/strict.")
    parser.add_argument("--adaptive_ccto_collapse_min_unique", type=int, default=2,
                        help="Minimum unique alpha count expected across categories. Default 2.")
    parser.add_argument("--adaptive_ccto_collapse_min_std", type=float, default=0.01,
                        help="Minimum alpha std expected across categories. Default 0.01.")
    parser.add_argument("--adaptive_ccto_collapse_min_spread", type=float, default=0.05,
                        help="Minimum alpha max-min spread expected across categories. Default 0.05.")

    # Scorer
    parser.add_argument("--scorer_type", type=str, default="prompt_bank",
                        choices=["prompt_bank", "clip_transfer", "custom", "external"])
    parser.add_argument("--use_scoring_head", action="store_true",
                        help="Use learned MLP scoring head instead of cosine similarity (requires Stage1.5 checkpoint)")
    parser.add_argument("--external_adapter", type=str, default="anomalyclip",
                        help="External model adapter (only with --scorer_type external)")
    parser.add_argument("--scorer_module", type=str, default="")
    parser.add_argument("--scorer_class", type=str, default="")
    parser.add_argument("--scorer_config", type=str, default="",
                        help="YAML/JSON config file for custom scorer kwargs")
    parser.add_argument("--scorer_kwargs_json", type=str, default="",
                        help="(deprecated) JSON string for custom scorer kwargs")

    # Resume
    parser.add_argument("--resume", action="store_true", default=False,
                        help="Resume from per-category checkpoint in save_path (skip already-optimized categories)")

    # Device
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=111)

    return parser


def main(args=None):
    """Entry point. Accepts externally supplied args (used when train_two_stage.py dispatches here)."""
    if args is None:
        parser = build_parser()
        args = parser.parse_args()
        _validate_transfer_regularizer_args(args, parser=parser)
    else:
        _validate_transfer_regularizer_args(args)

    _enforce_transfer_mainline(args)
    setup_seed(args.seed)
    run_stage2_universal(args)


if __name__ == "__main__":
    main()
