"""
Universal Test - unified inference using a scorer plus optimized EvoPrompt rules.

Supports a deterministic Prompt Bank and arbitrary external models (via the AnomalyScorer interface).
Hard constraint on pixel maps: for datasets requiring segmentation evaluation, if the scorer
returns pixel_maps=None, a ValueError is raised immediately.

Usage:
    # Prompt bank scorer
    python test_universal.py --scorer_type prompt_bank --dataset mvtec \
        --checkpoint_path ./my_exps/.../stage1_final.pth \
        --evo_rules_path ./my_exps/.../evo_prompt_cache.json

    # CLIP transfer scorer
    python test_universal.py --scorer_type clip_transfer --dataset visa \
        --evo_rules_path ./my_exps/.../evo_prompt_cache.json

    # Custom scorer
    python test_universal.py --scorer_type custom \
        --scorer_module models.scorer_template --scorer_class VanillaCLIPScorer \
        --dataset visa --evo_rules_path ./my_exps/.../evo_prompt_cache.json

:author: PromptBank Universal Test
:date: 2026
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib as _hashlib
import json
import logging
import math
import os
import random
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

from utils.cuda_memory import log_cuda_peak_memory, reset_cuda_peak_memory

_REQUIRED_FROZEN_KEYS = (
    "direction", "source_dataset", "target_dataset",
    "alpha", "k_ratio", "stat", "grid_hash", "fold_seed",
    "source_cal_log", "source_cal_auroc_sp_per_fold",
    "source_cal_auroc_sp_fold_mean", "source_cal_auroc_sp_fold_std",
    "fold_ranks", "candidate_pool_size", "created_at", "spec_commit_sha",
)


def _compute_current_grid_hash() -> str:
    """Recompute the canonical-JSON sha256 of the pre-registered grid.

    Read lazily, so the runtime cost is one YAML parse per process and only
    when validating a frozen config.
    """
    import yaml
    grid_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "configs", "score_fusion_grid.yaml",
    )
    with open(grid_path, "r", encoding="utf-8") as fh:
        grid = yaml.safe_load(fh)
    canonical = json.dumps(grid, sort_keys=True, separators=(",", ":"))
    return _hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _load_score_fusion_config(path):
    """Load and validate a frozen score-fusion config.

    Empty string → None (default path). Any failure → SystemExit (no silent
    fallback: a bad config is an error, never a silent default).
    """
    if not path:
        return None
    if not os.path.exists(path):
        sys.exit(f"score_fusion_config: file not found: {path!r}")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
    except Exception as e:
        sys.exit(f"score_fusion_config: malformed JSON at {path!r}: {e}")
    missing = [k for k in _REQUIRED_FROZEN_KEYS if k not in cfg]
    if missing:
        sys.exit(f"score_fusion_config: missing keys {missing} in {path!r}")
    expected_hash = _compute_current_grid_hash()
    if cfg["grid_hash"] != expected_hash:
        sys.exit(
            f"score_fusion_config: grid_hash mismatch.\n"
            f"  config: {cfg['grid_hash']}\n"
            f"  current grid: {expected_hash}\n"
            f"Re-calibrate against the current grid YAML."
        )
    return cfg


REMOVED_INTERFACE_FLAGS = {
    "--disable_role_fallback": "Removed. Mainline no longer supports role/shared fallback routing.",
    "--enable_semantic_fallback": "Removed. Mainline routing is strict class-name only.",
    "--semantic_fallback_template": "Removed together with semantic fallback.",
    "--semantic_fallback_min_sim": "Removed together with semantic fallback.",
    "--semantic_fallback_min_margin": "Removed together with semantic fallback.",
    "--enable_template_transfer": "Removed. Mainline no longer supports template transfer routing.",
    "--semantic_topk": "Removed. Top-k semantic routing is no longer in mainline.",
    "--semantic_tau": "Removed together with top-k semantic routing.",
    "--semantic_gate": "Removed together with top-k semantic routing.",
    "--asymmetric_transfer": "Removed from main test entrypoint.",
    "--visual_routing": "Removed. Visual routing is no longer in mainline.",
    "--visual_proto_path": "Removed together with visual routing.",
    "--enable_safe_switch": "Removed. Safe-switch mixing/fallback is no longer in mainline.",
    "--safe_switch_min_gain_cross": "Removed together with safe-switch.",
    "--safe_switch_min_gain_src": "Removed together with safe-switch.",
    "--safe_switch_max_score_std": "Removed together with safe-switch.",
    "--safe_switch_enable_mix": "Removed together with safe-switch.",
    "--safe_switch_mix_gain_cross": "Removed together with safe-switch.",
    "--safe_switch_mix_score_std": "Removed together with safe-switch.",
    "--safe_switch_mix_alpha_min": "Removed together with safe-switch.",
    "--safe_switch_mix_alpha_max": "Removed together with safe-switch.",
}


def _preflight_removed_flags_or_exit(argv: List[str]) -> None:
    hit = []
    for token in argv:
        if not token.startswith("--"):
            continue
        flag = token.split("=", 1)[0]
        if flag in REMOVED_INTERFACE_FLAGS and flag not in hit:
            hit.append(flag)
    if not hit:
        return
    detail = "\n".join(f"  - {flag}: {REMOVED_INTERFACE_FLAGS[flag]}" for flag in hit)
    raise SystemExit(
        "BREAKING: historical experiment flags were removed from mainline `test_universal.py`.\n"
        f"{detail}\n"
        "Use strict/mainline commands without these flags."
    )


def _active_entrypoint_profile() -> str:
    profile = str(os.environ.get("BAYES_PFL_TEST_ROUTE_PROFILE", "STRICT_MAINLINE")).strip().upper()
    return profile or "STRICT_MAINLINE"


if _active_entrypoint_profile() == "STRICT_MAINLINE":
    _preflight_removed_flags_or_exit(sys.argv[1:])

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor

from datasets import Makedataset
from models.evoprompt import EvoPromptOptimizer, build_route_audit_payload
from models.metric_and_visualization import calcuate_metric_image, calcuate_metric_pixel
from models.scorer import build_scorer
from utils.shared_bank_fusion import (
    fuse_cls_and_shared,
    resolve_alpha_from_source,
    softmax_from_similarities,
)
from utils.icsr_audit import (
    create_icsr_audit,
    format_icsr_alpha_lines,
    format_icsr_audit_lines,
    record_icsr_alpha,
    record_icsr_audit_event,
)
from utils.soft_mix import (
    SoftMixOptions,
    _SOFT_MIX_IGNORED_HARD_GATE_FLAGS,
    _blend_score_and_map,
    _clip_normalize,
    _compute_ignored_soft_mix_flags,
    _dump_debug_scores,
    _flag_explicitly_passed,
    compute_soft_mix_alpha,
)

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


DATASETS_ONLY_CLASSIFICATION = ["HeadCT", "BrainMRI", "Br35H"]
ROUTE_TEXT_TEMPLATE = "a photo of {}"

# Standard CLIP 7-template subset, used to diversify ICSR bank embeddings
# beyond a single "a photo of {}" prompt. Default OFF; opt-in via
# --icsr_bank_template_ensemble.
ICSR_BANK_TEMPLATES_DEFAULT: Tuple[str, ...] = (
    "a photo of a {}.",
    "a blurry photo of a {}.",
    "a low resolution photo of a {}.",
    "a close-up photo of a {}.",
    "a bright photo of a {}.",
    "a photo of the {}.",
    "a photo of many {}.",
)


@dataclass(frozen=True)
class RouteRuntimeConfig:
    profile_name: str = "STRICT_MAINLINE"
    allow_role_fallback: bool = False
    enable_semantic_fallback: bool = False
    enable_template_transfer: bool = False
    semantic_fallback_template: str = ROUTE_TEXT_TEMPLATE
    semantic_fallback_min_sim: float = 0.5
    semantic_fallback_min_margin: float = 0.0
    enable_safe_switch: bool = False
    safe_switch_min_gain_cross: float = 0.0
    safe_switch_min_gain_src: float = -0.02
    safe_switch_max_score_std: float = 0.05
    safe_switch_enable_mix: bool = False
    safe_switch_mix_gain_cross: float = 0.1
    safe_switch_mix_score_std: float = 0.02
    safe_switch_mix_alpha_min: float = 0.25
    safe_switch_mix_alpha_max: float = 0.75


STRICT_ROUTE_CONFIG = RouteRuntimeConfig(profile_name="STRICT_MAINLINE")


def build_supplementary_route_config(args) -> RouteRuntimeConfig:
    return RouteRuntimeConfig(
        profile_name="SUPPLEMENTARY_FALLBACK",
        allow_role_fallback=not bool(getattr(args, "disable_role_fallback", False)),
        enable_semantic_fallback=bool(getattr(args, "enable_semantic_fallback", True)),
        enable_template_transfer=bool(getattr(args, "enable_template_transfer", True)),
        semantic_fallback_template=str(getattr(args, "semantic_fallback_template", ROUTE_TEXT_TEMPLATE)),
        semantic_fallback_min_sim=float(getattr(args, "semantic_fallback_min_sim", 0.4)),
        semantic_fallback_min_margin=float(getattr(args, "semantic_fallback_min_margin", 0.0)),
        enable_safe_switch=bool(getattr(args, "enable_safe_switch", False)),
        safe_switch_min_gain_cross=float(getattr(args, "safe_switch_min_gain_cross", 0.0)),
        safe_switch_min_gain_src=float(getattr(args, "safe_switch_min_gain_src", -0.02)),
        safe_switch_max_score_std=float(getattr(args, "safe_switch_max_score_std", 0.05)),
        safe_switch_enable_mix=bool(getattr(args, "safe_switch_enable_mix", False)),
        safe_switch_mix_gain_cross=float(getattr(args, "safe_switch_mix_gain_cross", 0.1)),
        safe_switch_mix_score_std=float(getattr(args, "safe_switch_mix_score_std", 0.02)),
        safe_switch_mix_alpha_min=float(getattr(args, "safe_switch_mix_alpha_min", 0.25)),
        safe_switch_mix_alpha_max=float(getattr(args, "safe_switch_mix_alpha_max", 0.75)),
    )


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform_test(n_px):
    return Compose([
        Resize((n_px, n_px), interpolation=BICUBIC),
        CenterCrop((n_px, n_px)),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073),
                  (0.26862954, 0.26130258, 0.27577711)),
    ])


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(save_path, log_filename="log_optimized_universal.txt"):
    os.makedirs(save_path, exist_ok=True)
    txt_path = os.path.join(save_path, log_filename)

    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.setLevel(logging.WARNING)

    logger = logging.getLogger("test_universal")
    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s",
        datefmt="%y-%m-%d %H:%M:%S",
    )
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(txt_path, mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


def _load_model_config(args):
    config_path = getattr(args, "config_path", "")
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            model_configs = json.load(f)
        args.vision_width = model_configs["vision_cfg"]["width"]
        args.text_width = model_configs["text_cfg"]["width"]
        args.embed_dim = model_configs["embed_dim"]


def _enforce_transfer_mainline(args, logger=None, route_config: Optional[RouteRuntimeConfig] = None):
    """Test-entry constraints: a fixed test split and an auditable routing-configuration identity."""
    profile_name = (route_config.profile_name if route_config is not None else "STRICT_MAINLINE")
    msg = (
        f"Universal test route-profile='{profile_name}' "
        f"uses scorer='{getattr(args, 'scorer_type', 'prompt_bank')}'"
    )
    if logger is not None:
        logger.info(msg)


# ---------------------------------------------------------------------------
# EvoPrompt rule loading and fallback
# ---------------------------------------------------------------------------

def setup_evo_optimizer_for_test(
    args,
    device,
    scorer=None,
    logger=None,
    route_config: Optional[RouteRuntimeConfig] = None,
):
    """Load rules, detect dual/shared mode, configure fallback, and return the prepared evo_optimizer."""
    route_config = route_config or STRICT_ROUTE_CONFIG
    evo_optimizer = EvoPromptOptimizer(
        population_size=getattr(args, "evo_population", 8),
        generations=getattr(args, "evo_generations", 3),
        topk=getattr(args, "evo_topk", 4),
    )
    # Record whether usable rules were actually loaded at test time (decides the baseline path)
    evo_optimizer.has_loaded_rules = False

    rules_path = getattr(args, "evo_rules_path", "")
    if rules_path and os.path.exists(rules_path):
        evo_optimizer.load_optimized_rules(rules_path)
        evo_optimizer.has_loaded_rules = len(evo_optimizer.cache) > 0
        if logger:
            logger.info("Loaded rules: %s (%d entries)", rules_path, len(evo_optimizer.cache))
            if not evo_optimizer.has_loaded_rules:
                logger.warning(
                    "Rules file exists but cache is empty; this run will behave as no-evo-rules baseline"
                )
    else:
        if logger:
            logger.warning("Rules file not found: %s; fallback prompts will be used", rules_path)

    # auto-detect dual/shared mode
    role_stats = {"normal": 0, "abnormal": 0, "shared": 0, "other": 0}
    for key in evo_optimizer.cache.keys():
        if isinstance(key, tuple) and len(key) == 2:
            role = key[0]
            role_stats[role] = role_stats.get(role, 0) + 1
        else:
            role_stats["other"] += 1
    if logger:
        logger.info(
            "Rule stats: normal=%d abnormal=%d shared=%d other=%d",
            role_stats["normal"], role_stats["abnormal"],
            role_stats["shared"], role_stats["other"],
        )

    if (not getattr(args, "evo_dual_branch", False)
            and role_stats["shared"] == 0
            and (role_stats["normal"] > 0 and role_stats["abnormal"] > 0)):
        args.evo_dual_branch = True
        if logger:
            logger.info("Auto-enable --evo_dual_branch (detected dual rules)")

    # configure the inference mode (strict/supp is controlled by route_config)
    evo_optimizer.text_replace_only = True
    evo_optimizer.donor_route_enabled = not bool(getattr(args, "disable_donor_route", True))
    evo_optimizer.configure_inference_resolution(
        enable=True,
        allow_role_fallback=route_config.allow_role_fallback,
        enable_semantic_fallback=route_config.enable_semantic_fallback,
        enable_template_transfer=route_config.enable_template_transfer,
    )

    if route_config.enable_semantic_fallback:
        embedder = None
        ext_adapter = getattr(scorer, "adapter", None) if scorer is not None else None
        clip_model = getattr(scorer, "model_clip", None) if scorer is not None else None
        tokenizer = getattr(scorer, "tokenizer", None) if scorer is not None else None
        if ext_adapter is not None and hasattr(ext_adapter, "encode_text"):
            def _encode_texts_with_adapter(texts):
                with torch.no_grad():
                    return ext_adapter.encode_text(texts).cpu()

            embedder = _encode_texts_with_adapter
        elif clip_model is not None and tokenizer is not None:
            def _encode_texts_with_clip(texts):
                with torch.no_grad():
                    tokens = tokenizer(texts)
                    if hasattr(tokens, "to"):
                        tokens = tokens.to(device)
                    elif isinstance(tokens, dict):
                        tokens = {k: v.to(device) for k, v in tokens.items()}
                    else:
                        tokens = torch.as_tensor(tokens, device=device)
                    feats = clip_model.encode_text(tokens)
                    feats = F.normalize(feats.float(), dim=-1, eps=1e-12)
                    return feats.cpu()

            embedder = _encode_texts_with_clip

        if embedder is not None:
            evo_optimizer.set_semantic_embedder(
                embedder,
                template=route_config.semantic_fallback_template,
                min_similarity=route_config.semantic_fallback_min_sim,
                min_margin=route_config.semantic_fallback_min_margin,
            )
            if logger:
                logger.info(
                    "Semantic fallback enabled via text encoder (template='%s', min_sim=%.3f, min_margin=%.3f)",
                    route_config.semantic_fallback_template,
                    route_config.semantic_fallback_min_sim,
                    route_config.semantic_fallback_min_margin,
                )
        elif logger:
            logger.warning("Semantic fallback requested but scorer has no text encoder; disabled")

    return evo_optimizer


def resolve_prompts(
    args,
    evo_optimizer,
    cls_name_l,
    route_config: Optional[RouteRuntimeConfig] = None,
):
    """Resolve normal/abnormal prompts from the EvoPrompt cache."""
    route_config = route_config or STRICT_ROUTE_CONFIG
    allow_role_fallback = route_config.allow_role_fallback
    if getattr(args, "global_shared_prompts", False):
        if getattr(args, "evo_dual_branch", False):
            return (
                getattr(args, "shared_normal_prompt", "X normal object"),
                getattr(args, "shared_abnormal_prompt", "X anomalous object"),
                "global_shared",
                {"mode": "global_shared", "target_category": cls_name_l},
            )
        sp = getattr(args, "shared_prompt", "X object")
        return sp, sp, "global_shared", {"mode": "global_shared", "target_category": cls_name_l}

    resolver = getattr(evo_optimizer, "resolve_cached_prompt", None)
    if resolver is None:
        # no resolver; fall back to the default
        if getattr(args, "evo_dual_branch", False):
            return (
                f"X normal {cls_name_l}",
                f"X abnormal {cls_name_l}",
                "default",
                {
                    "mode": "dual",
                    "target_category": cls_name_l,
                    "normal_source_category": cls_name_l,
                    "abnormal_source_category": cls_name_l,
                    "normal_resolution": "default",
                    "abnormal_resolution": "default",
                },
            )
        return f"X {cls_name_l}", f"X {cls_name_l}", "default", {
            "mode": "shared",
            "target_category": cls_name_l,
            "shared_source_category": cls_name_l,
            "shared_resolution": "default",
        }

    if getattr(args, "evo_dual_branch", False):
        multi_resolver = getattr(evo_optimizer, "resolve_multi_source_prompts", None)
        if (
            callable(multi_resolver)
            and (route_config.enable_semantic_fallback or route_config.enable_template_transfer)
        ):
            multi_out = multi_resolver(
                cls_name_l,
                topk=max(2, min(3, int(getattr(args, "prompt_num", 3)))),
                tau=1.0,
                gate_threshold=float(getattr(route_config, "semantic_fallback_min_sim", 0.4)),
                allow_role_fallback=allow_role_fallback,
            )
            pairs = list(multi_out.get("pairs") or [])
            route_debug = multi_out.get("route_debug") or {}
            top_pair = (route_debug.get("pairs") or [{}])[0]
            if pairs:
                normal_prompt, abnormal_prompt = pairs[0][0], pairs[0][1]
                route_audit = [
                    build_route_audit_payload(
                        target_class=cls_name_l,
                        role="normal",
                        selected_source_class=top_pair.get("normal_source"),
                        donor_score=float(top_pair.get("normal_score", 0.0)),
                        semantic_sim=float(top_pair.get("normal_semantic_sim", top_pair.get("normal_raw_similarity", 0.0))),
                        attribute_match=float(top_pair.get("normal_attribute_match", 0.5) or 0.5),
                        behavior_consistency=float(top_pair.get("normal_behavior_consistency", 0.5) or 0.5),
                        margin=float(top_pair.get("normal_donor_margin", 0.0)),
                        gate_decision=str(top_pair.get("normal_gate_decision", "pass")),
                        source_tag=str(top_pair.get("normal_tag", "default")),
                    ),
                    build_route_audit_payload(
                        target_class=cls_name_l,
                        role="abnormal",
                        selected_source_class=top_pair.get("abnormal_source"),
                        donor_score=float(top_pair.get("abnormal_score", 0.0)),
                        semantic_sim=float(top_pair.get("abnormal_semantic_sim", top_pair.get("abnormal_raw_similarity", 0.0))),
                        attribute_match=float(top_pair.get("abnormal_attribute_match", 0.5) or 0.5),
                        behavior_consistency=float(top_pair.get("abnormal_behavior_consistency", 0.5) or 0.5),
                        margin=float(top_pair.get("abnormal_donor_margin", 0.0)),
                        gate_decision=str(top_pair.get("abnormal_gate_decision", "pass")),
                        source_tag=str(top_pair.get("abnormal_tag", "default")),
                    ),
                ]
                return normal_prompt, abnormal_prompt, multi_out.get("source", "dual:default/default"), {
                    "mode": "dual",
                    "target_category": cls_name_l,
                    "normal_source_category": top_pair.get("normal_source", cls_name_l),
                    "abnormal_source_category": top_pair.get("abnormal_source", cls_name_l),
                    "normal_resolution": top_pair.get("normal_tag", "default"),
                    "abnormal_resolution": top_pair.get("abnormal_tag", "default"),
                    "route_debug": route_debug,
                    "route_audit": route_audit,
                }
        normal_prompt, hit_n, src_n = resolver(
            cls_name_l, role="normal",
            default_prompt=f"X normal {cls_name_l}",
            allow_role_fallback=allow_role_fallback,
        )
        abnormal_prompt, hit_a, src_a = resolver(
            cls_name_l, role="abnormal",
            default_prompt=f"X abnormal {cls_name_l}",
            allow_role_fallback=allow_role_fallback,
        )
        return normal_prompt, abnormal_prompt, f"dual:{src_n}/{src_a}", {
            "mode": "dual",
            "target_category": cls_name_l,
            "normal_source_category": hit_n[1] if hit_n is not None else cls_name_l,
            "abnormal_source_category": hit_a[1] if hit_a is not None else cls_name_l,
            "normal_resolution": src_n,
            "abnormal_resolution": src_a,
        }

    shared_prompt, hit_s, src_s = resolver(
        cls_name_l, role="shared",
        default_prompt=f"X {cls_name_l}",
        allow_role_fallback=allow_role_fallback,
    )
    return shared_prompt, shared_prompt, f"shared:{src_s}", {
        "mode": "shared",
        "target_category": cls_name_l,
        "shared_source_category": hit_s[1] if hit_s is not None else cls_name_l,
        "shared_resolution": src_s,
    }


def resolve_icsr_pairs(
    args,
    evo_optimizer,
    cls_name_l: str,
    query_feat: torch.Tensor,
    route_config: Optional[RouteRuntimeConfig] = None,
) -> Dict[str, Any]:
    """Thin wrapper that forwards ICSR args to EvoPromptOptimizer.resolve_icsr."""
    del route_config  # not needed by resolve_icsr itself; kept for signature symmetry
    return evo_optimizer.resolve_icsr(
        name=cls_name_l,
        query_feat=query_feat,
        topk=int(getattr(args, "icsr_topk", 3)),
        tau=float(getattr(args, "icsr_tau", 0.1)),
        gate_entropy_threshold=float(getattr(args, "icsr_gate_entropy_threshold", 0.85)),
        min_sim=float(getattr(args, "icsr_min_sim", 0.15)),
        min_margin=float(getattr(args, "icsr_min_margin", 0.02)),
    )


def prepare_icsr_prompt_map(
    evo_optimizer,
    *,
    template: str,
    templates: Optional[Sequence[str]] = None,
    use_evo: bool = False,
) -> Tuple[Dict[str, List[str]], Dict[str, Any]]:
    """Resolve the per-class text list that feeds :func:`build_icsr_bank`.

    This layer handles all data-source decisions: which source classes are
    eligible, and what text(s) represent each class. The builder downstream only
    encodes/aggregates/normalizes, no cache access.

    Decision rules (mutually exclusive content sources):
      - ``use_evo=True``: use ``cache[("normal", cls)]`` and
        ``cache[("abnormal", cls)]`` prompt strings. If either evo entry is
        missing/empty, that class falls back to the template path (logged via
        the returned ``info['evo_fallback']`` list; wiring emits the warning).
      - ``templates`` non-None: each template formatted with ``cls``.
      - otherwise: single ``template.format(cls)``.

    Role completeness: classes lacking both ``("normal", cls)`` and
    ``("abnormal", cls)`` keys are excluded entirely (preserves legacy behavior
    for template/ensemble paths). Evo fallback only triggers when the KEY
    exists but the VALUE is empty/non-str.

    Returns:
      prompt_map: ``{cls: [text, ...]}`` — non-empty text list per included class.
      info: ``{'source': str, 'evo_fallback': List[str]}``. ``source`` is one of
        ``{'evo_cache_both', 'clip_ensemble', 'single_template'}`` and reflects
        what *most* classes used; per-class fallback classes are listed in
        ``evo_fallback`` (always empty when ``use_evo=False``).

    Raises:
      ValueError: if ``templates`` is an empty (but not None) sequence.
    """
    if templates is not None and len(templates) == 0:
        raise ValueError("prepare_icsr_prompt_map: `templates` must be None or non-empty")

    src_classes = sorted({
        k[1]
        for k in evo_optimizer.cache.keys()
        if isinstance(k, tuple) and len(k) == 2 and k[1]
    })
    template_list: Sequence[str] = tuple(templates) if templates else (template,)

    prompt_map: Dict[str, List[str]] = {}
    evo_fallback: List[str] = []

    for cls in src_classes:
        if ("normal", cls) not in evo_optimizer.cache:
            continue
        if ("abnormal", cls) not in evo_optimizer.cache:
            continue

        if use_evo:
            n_text = evo_optimizer.cache.get(("normal", cls))
            a_text = evo_optimizer.cache.get(("abnormal", cls))
            if isinstance(n_text, str) and n_text and isinstance(a_text, str) and a_text:
                prompt_map[cls] = [n_text, a_text]
                continue
            evo_fallback.append(cls)

        prompt_map[cls] = [t.format(cls) for t in template_list]

    if use_evo:
        source = "evo_cache_both"
    elif templates:
        source = "clip_ensemble"
    else:
        source = "single_template"

    return prompt_map, {"source": source, "evo_fallback": evo_fallback}


def build_icsr_bank(
    encoder,
    prompts_per_class: Dict[str, Sequence[str]],
    *,
    center: bool = False,
) -> Dict[str, torch.Tensor]:
    """Encode per-class prompts into a single L2-normalized bank vector per class.

    Pure math: for each ``(cls, texts)`` entry, encode all texts, L2-normalize
    each row, mean-pool, L2-renormalize. Optional bank-level centering.

    Args:
        encoder: callable mapping ``List[str] -> Tensor[N, D]`` (N = len(texts)).
        prompts_per_class: mapping ``cls -> [text, ...]`` produced upstream by
            :func:`prepare_icsr_prompt_map`. Empty text lists are skipped.
        center: if True and bank has >1 entry, subtract bank mean then
            L2-renormalize. On K=1 banks centering is a no-op (skipped with a
            warning) since subtracting the singleton's own vector yields zero.

    Returns ``{}`` when no class yields a valid embedding.
    """
    bank: Dict[str, torch.Tensor] = {}
    for cls, texts in prompts_per_class.items():
        if not texts:
            continue
        feat = encoder(list(texts))
        if feat is None:
            continue
        feat_cpu = feat.detach().cpu() if hasattr(feat, "detach") else torch.as_tensor(feat).cpu()
        feat_cpu = feat_cpu.float()
        feat_norm = F.normalize(feat_cpu, dim=-1, eps=1e-12)
        pooled = feat_norm.mean(dim=0)
        pooled = F.normalize(pooled, dim=-1, eps=1e-12)
        bank[cls] = pooled.contiguous()

    if center and len(bank) > 1:
        stacked = torch.stack(list(bank.values()), dim=0)
        mean_vec = stacked.mean(dim=0, keepdim=True)
        centered = F.normalize(stacked - mean_vec, dim=-1, eps=1e-12)
        for idx, cls in enumerate(bank.keys()):
            bank[cls] = centered[idx].contiguous()
    elif center and len(bank) == 1:
        logging.getLogger(__name__).warning(
            "ICSR bank center requested but K=1; skipping centering to avoid zero-vector output."
        )

    return bank


def _to_float_or_none(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def apply_safe_switch_decision(
    evo_optimizer: EvoPromptOptimizer,
    cls_name_l: str,
    normal_prompt: str,
    abnormal_prompt: str,
    source: str,
    route_config: RouteRuntimeConfig,
    source_category: Optional[str] = None,
) -> Tuple[str, str, str]:
    """Supplementary safe-switch gate: metadata-driven fallback/mix tag."""
    if not route_config.enable_safe_switch:
        return normal_prompt, abnormal_prompt, source

    default_normal = f"X normal {cls_name_l}"
    default_abnormal = f"X abnormal {cls_name_l}"
    getter = getattr(evo_optimizer, "get_rule_metadata", None)
    if not callable(getter):
        return default_normal, default_abnormal, f"{source}+safe_default"

    # For cross-domain transfer, metadata is keyed by the training category name, so look it up via source_category
    meta_key = source_category or cls_name_l
    meta = getter(meta_key, "pair")
    if not isinstance(meta, dict):
        return default_normal, default_abnormal, f"{source}+safe_default"

    gain_src = _to_float_or_none(meta.get("gain_src"))
    gain_cross = _to_float_or_none(meta.get("gain_cross"))
    score_std = _to_float_or_none(meta.get("score_std"))

    # gain_src is mandatory; gain_cross and score_std are optional
    # (absent when rules were trained without CCTO)
    if gain_src is None:
        return default_normal, default_abnormal, f"{source}+safe_default"

    # ── Source-only gate (always checked) ──
    if gain_src < route_config.safe_switch_min_gain_src:
        return default_normal, default_abnormal, f"{source}+safe_default"

    # ── Cross-category & std gates (only when metadata available) ──
    has_cross = gain_cross is not None
    has_std = score_std is not None

    if has_cross and gain_cross < route_config.safe_switch_min_gain_cross:
        return default_normal, default_abnormal, f"{source}+safe_default"
    if has_std and score_std > route_config.safe_switch_max_score_std:
        return default_normal, default_abnormal, f"{source}+safe_default"

    if route_config.safe_switch_enable_mix:
        need_mix = (
            (has_cross and gain_cross < route_config.safe_switch_mix_gain_cross)
            or (has_std and score_std > route_config.safe_switch_mix_score_std)
        )
        if need_mix:
            return normal_prompt, abnormal_prompt, f"{source}+safe_mix"

    return normal_prompt, abnormal_prompt, source


def _to_1d_unit(feat: Any) -> Optional[np.ndarray]:
    if feat is None:
        return None
    if isinstance(feat, torch.Tensor):
        if feat.ndim >= 2:
            feat = feat[0]
        vec = feat.detach().float().cpu().numpy().reshape(-1).astype(np.float32)
    else:
        vec = np.asarray(feat, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vec))
    if norm <= 0:
        return None
    return (vec / norm).astype(np.float32)


def _load_shared_bank(args, evo_optimizer: EvoPromptOptimizer, logger=None) -> List[Dict[str, Any]]:
    bank: List[Dict[str, Any]] = []
    bank_path = getattr(args, "shared_bank_path", "")
    if bank_path and os.path.exists(bank_path):
        try:
            with open(bank_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                if isinstance(raw.get("pairs"), list):
                    bank = list(raw["pairs"])
                elif isinstance(raw.get("shared_bank"), list):
                    bank = list(raw["shared_bank"])
            elif isinstance(raw, list):
                bank = list(raw)
            if logger:
                logger.info("Loaded shared bank: %s (%d entries)", bank_path, len(bank))
        except Exception as exc:
            if logger:
                logger.warning("Failed to load shared bank '%s': %s", bank_path, exc)

    if not bank:
        role_to_name = {}
        for key, prompt in evo_optimizer.cache.items():
            if not (isinstance(key, tuple) and len(key) == 2):
                continue
            role, name = key
            role_to_name.setdefault(name, {})[role] = prompt
        for name, item in sorted(role_to_name.items()):
            if "normal" in item and "abnormal" in item:
                bank.append(
                    {
                        "name": name,
                        "normal_prompt": item["normal"],
                        "abnormal_prompt": item["abnormal"],
                        "source": "cache_bootstrap",
                    }
                )
            elif "shared" in item:
                bank.append(
                    {
                        "name": name,
                        "normal_prompt": item["shared"],
                        "abnormal_prompt": item["shared"],
                        "source": "cache_bootstrap",
                    }
                )
        if logger:
            logger.info("Shared bank bootstrap from cache: %d entries", len(bank))

    dedup = []
    seen = set()
    for item in bank:
        if not isinstance(item, dict):
            continue
        n = str(item.get("normal_prompt", "")).strip()
        a = str(item.get("abnormal_prompt", "")).strip()
        if not n or not a:
            continue
        key = (n, a)
        if key in seen:
            continue
        seen.add(key)
        out = dict(item)
        out["normal_prompt"] = n
        out["abnormal_prompt"] = a
        out["name"] = str(item.get("name", "")).strip()
        dedup.append(out)

    shared_bank_size = int(getattr(args, "shared_bank_size", 4))
    if shared_bank_size > 0:
        dedup = dedup[:shared_bank_size]
    return dedup


def _extract_image_score_and_pixel_map(
    infer_result: Dict[str, Any],
    need_pixel: bool,
    args,
) -> Tuple[float, Optional[np.ndarray]]:
    image_scores = infer_result.get("image_scores", None)
    if image_scores is None:
        raise ValueError("scorer.infer() returned no image_scores")

    if isinstance(image_scores, torch.Tensor):
        score_val = image_scores.detach().float().cpu().numpy().reshape(-1)
        image_score = float(score_val[0]) if score_val.size > 0 else 0.0
    else:
        image_score = float(np.asarray(image_scores).reshape(-1)[0])

    pixel_maps = infer_result.get("pixel_maps", None)
    if need_pixel and pixel_maps is None:
        raise ValueError(
            f"Scorer returned pixel_maps=None but dataset '{args.dataset}' requires pixel-level evaluation."
        )
    if pixel_maps is None:
        return image_score, None

    if isinstance(pixel_maps, torch.Tensor):
        arr = pixel_maps.detach().float().cpu().numpy()
    else:
        arr = np.asarray(pixel_maps)
    pixel_map = arr[0] if arr.ndim >= 3 else arr
    if pixel_map is None:
        return image_score, None
    return image_score, pixel_map.astype(np.float32)


def _extract_single_image_logits(
    infer_result: Dict[str, Any],
    prompt_num: int,
) -> np.ndarray:
    image_logits = infer_result.get("image_logits", None)
    if image_logits is None:
        raise ValueError(
            "--dump_image_logits requires scorer output 'image_logits'. "
            "Use scorer_type=prompt_bank without score-level branches that replace infer_result."
        )

    if isinstance(image_logits, torch.Tensor):
        arr = image_logits.detach().float().cpu().numpy()
    else:
        arr = np.asarray(image_logits, dtype=np.float32)

    if arr.ndim > 1:
        arr = arr.reshape(arr.shape[0], -1)[0]
    else:
        arr = arr.reshape(-1)

    expected = 2 * int(prompt_num)
    if arr.size != expected:
        raw_shape = tuple(image_logits.shape) if hasattr(image_logits, "shape") else tuple(arr.shape)
        raise ValueError(
            f"image_logits must contain exactly 2*prompt_num={expected} values "
            f"for one image; got shape={raw_shape}."
        )
    return arr.astype(np.float32, copy=False)


def _infer_icsr_aggregation(
    active_scorer,
    prepared,
    image,
    icsr_pairs,
    stage,
    logger,
    *,
    need_pixel: bool,
    args,
) -> Dict[str, Any]:
    """Weighted ICSR score-level aggregation.

    Byte-compatible port of the inlined `elif icsr_pairs is not None:` branch.
    `need_pixel` and `args` are keyword-only and MUST come from the caller.
    """
    agg_score_icsr: float = 0.0
    agg_map_icsr: Optional[np.ndarray] = None
    _eval_pair = getattr(active_scorer, "evaluate_prompt_pair", None)
    if not callable(_eval_pair):
        logger.warning(
            "active_scorer (%s) lacks evaluate_prompt_pair; ICSR falls back "
            "to infer() with re-encoding.",
            type(active_scorer).__name__,
        )
    for pair_idx, (np_text, ap_text, weight) in enumerate(icsr_pairs):
        if callable(_eval_pair) and prepared is not None:
            try:
                pair_result = _eval_pair(prepared, np_text, ap_text, stage=stage)
            except NotImplementedError:
                if pair_idx == 0:
                    logger.warning(
                        "active_scorer.evaluate_prompt_pair not implemented; "
                        "falling back to infer() for ICSR aggregation.",
                    )
                try:
                    pair_result = active_scorer.infer(image, np_text, ap_text, stage=stage)
                except TypeError:
                    pair_result = active_scorer.infer(image, np_text, ap_text)
        else:
            try:
                pair_result = active_scorer.infer(image, np_text, ap_text, stage=stage)
            except TypeError:
                pair_result = active_scorer.infer(image, np_text, ap_text)
        pair_score, pair_map = _extract_image_score_and_pixel_map(
            infer_result=pair_result, need_pixel=need_pixel, args=args,
        )
        agg_score_icsr += float(weight) * float(pair_score)
        if pair_map is not None:
            contrib = float(weight) * pair_map.astype(np.float32)
            agg_map_icsr = contrib if agg_map_icsr is None else agg_map_icsr + contrib
    return {
        "image_scores": np.asarray([agg_score_icsr], dtype=np.float32),
        "pixel_maps": (
            agg_map_icsr[np.newaxis, ...] if agg_map_icsr is not None else None
        ),
    }


def _infer_single_pair(
    active_scorer,
    prepared,
    image,
    normal_prompt: str,
    abnormal_prompt: str,
    stage,
    logger,
) -> Dict[str, Any]:
    """Single prompt-pair inference with evaluate_prompt_pair reuse.

    Byte-compatible port of the inlined `else` branch. Returns the raw
    `infer_result` dict; caller runs `_extract_image_score_and_pixel_map`.
    """
    if prepared is not None and hasattr(active_scorer, "evaluate_prompt_pair"):
        try:
            return active_scorer.evaluate_prompt_pair(
                prepared, normal_prompt, abnormal_prompt, stage=stage,
            )
        except NotImplementedError:
            try:
                return active_scorer.infer(image, normal_prompt, abnormal_prompt, stage=stage)
            except TypeError:
                return active_scorer.infer(image, normal_prompt, abnormal_prompt)
    try:
        return active_scorer.infer(image, normal_prompt, abnormal_prompt, stage=stage)
    except TypeError:
        logger.warning("scorer.infer() does not accept 'stage' param, falling back to 3-arg call")
        return active_scorer.infer(image, normal_prompt, abnormal_prompt)


def compute_and_report_metrics(results, obj_list, logger, args):
    """Dispatch metric computation."""
    cfg = _load_score_fusion_config(getattr(args, "score_fusion_config", "") or "")
    if cfg is not None and args.dataset in DATASETS_ONLY_CLASSIFICATION:
        sys.exit(
            f"score_fusion_config: dataset {args.dataset!r} is classification-only; "
            f"score-fusion config is not supported for this dataset."
        )
    if cfg is not None:
        # TARGET_EVAL audit is emitted only once we know the config will be applied.
        config_hash = _hashlib.sha256(
            json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        logger.info(
            f"TARGET_EVAL uses config_hash={config_hash} direction={cfg['direction']} "
            f"source_dataset={cfg['source_dataset']} target_dataset={cfg['target_dataset']} "
            f"grid_hash={cfg['grid_hash']}"
        )
    fa = getattr(args, "fusion_alpha", 0.5)
    if args.dataset in DATASETS_ONLY_CLASSIFICATION:
        calcuate_metric_image(results, obj_list, logger, alpha=fa, sigm=args.pixel_sigma, args=args)
    else:
        calcuate_metric_pixel(
            results, obj_list, logger,
            alpha=fa, sigm=args.pixel_sigma, args=args,
            score_fusion_config=cfg,
        )


# ---------------------------------------------------------------------------
# Prototype bank loading
# ---------------------------------------------------------------------------

def _load_proto_banks(args, scorer, device, logger) -> Dict[str, Any]:
    """Load the prototype bank and build a PrototypeAugmentedScorer per category.

    :param scorer: an already-built base scorer instance (reused, not recreated)
    Returns a {category_lower: PrototypeAugmentedScorer} dict.
    """
    from models.prototype_bank import (
        PrototypeAugmentedScorer,
        load_all_category_names,
        load_category_bank,
    )

    bank_path = getattr(args, "proto_bank_path", "")
    if not bank_path:
        # try under save_path
        candidate = os.path.join(getattr(args, "save_path", "."), "prototype_bank.pt")
        if os.path.exists(candidate):
            bank_path = candidate
        else:
            # try the same directory as evo_rules_path
            evo_path = getattr(args, "evo_rules_path", "")
            if evo_path:
                candidate = os.path.join(os.path.dirname(evo_path), "prototype_bank.pt")
                if os.path.exists(candidate):
                    bank_path = candidate

    if not bank_path or not os.path.exists(bank_path):
        logger.warning(
            "Prototype bank enabled but file not found (tried: %s); "
            "falling back to text-only scoring",
            bank_path or "<not specified>",
        )
        return {}

    logger.info("Loading prototype bank: %s", bank_path)
    categories = load_all_category_names(bank_path)
    logger.info("  Categories in bank: %s", categories)

    banks: Dict[str, Any] = {}
    for cat in categories:
        result = load_category_bank(bank_path, cat, device=device)
        if result is None:
            continue
        bank, alpha_img, alpha_px = result

        # Only override the stored value when alpha is given explicitly on the CLI; otherwise reuse the Stage2 grid-search result
        cli_alpha_img = getattr(args, "proto_alpha_image", None)
        cli_alpha_px = getattr(args, "proto_alpha_pixel", None)
        if cli_alpha_img is None:
            cli_alpha_img = float(alpha_img)
        else:
            cli_alpha_img = float(cli_alpha_img)
        if cli_alpha_px is None:
            cli_alpha_px = float(alpha_px)
        else:
            cli_alpha_px = float(cli_alpha_px)

        banks[cat.lower()] = PrototypeAugmentedScorer(
            base_scorer=scorer,
            prototype_bank=bank,
            alpha_image=cli_alpha_img,
            alpha_pixel=cli_alpha_px,
            image_size=int(getattr(args, "image_size", 518)),
        )

    logger.info("Prototype bank loaded: %d categories augmented", len(banks))
    return banks


def _load_shared_proto_bank(args, scorer, device, logger):
    """Load the shared prototype bank and return a single PrototypeAugmentedScorer (shared by all categories).

    Lookup order:
      1. --shared_proto_bank_path, if given explicitly
      2. save_path/shared_prototype_bank.pt
      3. shared_prototype_bank.pt in the same directory as evo_rules_path

    Alpha source:
      1. CLI --proto_alpha_image / --proto_alpha_pixel (if given)
      2. the grid-search result stored in shared_prototype_bank_meta.json

    :returns: a PrototypeAugmentedScorer instance, or None if loading fails
    """
    from models.prototype_bank import PrototypeAugmentedScorer, PrototypeBank

    # -- locate the bank file --
    bank_path = getattr(args, "shared_proto_bank_path", "")
    if not bank_path or not os.path.exists(bank_path):
        candidates = []
        save_path = getattr(args, "save_path", "")
        if save_path:
            candidates.append(os.path.join(save_path, "shared_prototype_bank.pt"))
        evo_path = getattr(args, "evo_rules_path", "")
        if evo_path:
            candidates.append(
                os.path.join(os.path.dirname(evo_path), "shared_prototype_bank.pt")
            )
        for cand in candidates:
            if os.path.exists(cand):
                bank_path = cand
                break

    if not bank_path or not os.path.exists(bank_path):
        logger.warning(
            "Shared prototype bank enabled but file not found (tried: %s); "
            "falling back to text-only scoring",
            bank_path or "<not specified>",
        )
        return None

    logger.info("Loading shared prototype bank: %s", bank_path)
    bank = PrototypeBank.load(bank_path, device=device)
    logger.info("  Shared bank summary: %s", bank.summary)

    # -- look up alpha (meta json) --
    meta_path = bank_path.replace(".pt", "_meta.json")
    alpha_img = 0.3  # default
    alpha_px = 0.5
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            alpha_img = float(meta.get("alpha_image", alpha_img))
            alpha_px = float(meta.get("alpha_pixel", alpha_px))
            logger.info(
                "  Shared bank meta loaded: alpha_image=%.3f, alpha_pixel=%.3f",
                alpha_img, alpha_px,
            )
        except Exception as exc:
            logger.warning("Failed to load shared bank meta: %s", exc)
    else:
        logger.info("  No meta file found (%s), using default alpha", meta_path)

    # explicit CLI override
    cli_alpha_img = getattr(args, "proto_alpha_image", None)
    cli_alpha_px = getattr(args, "proto_alpha_pixel", None)
    if cli_alpha_img is not None:
        alpha_img = float(cli_alpha_img)
        logger.info("  CLI override: alpha_image=%.3f", alpha_img)
    if cli_alpha_px is not None:
        alpha_px = float(cli_alpha_px)
        logger.info("  CLI override: alpha_pixel=%.3f", alpha_px)

    augmented = PrototypeAugmentedScorer(
        base_scorer=scorer,
        prototype_bank=bank,
        alpha_image=alpha_img,
        alpha_pixel=alpha_px,
        image_size=int(getattr(args, "image_size", 518)),
    )
    logger.info("Shared prototype bank loaded successfully (all categories share one bank)")
    return augmented


# ---------------------------------------------------------------------------
# Main test loop
# ---------------------------------------------------------------------------

def run_universal_test(args, route_config: Optional[RouteRuntimeConfig] = None):
    route_config = route_config or STRICT_ROUTE_CONFIG
    _enforce_transfer_mainline(args, route_config=route_config)
    _load_model_config(args)

    if torch.cuda.is_available():
        torch.cuda.set_device(args.device_id)
    reset_cuda_peak_memory(args.device_id)
    device = torch.device(
        f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu"
    )

    logger = setup_logger(args.save_path)
    _enforce_transfer_mainline(args, logger, route_config=route_config)
    logger.info("ROUTE_PROFILE: %s", route_config.profile_name)
    logger.info(
        "Route controls: role_fallback=%s semantic=%s template=%s safe_switch=%s",
        route_config.allow_role_fallback,
        route_config.enable_semantic_fallback,
        route_config.enable_template_transfer,
        route_config.enable_safe_switch,
    )
    for arg in vars(args):
        logger.info("%s: %s", arg, getattr(args, arg))

    # build the scorer
    scorer = build_scorer(args, device)
    logger.info("Scorer: %s", type(scorer).__name__)

    # load the prototype bank
    # priority: shared prototype bank > per-category prototype bank
    shared_proto_scorer = None
    proto_banks_by_category: Dict[str, Any] = {}
    if getattr(args, "shared_prototype_bank", False):
        shared_proto_scorer = _load_shared_proto_bank(args, scorer, device, logger)
        if shared_proto_scorer is not None:
            logger.info("Using shared prototype bank — per-category routing disabled")
    elif getattr(args, "enable_prototype_bank", False):
        proto_banks_by_category = _load_proto_banks(args, scorer, device, logger)

    # load EvoPrompt rules
    evo_optimizer = setup_evo_optimizer_for_test(
        args, device, scorer, logger, route_config=route_config
    )

    # load ESPR refined embeddings
    espr_embeddings: Dict[str, torch.Tensor] = {}
    _espr_path = getattr(args, "espr_embeddings_path", "")
    if _espr_path:
        if not os.path.exists(_espr_path):
            raise FileNotFoundError(
                f"--espr_embeddings_path specified but file not found: {_espr_path}"
            )
        if not callable(getattr(scorer, "_infer_with_prepared_features", None)):
            raise ValueError(
                "--espr_embeddings_path requires the prompt_bank scorer because ESPR "
                f"embeddings are internal prompt-bank text embeddings; got {type(scorer).__name__}"
            )
        _loaded_espr = torch.load(_espr_path, map_location=device)
        if not isinstance(_loaded_espr, dict):
            raise ValueError(
                f"ESPR embeddings file must contain dict[str, Tensor], got {type(_loaded_espr).__name__}"
            )
        espr_embeddings = {str(k).lower(): v for k, v in _loaded_espr.items() if isinstance(v, torch.Tensor)}
        if len(espr_embeddings) != len(_loaded_espr):
            bad_keys = [str(k) for k, v in _loaded_espr.items() if not isinstance(v, torch.Tensor)]
            raise ValueError(
                "ESPR embeddings file must contain only Tensor values; "
                f"non-tensor keys: {bad_keys[:5]}"
            )
        logger.info("ESPR embeddings loaded: %s (%d categories)", _espr_path, len(espr_embeddings))

    # prepare the dataset
    preprocess_test = _transform_test(args.image_size)
    make_dataset = Makedataset(
        train_data_path=args.data_path,
        preprocess_test=preprocess_test,
        mode="test",
        image_size=args.image_size,
    )
    product_list_arg = str(getattr(args, "product_list", "") or "").strip()
    product_list = [item.strip() for item in product_list_arg.split(",") if item.strip()] or None
    test_dataloader, obj_list = make_dataset.mask_dataset(
        name=args.dataset,
        product_list=product_list,
        batchsize=1,
        shuf=False,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )
    logger.info("Test split: fixed test (mainline)")
    logger.info("Dataset: %s", args.dataset)
    logger.info("Product list filter: %s", product_list)
    logger.info("Categories: %s", obj_list)

    # -- cross-domain prototype bank matching ------------------------------
    # When the proto bank comes from the source domain (e.g. VisA) but the test set is the
    # target domain (e.g. MVTec), category names do not match exactly. CLIP text-feature
    # similarity is used to find the nearest source category for each target category.
    if proto_banks_by_category:
        _missing_cats = [
            c.lower() for c in obj_list
            if c.lower() not in proto_banks_by_category
        ]
        if _missing_cats:
            logger.info(
                "Cross-domain proto bank: %d/%d target categories missing exact match, "
                "building CLIP similarity mapping...",
                len(_missing_cats), len(obj_list),
            )
            # build the CLIP text encoder
            _clip_model = getattr(scorer, "model_clip", None)
            _tok_fn = getattr(scorer, "tokenizer", None)
            _ext_adapter2 = getattr(scorer, "adapter", None)
            _encode_texts = None
            _route_tmpl = route_config.semantic_fallback_template or ROUTE_TEXT_TEMPLATE
            if _ext_adapter2 is not None and hasattr(_ext_adapter2, "encode_text"):
                def _encode_texts(texts):
                    with torch.no_grad():
                        return _ext_adapter2.encode_text(texts).cpu()
            elif _clip_model is not None and _tok_fn is not None:
                def _encode_texts(texts):
                    with torch.no_grad():
                        tokens = _tok_fn(texts)
                        if hasattr(tokens, "to"):
                            tokens = tokens.to(device)
                        elif isinstance(tokens, dict):
                            tokens = {k: v.to(device) for k, v in tokens.items()}
                        else:
                            tokens = torch.as_tensor(tokens, device=device)
                        feats = _clip_model.encode_text(tokens)
                        return F.normalize(feats.float(), dim=-1, eps=1e-12).cpu()
            else:
                logger.warning(
                    "Cannot build CLIP encoder for cross-domain proto matching; "
                    "%d categories will use base scorer", len(_missing_cats),
                )
            if _encode_texts is not None:
                # compute CLIP embeddings for the source categories present in the bank
                _bank_cats = list(proto_banks_by_category.keys())
                _bank_embs = _encode_texts(
                    [_route_tmpl.format(c) for c in _bank_cats]
                )  # [N_src, D]

                # for each missing target category, compute similarity and take the nearest neighbour
                _query_embs = _encode_texts(
                    [_route_tmpl.format(c) for c in _missing_cats]
                )  # [N_miss, D]
                _sims = _query_embs @ _bank_embs.T  # [N_miss, N_src]

                for i, tgt_cat in enumerate(_missing_cats):
                    best_idx = int(_sims[i].argmax())
                    best_src = _bank_cats[best_idx]
                    best_sim = float(_sims[i, best_idx])
                    proto_banks_by_category[tgt_cat] = proto_banks_by_category[best_src]
                    logger.info(
                        "  proto bank match: '%s' → '%s' (cosine=%.4f)",
                        tgt_cat, best_src, best_sim,
                    )
                logger.info(
                    "Cross-domain mapping complete: %d categories now covered",
                    len(proto_banks_by_category),
                )

    stage = 2
    checkpoint_path = getattr(args, "checkpoint_path", "")
    if checkpoint_path and "epoch_post" in checkpoint_path:
        try:
            parsed_stage = int(checkpoint_path[:-4].split("_")[-1])
            if parsed_stage in (1, 2):
                stage = parsed_stage
            else:
                logger.warning(
                    "Parsed unsupported stage=%s from checkpoint '%s'; defaulting to stage=2",
                    parsed_stage,
                    checkpoint_path,
                )
        except Exception:
            stage = 2
            logger.warning("Cannot parse stage from checkpoint '%s', defaulting to stage=2", checkpoint_path)

    # The stage1_final checkpoint never trained fuse/image_mapping, so it must use the stage=1 forward path
    if checkpoint_path and "stage1_final" in os.path.basename(checkpoint_path) and stage == 2:
        stage = 1
        logger.warning(
            "Detected stage1_final checkpoint (fuse/image_mapping untrained), forcing stage=1"
        )

    need_pixel = args.dataset not in DATASETS_ONLY_CLASSIFICATION

    # result containers
    results = {
        "cls_names": [],
        "imgs_masks": [],
        "anomaly_maps": [],
        "pr_sp": [],
        "gt_sp": [],
        "path": [],
        "image_logits": [],
    }

    warned_missing_cls = set()
    warned_espr_missing_cls = set()
    _warned_icsr_bypass_official = False
    shared_bank = []
    class_route_cache: Dict[str, Optional[np.ndarray]] = {}
    shared_prepared_prompt_cache: Dict[Tuple[str, str, int], Any] = {}
    warned_no_prepared = False
    warned_infer_prepared = False
    icsr_audit = None
    soft_mix_opts: Optional[SoftMixOptions] = None

    # Build the route encoder: prefer semantic_embed_fn, else build it from the scorer's CLIP model
    _route_encoder = None

    def _build_route_encoder():
        """Build a route encoder from the scorer's CLIP model (used only for shared-bank routing)."""
        _ext_adapter3 = getattr(scorer, "adapter", None)
        if _ext_adapter3 is not None and hasattr(_ext_adapter3, "encode_text"):
            def _encode(texts):
                with torch.no_grad():
                    return _ext_adapter3.encode_text(texts).cpu()
            return _encode

        clip_model = getattr(scorer, "model_clip", None)
        tok_fn = getattr(scorer, "tokenizer", None)
        if clip_model is None or tok_fn is None:
            return None

        def _encode(texts):
            with torch.no_grad():
                tokens = tok_fn(texts)
                if hasattr(tokens, "to"):
                    tokens = tokens.to(device)
                elif isinstance(tokens, dict):
                    tokens = {k: v.to(device) for k, v in tokens.items()}
                else:
                    tokens = torch.as_tensor(tokens, device=device)
                feats = clip_model.encode_text(tokens)
                return F.normalize(feats.float(), dim=-1, eps=1e-12).cpu()

        return _encode

    if getattr(args, "enable_shared_bank", False):
        shared_bank = _load_shared_bank(args, evo_optimizer, logger)
        _route_encoder = getattr(evo_optimizer, "semantic_embed_fn", None)
        if _route_encoder is None:
            _route_encoder = _build_route_encoder()
            if _route_encoder is not None:
                logger.info("Route encoder built from scorer CLIP model")
            else:
                logger.warning("No route encoder available; shared bank routing will use uniform weights")
        route_template = route_config.semantic_fallback_template or ROUTE_TEXT_TEMPLATE
        for item in shared_bank:
            name_hint = item.get("name", "").strip()
            route_text = (
                route_template.format(name_hint)
                if name_hint else route_template.format(item["abnormal_prompt"].replace("X ", "").strip())
            )
            if _route_encoder is None:
                item["_route_feat"] = None
                continue
            try:
                item["_route_feat"] = _to_1d_unit(_route_encoder([route_text]))
            except Exception:
                item["_route_feat"] = None
        logger.info("Enable shared bank: entries=%d topk=%d", len(shared_bank), int(getattr(args, "shared_topk", 2)))

    # ---- ICSR bank build (once per run) ----
    if getattr(args, "enable_icsr", False):
        _ignored_flags = _compute_ignored_soft_mix_flags(
            soft_mix_enabled=bool(getattr(args, "icsr_soft_mix", False)),
        )
        if _ignored_flags:
            logger.warning(
                "soft_mix=true; the following hard-gate flags are IGNORED by this path: %s",
                _ignored_flags,
            )
        # Validate soft_mix options up front so unsupported range_sources
        # (e.g., 'adaptive') fail before any inference work runs.
        if bool(getattr(args, "icsr_soft_mix", False)):
            soft_mix_opts = SoftMixOptions(
                formula=args.icsr_alpha_formula,
                range_source=args.icsr_alpha_range_source,
            )
        if _route_encoder is None:
            _route_encoder = _build_route_encoder()
        if _route_encoder is None:
            logger.warning(
                "ICSR requested but no CLIP text encoder available; disabling ICSR for this run."
            )
            args.enable_icsr = False
        else:
            _icsr_template = (
                getattr(route_config, "semantic_fallback_template", None)
                or ROUTE_TEXT_TEMPLATE
            )
            _icsr_use_evo = bool(getattr(args, "icsr_bank_source_evo", False))
            _icsr_use_ensemble = bool(getattr(args, "icsr_bank_template_ensemble", False))
            _icsr_center = bool(getattr(args, "icsr_bank_center", False))
            if _icsr_use_evo and _icsr_use_ensemble:
                logger.warning(
                    "ICSR bank: --icsr_bank_source_evo overrides --icsr_bank_template_ensemble; "
                    "ensemble flag ignored for this run."
                )
            _icsr_templates = (
                ICSR_BANK_TEMPLATES_DEFAULT
                if (_icsr_use_ensemble and not _icsr_use_evo)
                else None
            )
            _prompt_map, _prompt_info = prepare_icsr_prompt_map(
                evo_optimizer,
                template=_icsr_template,
                templates=_icsr_templates,
                use_evo=_icsr_use_evo,
            )
            _bank = build_icsr_bank(
                encoder=_route_encoder,
                prompts_per_class=_prompt_map,
                center=_icsr_center,
            )
            if not _bank:
                logger.warning("ICSR bank is empty; disabling ICSR for this run.")
                args.enable_icsr = False
            else:
                evo_optimizer.set_icsr_bank(_bank)
                _source_tag = _prompt_info["source"]
                if _source_tag == "evo_cache_both":
                    _bank_template_source = f"evo_cache_both (n={len(_bank)})"
                elif _source_tag == "clip_ensemble":
                    _bank_template_source = (
                        f"clip_7_fixed (n={len(_icsr_templates)}, "
                        "overrides semantic_fallback_template)"
                    )
                else:
                    _bank_template_source = f"single_template={_icsr_template!r}"
                _fallback_cls = _prompt_info.get("evo_fallback", [])
                logger.info(
                    "ICSR bank built: %d source classes (template_source=%s, center=%s, evo_fallback_to_template=%d).",
                    len(_bank),
                    _bank_template_source,
                    _icsr_center,
                    len(_fallback_cls),
                )
                if _fallback_cls:
                    logger.warning(
                        "ICSR bank: %d classes missing evo role prompts; "
                        "used template fallback: %s",
                        len(_fallback_cls),
                        _fallback_cls,
                    )
                icsr_audit = create_icsr_audit()

    def _ensure_prepared_shared_prompt(normal_prompt: str, abnormal_prompt: str) -> Any:
        nonlocal warned_no_prepared
        cache_key = (normal_prompt, abnormal_prompt, int(stage))
        if cache_key in shared_prepared_prompt_cache:
            return shared_prepared_prompt_cache[cache_key]
        prep = None
        prepare_fn = getattr(scorer, "prepare_prompt_pair", None)
        if callable(prepare_fn):
            try:
                prep = prepare_fn(normal_prompt, abnormal_prompt, stage=stage)
            except Exception as exc:
                if not warned_no_prepared:
                    logger.warning("prepare_prompt_pair failed once; fallback to infer(): %s", exc)
                    warned_no_prepared = True
                prep = None
        elif not warned_no_prepared:
            logger.info("Scorer has no prepare_prompt_pair(); shared text pre-cache disabled")
            warned_no_prepared = True
        shared_prepared_prompt_cache[cache_key] = prep
        return prep

    def _select_shared_candidates(cls_name_l: str) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        if len(shared_bank) == 0:
            return [], np.zeros((0,), dtype=np.float32)

        topk = max(1, int(getattr(args, "shared_topk", 2)))
        topk = min(topk, len(shared_bank))
        route_temp = float(getattr(args, "shared_route_temp", 0.07))
        route_template = route_config.semantic_fallback_template or ROUTE_TEXT_TEMPLATE

        if cls_name_l not in class_route_cache:
            if _route_encoder is None:
                class_route_cache[cls_name_l] = None
            else:
                try:
                    class_route_cache[cls_name_l] = _to_1d_unit(
                        _route_encoder([route_template.format(cls_name_l)])
                    )
                except Exception:
                    class_route_cache[cls_name_l] = None

        query_feat = class_route_cache[cls_name_l]
        sims = []
        for item in shared_bank:
            cand_feat = item.get("_route_feat")
            if query_feat is None or cand_feat is None:
                sims.append(0.0)
            else:
                sims.append(float(np.dot(query_feat, cand_feat)))

        top_idx = sorted(range(len(shared_bank)), key=lambda i: sims[i], reverse=True)[:topk]
        selected = [shared_bank[i] for i in top_idx]
        weights = softmax_from_similarities([sims[i] for i in top_idx], temperature=route_temp)
        return selected, weights

    def _log_route_audit_once(cls_name_l: str, resolve_meta: Optional[Dict[str, Any]]) -> None:
        if cls_name_l in warned_missing_cls:
            return
        if not isinstance(resolve_meta, dict):
            return
        route_audit = resolve_meta.get("route_audit", []) or []
        for payload in route_audit:
            logger.info("ROUTE_AUDIT|%s", json.dumps(payload, sort_keys=True))

    def _resolve_espr_embedding(
        cls_name_l: str,
        resolve_meta: Optional[Dict[str, Any]],
    ) -> Optional[torch.Tensor]:
        if not espr_embeddings:
            return None

        def _lookup(key: Optional[str]) -> Optional[torch.Tensor]:
            if key is None:
                return None
            return espr_embeddings.get(str(key).lower())

        if isinstance(resolve_meta, dict):
            normal_src = str(resolve_meta.get("normal_source_category", cls_name_l)).lower()
            abnormal_src = str(resolve_meta.get("abnormal_source_category", normal_src)).lower()
        else:
            normal_src = abnormal_src = cls_name_l

        emb_n = _lookup(normal_src)
        emb_a = _lookup(abnormal_src)
        if emb_n is not None and emb_a is not None:
            emb_n = emb_n.to(device)
            emb_a = emb_a.to(device)
            if emb_n.ndim == 2:
                emb_n = emb_n.unsqueeze(0)
            if emb_a.ndim == 2:
                emb_a = emb_a.unsqueeze(0)
            if emb_n.ndim != 3 or emb_a.ndim != 3:
                raise ValueError(
                    "ESPR embeddings must be [1, 2*prompt_num, D] or [2*prompt_num, D]; "
                    f"got normal={tuple(emb_n.shape)} abnormal={tuple(emb_a.shape)}"
                )
            pn = int(getattr(args, "prompt_num", emb_n.shape[1] // 2))
            expected = 2 * pn
            if emb_n.shape[1] < expected or emb_a.shape[1] < expected:
                raise ValueError(
                    f"ESPR embedding prompt dimension mismatch: expected >= {expected}, "
                    f"got normal={emb_n.shape[1]} abnormal={emb_a.shape[1]}"
                )
            return torch.cat([emb_n[:, :pn, :], emb_a[:, pn:expected, :]], dim=1)

        emb_target = _lookup(cls_name_l)
        if emb_target is not None:
            return emb_target.to(device)

        if cls_name_l not in warned_espr_missing_cls:
            logger.warning(
                "ESPR embeddings missing for target '%s' (normal_src=%s abnormal_src=%s); "
                "falling back to text prompts",
                cls_name_l, normal_src, abnormal_src,
            )
            warned_espr_missing_cls.add(cls_name_l)
        return None

    if getattr(args, "enable_shared_bank", False) and len(shared_bank) > 0:
        for item in shared_bank:
            _ensure_prepared_shared_prompt(item["normal_prompt"], item["abnormal_prompt"])
        ready_count = sum(1 for v in shared_prepared_prompt_cache.values() if v is not None)
        logger.info(
            "shared prompt precompute done: total=%d prepared=%d",
            len(shared_prepared_prompt_cache), ready_count,
        )

    for items in tqdm(test_dataloader, desc="Testing"):
        image = items["img"].to(device)
        cls_name = items["cls_name"][0]
        cls_name_l = cls_name.lower()

        results["cls_names"].append(cls_name)
        results["gt_sp"].append(items["anomaly"].item())

        gt_mask = items["img_mask"]
        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
        results["imgs_masks"].append(gt_mask)

        # scorer choice: shared bank > per-category bank > base scorer
        if shared_proto_scorer is not None:
            active_scorer = shared_proto_scorer
        elif cls_name_l in proto_banks_by_category:
            active_scorer = proto_banks_by_category[cls_name_l]
        else:
            active_scorer = scorer

        use_official_baseline = (
            getattr(args, "scorer_type", "") == "external"
            and getattr(args, "external_adapter", "") == "anomalyclip"
            and not bool(getattr(evo_optimizer, "has_loaded_rules", False))
            and hasattr(active_scorer, "infer_official_baseline")
            and callable(getattr(active_scorer, "infer_official_baseline"))
        )

        icsr_pairs: Optional[List[Tuple[str, str, float]]] = None
        soft_mix_semantic: Optional[Tuple[str, str, str, Dict[str, Any]]] = None
        icsr_out: Optional[Dict[str, Any]] = None
        prepared = None
        query_feat_cpu: Optional[torch.Tensor] = None

        # ICSR one-shot bypass for incompatible modes (only official_baseline today)
        if getattr(args, "enable_icsr", False):
            if use_official_baseline:
                if not _warned_icsr_bypass_official:
                    logger.info("ICSR bypassed: official_baseline is active for this run.")
                    _warned_icsr_bypass_official = True
                icsr_active = False
            else:
                icsr_active = True
        else:
            icsr_active = False

        if icsr_active:
            image_features, patch_tokens = active_scorer.prepare_images(image)
            prepared = (image_features, patch_tokens)
            query_feat_cpu = F.normalize(
                image_features.float(), dim=-1, eps=1e-12,
            )[0].detach().cpu().contiguous()

        if use_official_baseline:
            # This path does not use resolved prompts; it is logged separately to keep the route evidence auditable.
            normal_prompt = f"X normal {cls_name_l}"
            abnormal_prompt = f"X abnormal {cls_name_l}"
            source = "external:official_baseline"
            if cls_name_l not in warned_missing_cls:
                logger.info("Class '%s' → %s", cls_name_l, source)
                warned_missing_cls.add(cls_name_l)
        elif icsr_active:
            icsr_out = resolve_icsr_pairs(
                args, evo_optimizer, cls_name_l,
                query_feat=query_feat_cpu, route_config=route_config,
            )
            if icsr_audit is not None:
                record_icsr_audit_event(
                    icsr_audit, cls_name=cls_name_l, meta=icsr_out.get("meta"),
                )

            if bool(getattr(args, "icsr_soft_mix", False)):
                sm_np, sm_ap, sm_source, sm_meta = resolve_prompts(
                    args, evo_optimizer, cls_name_l, route_config=route_config,
                )
                _sm_src_cat = sm_meta.get("normal_source_category", cls_name_l)
                sm_np, sm_ap, sm_source = apply_safe_switch_decision(
                    evo_optimizer=evo_optimizer, cls_name_l=cls_name_l,
                    normal_prompt=sm_np, abnormal_prompt=sm_ap,
                    source=sm_source, route_config=route_config,
                    source_category=_sm_src_cat,
                )
                soft_mix_semantic = (sm_np, sm_ap, sm_source, sm_meta)

            if icsr_out["pairs"] is not None:
                icsr_pairs = icsr_out["pairs"]
                top1_src_cls = icsr_out["meta"]["topk_classes"][0]
                source = icsr_out["source"]
                resolve_meta = {
                    "target_category": cls_name_l,
                    "normal_source_category": top1_src_cls,
                    "abnormal_source_category": top1_src_cls,
                    "normal_resolution": "icsr",
                    "abnormal_resolution": "icsr",
                    "mode": "icsr",
                    "icsr": icsr_out["meta"],
                }
                normal_prompt, abnormal_prompt = icsr_pairs[0][0], icsr_pairs[0][1]
            else:
                # gate_out → use standard resolve_prompts chain; prepared is kept for reuse in B
                normal_prompt, abnormal_prompt, source, resolve_meta = resolve_prompts(
                    args, evo_optimizer, cls_name_l, route_config=route_config
                )
                _src_cat = resolve_meta.get("normal_source_category", cls_name_l)
                normal_prompt, abnormal_prompt, source = apply_safe_switch_decision(
                    evo_optimizer=evo_optimizer,
                    cls_name_l=cls_name_l,
                    normal_prompt=normal_prompt,
                    abnormal_prompt=abnormal_prompt,
                    source=source,
                    route_config=route_config,
                    source_category=_src_cat,
                )
                resolve_meta = {**resolve_meta, "icsr": icsr_out["meta"]}
            _log_route_audit_once(cls_name_l, resolve_meta)
            if cls_name_l not in warned_missing_cls:
                logger.info("Class '%s' → %s", cls_name_l, source)
                warned_missing_cls.add(cls_name_l)
        else:
            # resolve prompts (mainline deterministic routing)
            normal_prompt, abnormal_prompt, source, resolve_meta = resolve_prompts(
                args, evo_optimizer, cls_name_l, route_config=route_config
            )
            # For cross-domain runs, source_category is the training-domain category name (found by semantic transfer)
            _src_cat = resolve_meta.get("normal_source_category", cls_name_l)
            normal_prompt, abnormal_prompt, source = apply_safe_switch_decision(
                evo_optimizer=evo_optimizer,
                cls_name_l=cls_name_l,
                normal_prompt=normal_prompt,
                abnormal_prompt=abnormal_prompt,
                source=source,
                route_config=route_config,
                source_category=_src_cat,
            )
            _log_route_audit_once(cls_name_l, resolve_meta)
            if cls_name_l not in warned_missing_cls and "exact" not in source:
                logger.info("Class '%s' → %s", cls_name_l, source)
                warned_missing_cls.add(cls_name_l)

        with torch.no_grad():
            if use_official_baseline:
                infer_result = active_scorer.infer_official_baseline(
                    image, cls_name_l, stage=stage,
                )
            elif icsr_active and bool(getattr(args, "icsr_soft_mix", False)):
                # --- soft-mix branch ---
                # soft_mix_opts is pre-built in the ICSR init block so invalid
                # range_source values fail before inference starts.

                icsr_score_val, icsr_map_val = None, None
                if icsr_pairs is not None:
                    icsr_infer = _infer_icsr_aggregation(
                        active_scorer, prepared, image, icsr_pairs, stage, logger,
                        need_pixel=need_pixel, args=args,
                    )
                    icsr_score_val, icsr_map_val = _extract_image_score_and_pixel_map(
                        infer_result=icsr_infer, need_pixel=need_pixel, args=args,
                    )

                if soft_mix_semantic is None:
                    sem_np, sem_ap = normal_prompt, abnormal_prompt
                else:
                    sem_np, sem_ap, _sem_src, _sem_meta = soft_mix_semantic
                sem_infer = _infer_single_pair(
                    active_scorer, prepared, image, sem_np, sem_ap, stage, logger,
                )
                sem_score_val, sem_map_val = _extract_image_score_and_pixel_map(
                    infer_result=sem_infer, need_pixel=need_pixel, args=args,
                )

                alpha_val = compute_soft_mix_alpha(
                    icsr_out.get("meta") if isinstance(icsr_out, dict) else None,
                    soft_mix_opts,
                )
                blended_score, blended_map = _blend_score_and_map(
                    alpha_val, icsr_score_val, icsr_map_val,
                    sem_score_val, sem_map_val,
                )

                if icsr_audit is not None:
                    record_icsr_alpha(icsr_audit, cls_name=cls_name_l, alpha=alpha_val)
                resolve_meta = {
                    **(resolve_meta if isinstance(resolve_meta, dict) else {}),
                    "soft_mix_alpha": float(alpha_val),
                    "soft_mix_semantic_source": (
                        soft_mix_semantic[2] if soft_mix_semantic is not None else None
                    ),
                }
                infer_result = {
                    "image_scores": np.asarray([blended_score], dtype=np.float32),
                    "pixel_maps": (
                        blended_map[np.newaxis, ...] if blended_map is not None else None
                    ),
                }
                # --- end soft-mix branch ---
            elif icsr_pairs is not None:
                infer_result = _infer_icsr_aggregation(
                    active_scorer, prepared, image, icsr_pairs, stage, logger,
                    need_pixel=need_pixel, args=args,
                )
            elif espr_embeddings:
                _espr_emb = _resolve_espr_embedding(cls_name_l, resolve_meta)
                if _espr_emb is not None:
                    _espr_prep = {"text_embeddings": _espr_emb}
                    infer_result = scorer.infer_prepared(image, _espr_prep, stage=stage)
                else:
                    infer_result = _infer_single_pair(
                        active_scorer, prepared, image, normal_prompt, abnormal_prompt,
                        stage, logger,
                    )
            else:
                infer_result = _infer_single_pair(
                    active_scorer, prepared, image, normal_prompt, abnormal_prompt,
                    stage, logger,
                )

        cls_score, cls_map = _extract_image_score_and_pixel_map(
            infer_result=infer_result, need_pixel=need_pixel, args=args
        )
        final_score = cls_score
        final_map = cls_map

        if (
            getattr(args, "enable_shared_bank", False)
            and len(shared_bank) > 0
            and not use_official_baseline
        ):
            alpha = resolve_alpha_from_source(
                source=source,
                alpha_exact=float(getattr(args, "shared_alpha_exact", 0.0)),
                alpha_semantic=float(getattr(args, "shared_alpha_semantic", 0.35)),
                alpha_missing=float(getattr(args, "shared_alpha_missing", 1.0)),
                alpha_icsr=float(getattr(args, "shared_alpha_icsr", 0.35)),
            )
            if alpha > 0.0:
                shared_candidates, route_weights = _select_shared_candidates(cls_name_l)
                shared_items = []
                for cand in shared_candidates:
                    prep = _ensure_prepared_shared_prompt(
                        cand["normal_prompt"], cand["abnormal_prompt"]
                    )
                    with torch.no_grad():
                        infer_shared = None
                        if prep is not None:
                            infer_prepared = getattr(scorer, "infer_prepared", None)
                            if callable(infer_prepared):
                                try:
                                    infer_shared = infer_prepared(image, prep, stage=stage)
                                except Exception as exc:
                                    if not warned_infer_prepared:
                                        logger.warning("infer_prepared failed once; fallback to infer(): %s", exc)
                                        warned_infer_prepared = True
                                    infer_shared = None
                        if infer_shared is None:
                            try:
                                infer_shared = scorer.infer(
                                    image,
                                    cand["normal_prompt"],
                                    cand["abnormal_prompt"],
                                    stage=stage,
                                )
                            except TypeError:
                                infer_shared = scorer.infer(
                                    image,
                                    cand["normal_prompt"],
                                    cand["abnormal_prompt"],
                                )
                    score_s, map_s = _extract_image_score_and_pixel_map(
                        infer_result=infer_shared, need_pixel=need_pixel, args=args
                    )
                    shared_items.append((score_s, map_s))

                final_score, final_map = fuse_cls_and_shared(
                    image_score_cls=cls_score,
                    pixel_map_cls=cls_map,
                    shared_items=shared_items,
                    alpha=alpha,
                    weights=route_weights,
                    confidence_weighted=getattr(args, "confidence_weighted_fusion", False),
                )

        if final_map is None:
            final_map = np.zeros((args.image_size, args.image_size), dtype=np.float32)

        results["pr_sp"].append(float(final_score))
        results["anomaly_maps"].append(final_map)
        if getattr(args, "dump_image_logits", ""):
            results["image_logits"].append(
                _extract_single_image_logits(infer_result, args.prompt_num)
            )
        results["path"].extend(items["img_path"])

    if icsr_audit is not None and int(icsr_audit.get("total", 0)) > 0:
        for audit_line in format_icsr_audit_lines(icsr_audit):
            logger.info(audit_line)

    if icsr_audit is not None and icsr_audit.get("alpha_stats"):
        for audit_line in format_icsr_alpha_lines(icsr_audit):
            logger.info(audit_line)

    # compute metrics
    compute_and_report_metrics(results, obj_list, logger, args)

    _dump_debug_scores(getattr(args, "debug_dump_scores", "") or "", results)

    _dump_path_sf = getattr(args, "dump_score_fusion_inputs", "") or ""
    if _dump_path_sf:
        from utils.score_fusion_dump import dump_score_fusion_inputs
        dump_score_fusion_inputs(_dump_path_sf, results)
        logger.info(f"SOURCE_CAL_GRID_HASH={_compute_current_grid_hash()}")

    _dump_path_ens = getattr(args, "dump_ensemble_inputs", "") or ""
    if _dump_path_ens:
        from utils.ensemble_dump import dump_ensemble_inputs
        dump_ensemble_inputs(_dump_path_ens, results)
        logger.info("ENSEMBLE_INPUT_DUMP path=%s n=%d", _dump_path_ens, len(results["cls_names"]))

    _dump_path_logits = getattr(args, "dump_image_logits", "") or ""
    if _dump_path_logits:
        from utils.image_logit_dump import dump_image_logits
        dump_image_logits(
            _dump_path_logits,
            results,
            prompt_num=args.prompt_num,
            scorer_temperature=getattr(args, "scorer_temperature", 1.0),
        )
        logger.info(
            "IMAGE_LOGIT_DUMP path=%s n=%d prompt_num=%d scorer_temperature=%.6g",
            _dump_path_logits,
            len(results["image_logits"]),
            args.prompt_num,
            float(getattr(args, "scorer_temperature", 1.0)),
        )

    log_cuda_peak_memory(logger, "test_universal", args.device_id)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_base_parser():
    parser = argparse.ArgumentParser(
        "Universal test with optimized prompts", add_help=True
    )

    parser.add_argument("--dataset", type=str, default="mvtec")
    parser.add_argument(
        "--product_list",
        type=str,
        default="",
        help="Optional comma-separated class filter for diagnostic/source-heldout evaluation. "
             "Empty string evaluates all classes.",
    )
    parser.add_argument("--model", type=str, default="ViT-L-14-336")
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--pretrained", type=str, default="openai")
    parser.add_argument("--features_list", type=int, nargs="+",
                        default=[6, 12, 18, 24])

    parser.add_argument("--data_path", type=str, default="./dataset/mvisa/data")
    parser.add_argument("--save_path", type=str,
                        default="./results-pro/test_universal")
    parser.add_argument("--save_visualizations", action="store_true",
                        help="Save input/anomaly-map/overlay visualizations under save_path/imgs.")
    parser.add_argument("--visualization_limit_per_class", type=int, default=-1,
                        help="Limit saved visualizations per class; -1 saves all images.")
    parser.add_argument("--config_path", type=str,
                        default="./open_clip_local/model_configs/ViT-L-14-336.json")
    parser.add_argument("--pretrained_path", type=str,
                        default="./pretrained_weight/ViT-L-14-336px.pt")
    parser.add_argument("--checkpoint_path", type=str, default="")
    parser.add_argument("--evo_rules_path", type=str, default="")

    parser.add_argument("--prompt_context_len", type=int, default=5)
    parser.add_argument("--prompt_num", type=int, default=8)
    parser.add_argument("--prompt_agg", type=str, default="mean", choices=["mean", "max"],
                        help="Prompt aggregation: mean (weighted average) or max (per-prompt max)")
    parser.add_argument("--prompt_state_len", type=int, default=5)
    parser.add_argument("--per_slot_mapping", action="store_true",
                        help="Use per-slot class_mapping (auto-detected from checkpoint if omitted)")

    parser.add_argument("--evo_population", type=int, default=8)
    parser.add_argument("--evo_generations", type=int, default=3)
    parser.add_argument("--evo_topk", type=int, default=4)
    parser.add_argument("--evo_dual_branch", action="store_true")
    parser.add_argument("--zero_shot_scoring", dest="zero_shot_scoring", action="store_true",
                        help="Compatibility mode for legacy PFL-style text encoders/checkpoints: replace latent prompt bias with zeros")
    parser.add_argument("--no_zero_shot_scoring", dest="zero_shot_scoring", action="store_false",
                        help="If a legacy PFL-compatible scorer/model is present, use its learned latent prompt bias")
    parser.set_defaults(zero_shot_scoring=False)
    parser.add_argument("--global_shared_prompts", action="store_true")
    parser.add_argument("--shared_prompt", type=str, default="X object")
    parser.add_argument("--shared_normal_prompt", type=str,
                        default="X normal object")
    parser.add_argument("--shared_abnormal_prompt", type=str,
                        default="X anomalous object")
    parser.add_argument("--enable_shared_bank", action="store_true")
    parser.add_argument("--shared_bank_path", type=str, default="")
    parser.add_argument("--shared_bank_size", type=int, default=4)
    parser.add_argument("--shared_topk", type=int, default=2)
    parser.add_argument("--shared_route_temp", type=float, default=0.07)
    parser.add_argument("--shared_alpha_exact", type=float, default=0.0)
    parser.add_argument("--shared_alpha_semantic", type=float, default=0.35)
    parser.add_argument("--shared_alpha_missing", type=float, default=1.0)

    parser.add_argument("--ablate_prompt_bank", action="store_true",
                        help="Ablation: bypass learnable prompt bank, use fixed CLIP text encoding")
    parser.add_argument("--scorer_type", type=str, default="prompt_bank",
                        choices=["prompt_bank", "clip_transfer", "custom", "external"])
    parser.add_argument("--use_scoring_head", action="store_true",
                        help="Use learned MLP scoring head instead of cosine similarity (requires Stage1.5 checkpoint)")
    parser.add_argument("--external_adapter", type=str, default="anomalyclip",
                        help="External model adapter (only with --scorer_type external)")
    parser.add_argument("--scorer_module", type=str, default="")
    parser.add_argument("--scorer_class", type=str, default="")
    parser.add_argument("--scorer_config", type=str, default="")
    parser.add_argument("--scorer_kwargs_json", type=str, default="")

    # Pixel-level tuning
    parser.add_argument("--pixel_sigma", type=float, default=8.0,
                        help="Gaussian sigma for pixel-level metric smoothing")
    parser.add_argument("--pixel_layer_weights", type=float, nargs=4,
                        default=[1, 1, 1, 1],
                        help="Per-layer weights for pixel anomaly map aggregation")
    parser.add_argument("--enable_layer_taa", action="store_true",
                        help="Enable test-time adaptive aggregation over per-layer pixel anomaly maps")
    parser.add_argument("--layer_taa_tau", type=float, default=1.0,
                        help="Temperature for Layer-TAA softmax weighting over per-layer anomaly strengths")
    parser.add_argument("--upsample_mode", type=str, default="bicubic",
                        choices=["bilinear", "bicubic"],
                        help="Interpolation mode for anomaly map upsampling")
    parser.add_argument("--confidence_weighted_fusion", action="store_true",
                        help="Use pixel-level confidence weighting in shared bank fusion")

    parser.add_argument("--skip_pro", action="store_true",
                        help="Skip AUPRO computation (slow) for faster evaluation")
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=111)

    # [DEPRECATED] Prototype bank — abandoned. Hidden no-ops for backward compat.
    parser.add_argument("--enable_prototype_bank", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--proto_bank_path", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--proto_alpha_image", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--proto_alpha_pixel", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--proto_temperature", type=float, default=20.0, help=argparse.SUPPRESS)
    parser.add_argument("--proto_topk_percent", type=float, default=0.1, help=argparse.SUPPRESS)
    parser.add_argument("--shared_prototype_bank", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--shared_proto_bank_path", type=str, default="", help=argparse.SUPPRESS)

    # ICSR (Image-Conditioned Source-class Routing)
    parser.add_argument("--enable_icsr", action="store_true",
                        help="Enable image-conditioned source-class routing.")
    parser.add_argument("--icsr_topk", type=int, default=3)
    parser.add_argument("--icsr_tau", type=float, default=0.1)
    parser.add_argument("--icsr_gate_entropy_threshold", type=float, default=0.85)
    parser.add_argument("--icsr_min_sim", type=float, default=0.15)
    parser.add_argument("--icsr_min_margin", type=float, default=0.02)
    parser.add_argument("--shared_alpha_icsr", type=float, default=0.35,
                        help="Fusion alpha used when shared_bank sees an 'icsr' source.")
    parser.add_argument("--icsr_bank_template_ensemble", action="store_true",
                        help="Encode each source class with a 7-template CLIP set; mean-pool then L2-renormalize.")
    parser.add_argument("--icsr_bank_center", action="store_true",
                        help="Subtract bank mean from each entry then L2-renormalize, removing the global text direction.")
    parser.add_argument("--icsr_bank_source_evo", action="store_true",
                        help="Use evo-cache prompts (normal+abnormal per class) as bank content "
                             "instead of template. Wins over --icsr_bank_template_ensemble when both are set. "
                             "Classes missing either evo role fall back to the template path with a warning.")
    parser.add_argument("--icsr_soft_mix", action="store_true",
                        help="Enable the soft-mixture gate: blend ICSR and semantic paths by a "
                             "monotonic alpha derived from meta. When on, the three hard-gate "
                             "flags (--icsr_gate_entropy_threshold, --icsr_min_sim, --icsr_min_margin) "
                             "are IGNORED with a one-time warning.")
    parser.add_argument("--icsr_alpha_formula", type=str, default="F_A",
                        choices=["F_A", "F_C"],
                        help="Alpha formula. F_A=arithmetic mean (default). F_C=margin-weighted "
                             "(not implemented).")
    parser.add_argument("--icsr_alpha_range_source", type=str, default="fixed",
                        choices=["fixed", "adaptive"],
                        help="Source of normalization ranges. fixed=precomputed constants (default). "
                             "adaptive=per-run quantile calibration (not implemented).")
    parser.add_argument("--espr_embeddings_path", type=str, default="",
                        help="Path to a precomputed espr_embeddings.pt. "
                             "When set, bypasses text encoding for categories with refined embeddings.")
    parser.add_argument("--debug_dump_scores", type=str, default="",
                        help="If non-empty, write per-image scores/gt/class/path JSON to this path after metric computation. Useful for regression checks.")
    parser.add_argument(
        "--score_fusion_config",
        type=str,
        default="",
        help="Path to a frozen score-fusion JSON config. "
             "Empty string → default scoring head (bit-equal to legacy). "
             "Non-empty path that fails to load is a HARD ERROR (no silent fallback).",
    )

    parser.add_argument(
        "--dump_score_fusion_inputs",
        type=str,
        default="",
        help="Path to write a float32 npz of top-65,536 raw anomaly + image_score_input "
             "per image, for offline score-fusion calibration. "
             "Empty string → no dump.",
    )

    parser.add_argument(
        "--dump_ensemble_inputs",
        type=str,
        default="",
        help="Path to write full anomaly maps, masks, image scores, labels, classes, and paths "
             "for offline routing-ensemble replay. Empty string -> no dump.",
    )

    parser.add_argument(
        "--dump_image_logits",
        type=str,
        default="",
        help="Path to write raw prompt-bank image logits for offline scoring-geometry replay. "
             "Empty string → no dump.",
    )

    parser.add_argument(
        "--fusion_alpha",
        type=float,
        default=0.5,
        help="Image-vs-pixel blend weight for pr_sp = alpha*image + (1-alpha)*topk_pixel. "
             "Default 0.5 matches legacy. Use 1.0 for pure image score.",
    )

    parser.add_argument(
        "--scorer_temperature",
        type=float,
        default=1.0,
        help="Temperature divisor for the normal/abnormal softmax in scoring. "
             "T<1 sharpens discrimination, T>1 smooths. Default 1.0 matches legacy.",
    )

    return parser


def build_parser():
    return build_base_parser()


def _detect_removed_flags(argv: List[str]) -> List[str]:
    removed = []
    for token in argv:
        if not token.startswith("--"):
            continue
        name = token.split("=", 1)[0]
        if name in REMOVED_INTERFACE_FLAGS and name not in removed:
            removed.append(name)
    return removed


def parse_args_with_breaking_guard(parser: argparse.ArgumentParser, argv: Optional[List[str]] = None):
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    removed = _detect_removed_flags(raw_argv)
    if removed:
        detail = "\n".join(f"  - {flag}: {REMOVED_INTERFACE_FLAGS[flag]}" for flag in removed)
        parser.error(
            "BREAKING: historical experiment flags were removed from mainline `test_universal.py`.\n"
            f"{detail}\n"
            "Use strict/mainline commands without these flags."
        )
    return parser.parse_args(raw_argv)


def main():
    parser = build_parser()
    args = parse_args_with_breaking_guard(parser)
    _enforce_transfer_mainline(args, route_config=STRICT_ROUTE_CONFIG)
    setup_seed(args.seed)
    run_universal_test(args, route_config=STRICT_ROUTE_CONFIG)


if __name__ == "__main__":
    main()
