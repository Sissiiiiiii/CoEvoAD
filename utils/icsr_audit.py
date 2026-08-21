from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List


_REASON_ORDER = (
    "ok",
    "below_min_sim",
    "below_min_margin",
    "entropy_uniform",
    "no_role_pairs",
    "bank_empty",
    "unknown",
)


def create_icsr_audit() -> Dict[str, Any]:
    return {
        "total": 0,
        "hit": 0,
        "gate_reason_counts": Counter(),
        "first_examples": {},
        "alpha_stats": [],
        "alpha_per_class": {},
        "_alpha_sums": {},
    }


def record_icsr_alpha(audit: Dict[str, Any], cls_name: str, alpha: float) -> None:
    """Record a per-query alpha scalar and update per-class running mean."""
    alpha = float(alpha)
    audit.setdefault("alpha_stats", []).append(alpha)
    sums = audit.setdefault("_alpha_sums", {})
    s, n = sums.get(cls_name, (0.0, 0))
    sums[cls_name] = (s + alpha, n + 1)
    audit.setdefault("alpha_per_class", {})[cls_name] = (s + alpha) / (n + 1)


def format_icsr_alpha_lines(audit: Dict[str, Any]) -> List[str]:
    stats = audit.get("alpha_stats") or []
    if not stats:
        return []
    sorted_a = sorted(stats)
    n = len(sorted_a)
    mean = sum(sorted_a) / n
    p50 = sorted_a[n // 2] if n % 2 == 1 else 0.5 * (sorted_a[n // 2 - 1] + sorted_a[n // 2])
    p90 = sorted_a[max(0, int(0.9 * n) - 1)]
    zero_frac = sum(1 for a in sorted_a if a <= 1e-9) / n
    one_frac = sum(1 for a in sorted_a if a >= 1.0 - 1e-9) / n
    lines = [
        f"SOFT_MIX alpha stats: mean={mean:.3f} p50={p50:.3f} p90={p90:.3f} "
        f"zero_frac={zero_frac:.2f} one_frac={one_frac:.2f} n={n}",
    ]
    per_cls = audit.get("alpha_per_class") or {}
    if per_cls:
        joined = ", ".join(f"{cls}={v:.3f}" for cls, v in sorted(per_cls.items()))
        lines.append(f"SOFT_MIX per-class alpha: {{{joined}}}")
    return lines


def record_icsr_audit_event(
    audit: Dict[str, Any],
    cls_name: str,
    meta: Dict[str, Any] | None,
) -> None:
    meta = meta or {}
    reason = str(meta.get("gate_reason", "unknown") or "unknown")
    gate_passed = bool(meta.get("gate_passed", False))

    audit["total"] += 1
    audit["gate_reason_counts"][reason] += 1
    if gate_passed:
        audit["hit"] += 1

    if reason in audit["first_examples"]:
        return

    audit["first_examples"][reason] = {
        "class": cls_name,
        "top1_sim": _to_float_or_none(meta.get("top1_sim")),
        "top1_top2_margin": _to_float_or_none(meta.get("top1_top2_margin")),
        "H_norm": _to_float_or_none(meta.get("H_norm")),
        "topk_classes": [str(x) for x in (meta.get("topk_classes") or [])[:3]],
    }


def format_icsr_audit_lines(audit: Dict[str, Any]) -> List[str]:
    total = int(audit.get("total", 0))
    hit = int(audit.get("hit", 0))
    hit_rate = (100.0 * float(hit) / float(total)) if total > 0 else 0.0

    counts = audit.get("gate_reason_counts", Counter())
    ordered_reasons = [r for r in _REASON_ORDER if counts.get(r, 0) > 0]
    ordered_reasons.extend(
        sorted(r for r in counts.keys() if r not in _REASON_ORDER and counts.get(r, 0) > 0)
    )

    lines = [
        f"ICSR summary: total={total} hit={hit} hit_rate={hit_rate:.2f}%",
        "ICSR gate reasons: " + ", ".join(
            f"{reason}={int(counts[reason])}" for reason in ordered_reasons
        ),
    ]

    first_examples = audit.get("first_examples", {})
    for reason in ordered_reasons:
        example = first_examples.get(reason)
        if not example:
            continue
        topk_classes = ",".join(example.get("topk_classes", [])) or "-"
        lines.append(
            "ICSR example[{reason}]: class={cls} top1={top1} margin={margin} "
            "H_norm={entropy} topk={topk}".format(
                reason=reason,
                cls=example.get("class", "unknown"),
                top1=_fmt_or_na(example.get("top1_sim")),
                margin=_fmt_or_na(example.get("top1_top2_margin")),
                entropy=_fmt_or_na(example.get("H_norm")),
                topk=topk_classes,
            )
        )
    return lines


def _fmt_or_na(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _to_float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
