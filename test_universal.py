"""
Universal test entrypoint (paper mainline).

Enables the transfer routing used in the paper (semantic fallback + template
transfer + optional safe-switch) on top of the strict evaluation core in
test_strict.py.
"""

from __future__ import annotations

import argparse
import os

# Ensure importing test_strict does not trigger the strict removed-flag preflight.
os.environ.setdefault("COEVOAD_TEST_ROUTE_PROFILE", "SUPPLEMENTARY_FALLBACK")

from test_strict import (  # noqa: E402
    build_base_parser,
    build_supplementary_route_config,
    run_universal_test,
    setup_seed,
)


def build_supplementary_parser() -> argparse.ArgumentParser:
    parser = build_base_parser()

    parser.add_argument("--disable_role_fallback", action="store_true")
    parser.add_argument("--enable_donor_route", dest="disable_donor_route", action="store_false")
    parser.add_argument("--disable_donor_route", dest="disable_donor_route", action="store_true")
    parser.set_defaults(disable_donor_route=True)
    parser.add_argument("--enable_semantic_fallback", dest="enable_semantic_fallback", action="store_true")
    parser.add_argument("--disable_semantic_fallback", dest="enable_semantic_fallback", action="store_false")
    parser.set_defaults(enable_semantic_fallback=True)
    parser.add_argument("--semantic_fallback_template", type=str, default="a photo of {}")
    parser.add_argument("--semantic_fallback_min_sim", type=float, default=0.4)
    parser.add_argument("--semantic_fallback_min_margin", type=float, default=0.0)

    parser.add_argument("--enable_template_transfer", dest="enable_template_transfer", action="store_true")
    parser.add_argument("--disable_template_transfer", dest="enable_template_transfer", action="store_false")
    parser.set_defaults(enable_template_transfer=True)

    parser.add_argument("--enable_safe_switch", action="store_true")
    parser.add_argument("--safe_switch_min_gain_cross", type=float, default=0.0)
    parser.add_argument("--safe_switch_min_gain_src", type=float, default=-0.02)
    parser.add_argument("--safe_switch_max_score_std", type=float, default=0.05)
    parser.add_argument("--safe_switch_enable_mix", action="store_true")
    parser.add_argument("--safe_switch_mix_gain_cross", type=float, default=0.1)
    parser.add_argument("--safe_switch_mix_score_std", type=float, default=0.02)
    parser.add_argument("--safe_switch_mix_alpha_min", type=float, default=0.25)
    parser.add_argument("--safe_switch_mix_alpha_max", type=float, default=0.75)

    return parser


def validate_supplementary_args(args, parser: argparse.ArgumentParser) -> None:
    all_disabled = (
        not bool(getattr(args, "enable_semantic_fallback", False))
        and not bool(getattr(args, "enable_template_transfer", False))
        and not bool(getattr(args, "enable_safe_switch", False))
        and bool(getattr(args, "disable_role_fallback", False))
    )
    if all_disabled:
        parser.error(
            "All supplementary routing controls are disabled. "
            "For strict runs, use test_strict.py instead."
        )


def main():
    parser = build_supplementary_parser()
    args = parser.parse_args()
    validate_supplementary_args(args, parser)
    setup_seed(args.seed)
    route_config = build_supplementary_route_config(args)
    run_universal_test(args, route_config=route_config)


if __name__ == "__main__":
    main()
