"""
EvoPrompt two-stage optimizer.

Stage 1: train the deterministic prompt-bank model (or reuse an already-trained one)
    - train the prompt embeddings (prompt_context, prompt_state)
    - train the remaining trainable parameters

Stage 2: optimize the textual prompts (model parameters frozen)
    - freeze all model parameters
    - take a subset of the training set
    - search for the best prompt on that subset with a genetic algorithm
    - no training; only evaluation of which prompt performs best
    - save the optimized prompt rules
"""

import logging
import math
import re
import random
import json
import os
import hashlib
from collections import Counter
from typing import List, Dict, Tuple, Optional, Callable, Any, Set

logger = logging.getLogger(__name__)
stage2_logger = logging.getLogger("optimize_universal")
try:
    import torch
except ImportError:
    torch = None

# Normal-semantics vocabulary: filters adjectives/templates on the abnormal role so that
# defect prompts do not end up carrying normal semantics
_NORMAL_SEMANTIC_WORDS = frozenset({
    "typical", "standard", "normal", "common", "clean", "pristine",
    "flawless", "perfect", "regular", "pure", "natural",
})

RULE_MUTATION_OPS = (
    "descriptor_replace",
    "descriptor_insert",
    "descriptor_drop",
    "template_swap",
    "descriptor_mix",
    "synonym",
)

LLM_MUTATION_OPS = ("llm_rephrase", "llm_creative")


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def build_route_audit_payload(
    *,
    target_class: str,
    role: str,
    selected_source_class: Optional[str],
    donor_score: float,
    semantic_sim: float,
    attribute_match: float,
    behavior_consistency: float,
    margin: float,
    gate_decision: str,
    source_tag: str,
) -> Dict[str, Any]:
    return {
        "target_class": str(target_class),
        "role": str(role),
        "selected_source_class": selected_source_class,
        "donor_score": float(donor_score),
        "semantic_sim": float(semantic_sim),
        "attribute_match": float(attribute_match),
        "behavior_consistency": float(behavior_consistency),
        "margin": float(margin),
        "gate_decision": str(gate_decision),
        "source_tag": str(source_tag),
    }


class EvoPromptOptimizer:
    """Text-level EvoPrompt optimizer (no embedding or cross-attention fusion; returns substituted text only).

    :param population_size: initial population size
    :param generations: number of generations
    :param topk: number of candidates kept after selection
    :param templates: template pool (the 'X ' prefix must be kept, since the existing TextEncoder inserts visual tokens at 'X')
    :param adjectives: adjective pool (may be empty; used only as a mutation source)
    :param lambda_diversity: MMR diversity weight (0.0-1.0; 0.0 selects by score alone)
    """

    def __init__(
        self,
        population_size: int = 8,
        generations: int = 3,
        topk: int = 4,
        templates: List[str] = None,
        adjectives: List[str] = None,
        lambda_diversity: float = 0.2,
        llm_mutation_enabled: bool = False,
        llm_model_id: str = "",
        llm_mutation_max_tokens: int = 32,
        evo_crossover_rate: float = 0.0,
        evo_random_search: bool = False,
        evo_random_search_seed: Optional[int] = None,
        evo_mutation_ops: Optional[Any] = None,
    ) -> None:
        # Marks text-substitution-only mode, checked by the forward_ensemble branch
        self.text_replace_only: bool = True

        # Enlarged template pool (following EAOT, for more diversity)
        self.templates = templates or [
            "X {name}",
            "X the {name}",
            "X a photo of {name}",
            "X a photo of a {name}",
            "X an image of {name}",
            "X a sample of {name}",
            "X object: {name}",
            "X item: {name}",
            "X clean {name}",
            "X typical {name}",
            "X plain {name}",
            "X standard {name}",
            "X a {name} product",
            "X a {name} sample",
            "X a view of {name}",
        ]
        
        # normal-specific templates (richer phrasing)
        self.normal_templates = [
            "X normal {name}",
            "X typical {name}",
            "X clean {name}",
            "X defect-free {name}",
            "X perfect {name}",
            "X flawless {name}",
            "X pristine {name}",
            "X good quality {name}",
            "X standard {name}",
            "X regular {name}",
        ]
        
        # abnormal-specific templates (richer phrasing)
        self.abnormal_templates = [
            "X abnormal {name}",
            "X defective {name}",
            "X damaged {name}",
            "X flawed {name}",
            "X anomalous {name}",
            "X faulty {name}",
            "X broken {name}",
            "X irregular {name}",
            "X bad quality {name}",
            "X defect in {name}",
        ]
        
        # Enlarged adjective pool (richer descriptors)
        self.adjectives = adjectives or [
            "plain", "typical", "simple", "clear", "standard", 
            "common", "general", "regular", "basic", "ordinary",
            "pure", "natural", "original", "authentic", "genuine"
        ]
        
        # normal-specific adjectives (more varied)
        self.normal_adjectives = [
            "normal", "clean", "typical", "standard", "defect-free", 
            "perfect", "flawless", "pristine", "intact", "unblemished",
            "good", "fine", "healthy", "proper", "regular"
        ]
        
        # abnormal-specific adjectives (more varied)
        self.abnormal_adjectives = [
            "abnormal", "defective", "damaged", "flawed", "anomalous",
            "faulty", "broken", "irregular", "imperfect", "blemished",
            "bad", "poor", "corrupted", "deteriorated", "compromised"
        ]

        # -- Category-agnostic templates (Category-Agnostic Anomaly Prompt, CAAP) --
        # No {name} placeholder, so cross-domain transfer does not depend on the target category name
        self.agnostic_normal_templates = [
            "X a photo of a flawless object",
            "X a clean surface without defects",
            "X normal undamaged product",
            "X intact object with no anomaly",
            "X good quality item",
            "X a photo of a standard sample",
            "X pristine surface texture",
            "X well-manufactured component",
        ]
        self.agnostic_abnormal_templates = [
            "X a photo showing surface defect",
            "X visible anomaly on object",
            "X damaged region with structural flaw",
            "X abnormal pattern indicating defect",
            "X poor quality item with blemish",
            "X a photo of a scratched or cracked surface",
            "X manufacturing defect on component",
            "X irregular texture with contamination",
        ]

        # -- Fine-grained anomaly-type templates --
        # Part C: Structural, Surface, Textural
        self.anomaly_type_templates = {
            "structural": [
                "X structural defect like a hole or cut",
                "X broken shape with missing parts",
                "X deformed structure",
                "X physical damage on the object",
            ],
            "surface": [
                "X scratch or blemish on the surface",
                "X contaminated spot or stain",
                "X discoloration or paint chip",
                "X visible surface damage",
            ],
            "textural": [
                "X rough or irregular texture",
                "X inconsistent pattern or weaving",
                "X worn out fabric or material",
                "X abnormal textural variation",
            ]
        }
        # Extend the fine-grained templates into the category-agnostic pool on the abnormal side
        for templates in self.anomaly_type_templates.values():
            self.agnostic_abnormal_templates.extend(templates)

        self.population_size = population_size
        self.generations = generations
        self.topk = topk
        self.lambda_diversity = lambda_diversity
        
        # Cache: keyed by the (role, name) tuple, valued by the optimized prompt
        # role may be "normal", "abnormal", "shared", or None (defaults to "shared")
        self.cache: Dict[Tuple[str, str], str] = {}

        # Rule metadata: used by the zero-shot safe gate / source-only diagnostics
        self.rule_metadata: Dict[Tuple[str, str], Dict[str, Any]] = {}

        # Equal-budget prompt-search controls. Defaults preserve the original
        # mutation-only EvoPrompt path.
        self.evo_crossover_rate = float(max(0.0, min(1.0, evo_crossover_rate)))
        self.evo_random_search = bool(evo_random_search)
        self.evo_random_search_seed = evo_random_search_seed
        self._custom_mutation_ops = evo_mutation_ops is not None and str(evo_mutation_ops).strip() != ""
        self.evo_mutation_ops = self._parse_mutation_ops(evo_mutation_ops)
        
        # Text-feature cache (used by MMR diversity selection)
        # keyed by the (role, candidate) tuple, valued by the normalized text-feature vector
        self.text_feat_cache: Dict[Tuple[str, str], any] = {}

        # Inference-time cache-resolution switch (off by default; does not affect training/optimization)
        self.inference_resolve_missing_from_cache: bool = False
        self.inference_allow_role_fallback: bool = False

        # LLM mutation state (lazily loaded)
        self.llm_model_id = llm_model_id
        self.llm_mutation_max_tokens = llm_mutation_max_tokens
        self._llm_model = None
        self._llm_tokenizer = None
        self._llm_device = None
        self._llm_cache: Dict[str, str] = {}  # prompt_key -> generated text
        self._llm_stats = {"llm_rephrase": 0, "llm_creative": 0, "llm_fallback": 0, "rule_ops": 0}
        # Auto-disable if no model path provided
        if llm_mutation_enabled and not llm_model_id:
            logger.warning("--llm_mutation_enabled set but --llm_mutation_model_id is empty; disabling LLM mutation.")
            self.llm_mutation_enabled = False
        else:
            self.llm_mutation_enabled = llm_mutation_enabled

        # Template-transfer configuration
        self.enable_template_transfer: bool = False
        self._transfer_templates_cache: Optional[Dict[str, Dict[str, Any]]] = None

        # Semantic-fallback configuration (CLIP text similarity)
        self.enable_semantic_fallback: bool = False
        self.semantic_template: str = "a photo of {}"
        self.semantic_embed_fn: Optional[Callable[[List[str]], Any]] = None
        self.semantic_min_similarity: float = 0.5
        self.semantic_min_margin: float = 0.0
        self.semantic_name_feat_cache: Dict[str, Any] = {}
        self.donor_route_enabled: bool = False
        self.donor_weight_sem: float = 0.60
        self.donor_weight_attr: float = 0.15
        self.donor_weight_beh: float = 0.25
        self.donor_min_similarity: float = 0.40
        self.donor_min_score: float = 0.45
        self.donor_min_margin: float = 0.0

        # ICSR state — populated by set_icsr_bank; empty by default.
        self._icsr_classes: List[str] = []
        self._icsr_matrix: Optional[Any] = None

    @staticmethod
    def _parse_mutation_ops(raw_ops: Optional[Any]) -> Tuple[str, ...]:
        if raw_ops is None or (isinstance(raw_ops, str) and not raw_ops.strip()):
            return RULE_MUTATION_OPS
        if isinstance(raw_ops, str):
            ops = [op.strip() for op in raw_ops.split(",") if op.strip()]
        else:
            ops = [str(op).strip() for op in raw_ops if str(op).strip()]
        if not ops:
            raise ValueError("evo_mutation_ops must contain at least one operator")
        allowed = set(RULE_MUTATION_OPS) | set(LLM_MUTATION_OPS)
        unknown = sorted(set(ops) - allowed)
        if unknown:
            raise ValueError(f"unknown mutation operators: {', '.join(unknown)}")
        return tuple(dict.fromkeys(ops))

    def configure_inference_resolution(
        self,
        enable: bool = True,
        allow_role_fallback: bool = False,
        enable_semantic_fallback: bool = False,
        enable_template_transfer: bool = False,
    ) -> None:
        """Configure how missing categories are resolved from the cache at inference time."""
        self.inference_resolve_missing_from_cache = bool(enable)
        self.inference_allow_role_fallback = bool(allow_role_fallback)
        self.enable_semantic_fallback = bool(enable_semantic_fallback)
        self.enable_template_transfer = bool(enable_template_transfer)
        if enable_template_transfer:
            self._transfer_templates_cache = None  # force re-extraction

    def set_semantic_embedder(
        self,
        embed_fn: Optional[Callable[[List[str]], Any]],
        template: str = "a photo of {}",
        min_similarity: float = 0.5,
        min_margin: float = 0.0,
    ) -> None:
        """Set the text-encoding function used by semantic fallback (usually CLIP encode_text)."""
        self.semantic_embed_fn = embed_fn
        self.semantic_template = template or "a photo of {}"
        self.semantic_min_similarity = float(min_similarity)
        self.semantic_min_margin = float(max(0.0, min_margin))
        self.semantic_name_feat_cache.clear()

    def _to_1d_unit_tensor(self, x: Any):
        if torch is None:
            return None
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            t = x.detach().float().cpu()
        else:
            try:
                t = torch.as_tensor(x, dtype=torch.float32)
            except Exception:
                return None
        if t.ndim == 0:
            t = t.view(1)
        elif t.ndim > 1:
            t = t.view(t.shape[0], -1)
            if t.shape[0] == 0:
                return None
            t = t[0]
        else:
            t = t.flatten()
        n = torch.norm(t, p=2)
        if not torch.isfinite(n) or n.item() <= 0:
            return None
        return t / n

    def _encode_name_semantic(self, name: str):
        if torch is None or self.semantic_embed_fn is None:
            return None
        key = self._normalize_name(name)
        if not key:
            return None
        if key in self.semantic_name_feat_cache:
            return self.semantic_name_feat_cache[key]
        try:
            text = self.semantic_template.format(key)
            feat = self.semantic_embed_fn([text])
            # accept either [D] or [1, D]
            if isinstance(feat, torch.Tensor) and feat.ndim >= 2 and feat.shape[0] > 0:
                feat = feat[0]
            unit = self._to_1d_unit_tensor(feat)
            if unit is None:
                return None
            self.semantic_name_feat_cache[key] = unit
            return unit
        except Exception:
            return None

    def _semantic_fallback_lookup(
        self,
        name_candidates: List[str],
        fallback_roles: List[str],
    ) -> Optional[Tuple[Tuple[str, str], str]]:
        """Pick the closest rule from the same-role cache by CLIP text similarity."""
        if not self.enable_semantic_fallback or self.semantic_embed_fn is None or torch is None:
            return None

        # query vector: mean of the candidate alias vectors
        q_feats = []
        for n in name_candidates:
            f = self._encode_name_semantic(n)
            if f is not None:
                q_feats.append(f)
        if len(q_feats) == 0:
            return None
        q = torch.stack(q_feats, dim=0).mean(dim=0)
        qn = torch.norm(q, p=2)
        if not torch.isfinite(qn) or qn.item() <= 0:
            return None
        q = q / qn

        candidates: List[Tuple[float, Tuple[str, str], str]] = []

        for r in fallback_roles:
            for key, prompt in self.cache.items():
                if not (isinstance(key, tuple) and len(key) == 2):
                    continue
                if key[0] != r:
                    continue
                cand_name = self._normalize_name(key[1])
                cand_feat = self._encode_name_semantic(cand_name)
                if cand_feat is None:
                    continue
                sim = torch.dot(q, cand_feat).item()
                candidates.append((sim, key, prompt))

        if not candidates:
            return None

        # Similarity first; ties are broken by lexicographic key order for reproducibility.
        candidates.sort(key=lambda x: (-x[0], x[1]))
        best_sim, best_key, best_prompt = candidates[0]
        second_sim = candidates[1][0] if len(candidates) > 1 else None

        # Confidence threshold on semantic hits, so a nearest-but-unreliable category is not forced through.
        query_label = name_candidates[0] if name_candidates else "?"
        if best_sim < self.semantic_min_similarity:
            logger.warning(
                "Semantic fallback REJECTED: '%s' -> '%s' "
                "(sim=%.3f < threshold=%.3f); will use shared prompt",
                query_label, best_key[1], best_sim, self.semantic_min_similarity,
            )
            return None
        if second_sim is not None and (best_sim - second_sim) < self.semantic_min_margin:
            logger.warning(
                "Semantic fallback REJECTED (ambiguous): '%s' -> '%s' "
                "(sim=%.3f, margin=%.3f < %.3f); will use shared prompt",
                query_label, best_key[1], best_sim,
                best_sim - second_sim, self.semantic_min_margin,
            )
            return None

        return best_key, best_prompt

    def _topk_semantic_fallback_lookup(
        self,
        name_candidates: List[str],
        fallback_roles: List[str],
        topk: int = 3,
        tau: float = 1.0,
    ) -> List[Tuple[Tuple[str, str], str, float]]:
        """Return the top-k cached rules matched by CLIP text similarity, with raw similarities.

        Returns: [(key, prompt, raw_similarity), ...]
        `tau` is kept in the signature for backward compatibility only; the actual weights are computed uniformly at pair level.
        """
        del tau
        if not self.enable_semantic_fallback or self.semantic_embed_fn is None or torch is None:
            return []

        q_feats = []
        for n in name_candidates:
            f = self._encode_name_semantic(n)
            if f is not None:
                q_feats.append(f)
        if len(q_feats) == 0:
            return []
        q = torch.stack(q_feats, dim=0).mean(dim=0)
        qn = torch.norm(q, p=2)
        if not torch.isfinite(qn) or qn.item() <= 0:
            return []
        q = q / qn

        candidates: List[Tuple[float, Tuple[str, str], str]] = []
        for r in fallback_roles:
            for key, prompt in self.cache.items():
                if not (isinstance(key, tuple) and len(key) == 2):
                    continue
                if key[0] != r:
                    continue
                cand_feat = self._encode_name_semantic(self._normalize_name(key[1]))
                if cand_feat is None:
                    continue
                sim = torch.dot(q, cand_feat).item()
                if sim >= self.semantic_min_similarity:
                    candidates.append((sim, key, prompt))

        if not candidates:
            return []

        candidates.sort(key=lambda x: (-x[0], x[1]))
        selected = candidates[:topk]

        return [(key, prompt, sim) for sim, key, prompt in selected]

    @staticmethod
    def _route_score_from_tag(source_tag: str, raw_similarity: float) -> float:
        source_tag = str(source_tag or "")
        if source_tag in {"exact", "alias"}:
            return 1.0
        if source_tag in {"semantic_transfer", "semantic_fallback", "baseline_gate"}:
            return float(raw_similarity)
        return 0.0

    def _build_prompt_route_candidate(
        self,
        prompt: str,
        source_tag: str,
        source_category: Optional[str] = None,
        raw_similarity: float = 0.0,
        gate_inserted: bool = False,
        semantic_sim: Optional[float] = None,
        attribute_match: Optional[float] = None,
        behavior_consistency: Optional[float] = None,
        donor_score: Optional[float] = None,
        gate_decision: str = "pass",
        donor_margin: float = 0.0,
    ) -> Dict[str, Any]:
        score = (
            float(donor_score)
            if donor_score is not None
            else self._route_score_from_tag(source_tag, raw_similarity)
        )
        return {
            "prompt": prompt,
            "source_tag": source_tag,
            "source_category": source_category,
            "raw_similarity": float(raw_similarity),
            "score": score,
            "semantic_sim": float(raw_similarity if semantic_sim is None else semantic_sim),
            "attribute_match": None if attribute_match is None else float(attribute_match),
            "behavior_consistency": None if behavior_consistency is None else float(behavior_consistency),
            "donor_score": None if donor_score is None else float(donor_score),
            "gate_decision": str(gate_decision),
            "donor_margin": float(donor_margin),
            "gate_inserted": bool(gate_inserted),
        }

    def _resolve_role_prompt_candidates(
        self,
        name: str,
        role: str,
        default_prompt: str,
        allow_role_fallback: bool,
        topk: int,
        tau: float,
        gate_threshold: float,
    ) -> Dict[str, Any]:
        role_key = role or "shared"
        name_candidates = self._candidate_names(name)
        tgt_name = self._normalize_name(name)

        role_search = [role_key]
        if role_key != "shared":
            role_search.append("shared")
        elif role is None:
            role_search.extend(["normal", "abnormal"])

        for r in role_search:
            for cand_name in name_candidates:
                key = (r, cand_name)
                if key in self.cache:
                    source_tag = "exact" if (r == role_key and cand_name == tgt_name) else "alias"
                    return {
                        "candidates": [
                            self._build_prompt_route_candidate(
                                prompt=self.cache[key],
                                source_tag=source_tag,
                                source_category=key[1],
                                raw_similarity=1.0,
                            )
                        ],
                        "source": source_tag,
                        "used_semantic_topk": False,
                        "gate_inserted": False,
                    }

        fallback_roles = [role_key]
        if role_key != "shared":
            fallback_roles.append("shared")
        else:
            fallback_roles.extend(["normal", "abnormal"])

        semantic_hits = self._topk_semantic_fallback_lookup(
            name_candidates,
            fallback_roles,
            topk=topk,
            tau=tau,
        )
        if semantic_hits:
            candidates = []
            for key, prompt, raw_similarity in semantic_hits:
                src_name = key[1]
                adapted = self._adapt_prompt_to_target(prompt, src_name, tgt_name)
                source_tag = "semantic_transfer" if adapted is not None and adapted != prompt else "semantic_fallback"
                if self.donor_route_enabled:
                    donor_parts = self._donor_score(
                        target_name=tgt_name,
                        source_name=src_name,
                        role=role_key,
                        semantic_sim=raw_similarity,
                        prompt=adapted or prompt,
                    )
                else:
                    donor_parts = {
                        "semantic_sim": float(raw_similarity),
                        "attribute_match": None,
                        "behavior_consistency": None,
                        "donor_score": None,
                    }
                candidates.append(
                    self._build_prompt_route_candidate(
                        prompt=adapted or prompt,
                        source_tag=source_tag,
                        source_category=src_name,
                        raw_similarity=raw_similarity,
                        semantic_sim=donor_parts["semantic_sim"],
                        attribute_match=donor_parts["attribute_match"],
                        behavior_consistency=donor_parts["behavior_consistency"],
                        donor_score=donor_parts["donor_score"],
                    )
                )
            if self.donor_route_enabled:
                candidates.sort(
                    key=lambda item: (-item["score"], -item["raw_similarity"], item["source_category"] or "", item["prompt"])
                )
            else:
                candidates.sort(
                    key=lambda item: (-item["raw_similarity"], item["source_category"] or "", item["prompt"])
                )
            gate_inserted = False
            if self.donor_route_enabled:
                margin = -1.0
                has_comparison = len(candidates) > 1
                if has_comparison:
                    margin = max(0.0, float(candidates[0]["score"]) - float(candidates[1]["score"]))
                for cand in candidates:
                    cand["donor_margin"] = margin
                gate_decision = "pass"
                effective_min_sim = max(float(self.donor_min_similarity), float(self.semantic_min_similarity))
                effective_min_score = max(float(self.donor_min_score), float(gate_threshold))
                if float(candidates[0]["semantic_sim"]) < effective_min_sim:
                    gate_decision = "below_min_semantic_sim"
                elif float(candidates[0]["score"]) < effective_min_score:
                    gate_decision = "below_min_donor_score"
                elif has_comparison and margin < float(self.donor_min_margin):
                    gate_decision = "below_min_margin"
                if gate_decision != "pass":
                    candidates[0]["gate_decision"] = gate_decision
                    candidates = [
                        self._build_prompt_route_candidate(
                            prompt=default_prompt,
                            source_tag="baseline_gate",
                            source_category=None,
                            raw_similarity=float(candidates[0]["semantic_sim"]),
                            gate_inserted=True,
                            semantic_sim=float(candidates[0]["semantic_sim"]),
                            attribute_match=float(candidates[0]["attribute_match"] or 0.5),
                            behavior_consistency=float(candidates[0]["behavior_consistency"] or 0.5),
                            donor_score=float(candidates[0]["score"]),
                            gate_decision=gate_decision,
                            donor_margin=margin,
                        )
                    ]
                    gate_inserted = True
            else:
                if candidates[0]["raw_similarity"] < float(gate_threshold):
                    candidates.append(
                        self._build_prompt_route_candidate(
                            prompt=default_prompt,
                            source_tag="baseline_gate",
                            source_category=None,
                            raw_similarity=float(gate_threshold),
                            gate_inserted=True,
                        )
                    )
                    gate_inserted = True
            return {
                "candidates": candidates,
                "source": candidates[0]["source_tag"],
                "used_semantic_topk": True,
                "gate_inserted": gate_inserted,
            }

        prompt, hit_key, source = self.resolve_cached_prompt(
            name=name,
            role=role_key,
            default_prompt=default_prompt,
            allow_role_fallback=allow_role_fallback,
        )
        return {
            "candidates": [
                self._build_prompt_route_candidate(
                    prompt=prompt,
                    source_tag=source,
                    source_category=hit_key[1] if hit_key is not None else None,
                    raw_similarity=1.0 if source in {"exact", "alias"} else 0.0,
                )
            ],
            "source": source,
            "used_semantic_topk": False,
            "gate_inserted": False,
        }

    def resolve_multi_source_prompts(
        self,
        name: str,
        topk: int = 3,
        tau: float = 1.0,
        gate_threshold: float = 0.4,
        allow_role_fallback: bool = False,
    ) -> Dict[str, Any]:
        """Return a structured multi-source route result.

        Only the semantic-fallback stage expands to top-k candidates; the priority of
        exact/alias/template/shared/default is unchanged from the previous mainline.
        """
        topk = max(1, int(topk))
        tau_safe = max(float(tau), 1e-6)
        normal_prompt = f"X normal {self._normalize_name(name)}"
        abnormal_prompt = f"X abnormal {self._normalize_name(name)}"

        normal_info = self._resolve_role_prompt_candidates(
            name=name,
            role="normal",
            default_prompt=normal_prompt,
            allow_role_fallback=allow_role_fallback,
            topk=topk,
            tau=tau_safe,
            gate_threshold=gate_threshold,
        )
        abnormal_info = self._resolve_role_prompt_candidates(
            name=name,
            role="abnormal",
            default_prompt=abnormal_prompt,
            allow_role_fallback=allow_role_fallback,
            topk=topk,
            tau=tau_safe,
            gate_threshold=gate_threshold,
        )

        pair_candidates: List[Dict[str, Any]] = []
        for cand_n in normal_info["candidates"]:
            for cand_a in abnormal_info["candidates"]:
                pair_candidates.append(
                    {
                        "normal_prompt": cand_n["prompt"],
                        "abnormal_prompt": cand_a["prompt"],
                        "normal_source": cand_n["source_category"] or ("baseline" if cand_n["source_tag"] == "baseline_gate" else "-"),
                        "abnormal_source": cand_a["source_category"] or ("baseline" if cand_a["source_tag"] == "baseline_gate" else "-"),
                        "normal_tag": cand_n["source_tag"],
                        "abnormal_tag": cand_a["source_tag"],
                        "normal_raw_similarity": float(cand_n["raw_similarity"]),
                        "abnormal_raw_similarity": float(cand_a["raw_similarity"]),
                        "normal_semantic_sim": float(cand_n["semantic_sim"]),
                        "abnormal_semantic_sim": float(cand_a["semantic_sim"]),
                        "normal_attribute_match": cand_n["attribute_match"],
                        "abnormal_attribute_match": cand_a["attribute_match"],
                        "normal_behavior_consistency": cand_n["behavior_consistency"],
                        "abnormal_behavior_consistency": cand_a["behavior_consistency"],
                        "normal_score": float(cand_n["score"]),
                        "abnormal_score": float(cand_a["score"]),
                        "normal_gate_decision": cand_n["gate_decision"],
                        "abnormal_gate_decision": cand_a["gate_decision"],
                        "normal_donor_margin": float(cand_n["donor_margin"]),
                        "abnormal_donor_margin": float(cand_a["donor_margin"]),
                        "joint_score": float((cand_n["score"] + cand_a["score"]) / 2.0),
                    }
                )

        dedup_pairs: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for pair in pair_candidates:
            key = (pair["normal_prompt"], pair["abnormal_prompt"])
            prev = dedup_pairs.get(key)
            if prev is None or pair["joint_score"] > prev["joint_score"]:
                dedup_pairs[key] = pair
        ranked_pairs = sorted(
            dedup_pairs.values(),
            key=lambda item: (
                -item["joint_score"],
                item["normal_tag"],
                item["abnormal_tag"],
                item["normal_prompt"],
                item["abnormal_prompt"],
            ),
        )
        selected_pairs = ranked_pairs[: max(1, min(topk, len(ranked_pairs)))]

        if len(selected_pairs) == 1:
            selected_pairs[0]["weight"] = 1.0
        else:
            import math

            joint_scores = [pair["joint_score"] for pair in selected_pairs]
            max_score = max(joint_scores)
            exp_scores = [math.exp((score - max_score) / tau_safe) for score in joint_scores]
            denom = sum(exp_scores)
            for pair, exp_score in zip(selected_pairs, exp_scores):
                pair["weight"] = float(exp_score / denom) if denom > 0 else 1.0 / len(selected_pairs)

        top_pair = selected_pairs[0]
        if len(selected_pairs) > 1:
            source = f"dual:{top_pair['normal_tag']}/{top_pair['abnormal_tag']}+semantic_topk"
        else:
            source = f"dual:{top_pair['normal_tag']}/{top_pair['abnormal_tag']}"

        return {
            "pairs": [
                (pair["normal_prompt"], pair["abnormal_prompt"], float(pair["weight"]))
                for pair in selected_pairs
            ],
            "source": source,
            "route_debug": {
                "used_topk": bool(normal_info["used_semantic_topk"] or abnormal_info["used_semantic_topk"]),
                "normal_gate_inserted": bool(normal_info["gate_inserted"]),
                "abnormal_gate_inserted": bool(abnormal_info["gate_inserted"]),
                "pairs": selected_pairs,
            },
        }

    def set_icsr_bank(self, bank_embs: Dict[str, "torch.Tensor"]) -> None:
        """Install the ICSR class-embedding bank."""
        if not bank_embs:
            self._icsr_classes = []
            self._icsr_matrix = None
            return

        sorted_classes = sorted(bank_embs.keys())
        vectors: List["torch.Tensor"] = []
        ref_dim: Optional[int] = None
        for cls in sorted_classes:
            t = bank_embs[cls]
            if not isinstance(t, torch.Tensor):
                raise ValueError(f"ICSR bank[{cls}] must be a torch.Tensor, got {type(t)}")
            if t.dim() != 1:
                raise ValueError(
                    f"ICSR bank[{cls}] must be 1-D, got shape {tuple(t.shape)}"
                )
            if not t.is_cpu:
                raise RuntimeError(
                    f"ICSR bank[{cls}] must be on CPU, got device={t.device}"
                )
            if t.dtype != torch.float32:
                raise RuntimeError(
                    f"ICSR bank[{cls}] must be float32, got dtype={t.dtype}"
                )
            if ref_dim is None:
                ref_dim = int(t.shape[0])
            elif int(t.shape[0]) != ref_dim:
                raise ValueError(
                    f"ICSR bank[{cls}] dim {int(t.shape[0])} != ref dim {ref_dim}"
                )
            vectors.append(t)

        self._icsr_classes = sorted_classes
        self._icsr_matrix = torch.stack(vectors, dim=0).contiguous()

    def resolve_icsr(
        self,
        name: str,
        query_feat: "torch.Tensor",
        topk: int = 3,
        tau: float = 0.1,
        gate_entropy_threshold: float = 0.85,
        min_sim: float = 0.15,
        min_margin: float = 0.02,
    ) -> Dict[str, Any]:
        """Image-Conditioned Source-class Routing."""
        if not isinstance(query_feat, torch.Tensor):
            raise RuntimeError("query_feat must be a torch.Tensor")
        if query_feat.dim() != 1:
            raise RuntimeError(f"query_feat must be 1-D, got shape {tuple(query_feat.shape)}")
        if not query_feat.is_cpu:
            raise RuntimeError(f"query_feat must be CPU, got device={query_feat.device}")
        if query_feat.dtype != torch.float32:
            raise RuntimeError(f"query_feat must be float32, got dtype={query_feat.dtype}")

        if self._icsr_matrix is None or self._icsr_matrix.size(0) == 0:
            return {
                "pairs": None,
                "source": "icsr_gated_out",
                "meta": {
                    "enabled": True,
                    "gate_passed": False,
                    "gate_reason": "bank_empty",
                    "topk_sims": [],
                    "topk_classes": [],
                    "topk_weights": None,
                    "H_norm": None,
                    "top1_sim": float("nan"),
                    "top1_top2_margin": None,
                    "tau": float(tau),
                    "dedup_pair_count": 0,
                },
            }

        sims = self._icsr_matrix @ query_feat
        k_eff = min(int(max(1, topk)), int(sims.numel()))
        top_vals, top_idx = sims.topk(k_eff)
        top_classes: List[str] = [self._icsr_classes[i] for i in top_idx.tolist()]

        tau_safe = max(float(tau), 1e-6)
        logits = top_vals / tau_safe
        weights = torch.softmax(logits, dim=0)

        def _gated_out(reason: str, H_norm_val: Optional[float]) -> Dict[str, Any]:
            return {
                "pairs": None,
                "source": "icsr_gated_out",
                "meta": {
                    "enabled": True,
                    "gate_passed": False,
                    "gate_reason": reason,
                    "topk_sims": [float(v) for v in top_vals.tolist()],
                    "topk_classes": top_classes,
                    "topk_weights": [float(w) for w in weights.tolist()],
                    "H_norm": H_norm_val,
                    "top1_sim": float(top_vals[0].item()),
                    "top1_top2_margin": (
                        float((top_vals[0] - top_vals[1]).item()) if k_eff > 1 else None
                    ),
                    "tau": float(tau),
                    "dedup_pair_count": 0,
                },
            }

        if k_eff > 1:
            H = -(weights * torch.log(weights + 1e-12)).sum()
            H_norm = float((H / math.log(k_eff)).item())
        else:
            H_norm = 0.0

        if top_vals[0].item() < float(min_sim):
            return _gated_out("below_min_sim", H_norm)

        if k_eff > 1 and (top_vals[0] - top_vals[1]).item() < float(min_margin):
            return _gated_out("below_min_margin", H_norm)

        if H_norm > float(gate_entropy_threshold):
            return _gated_out("entropy_uniform", H_norm)

        tgt_name = self._normalize_name(name)
        pair_weights: Dict[Tuple[str, str], float] = {}
        for src_cls, w in zip(top_classes, weights.tolist()):
            n_key = ("normal", src_cls)
            a_key = ("abnormal", src_cls)
            if n_key not in self.cache or a_key not in self.cache:
                continue
            np_raw = self.cache[n_key]
            ap_raw = self.cache[a_key]
            np_adapted = self._adapt_prompt_to_target(np_raw, src_cls, tgt_name) or np_raw
            ap_adapted = self._adapt_prompt_to_target(ap_raw, src_cls, tgt_name) or ap_raw
            key = (np_adapted, ap_adapted)
            pair_weights[key] = pair_weights.get(key, 0.0) + float(w)

        if not pair_weights:
            logger.warning(
                "ICSR: no_role_pairs for '%s' — cache may have changed after set_icsr_bank; "
                "top_classes=%s", name, top_classes,
            )
            return _gated_out("no_role_pairs", H_norm)

        total = sum(pair_weights.values())
        pair_list: List[Tuple[str, str, float]] = [
            (n, a, float(w) / float(total)) for (n, a), w in pair_weights.items()
        ]
        pair_list.sort(key=lambda x: -x[2])

        return {
            "pairs": pair_list,
            "source": "icsr",
            "meta": {
                "enabled": True,
                "gate_passed": True,
                "gate_reason": "ok",
                "topk_sims": [float(v) for v in top_vals.tolist()],
                "topk_classes": top_classes,
                "topk_weights": [float(x) for x in weights.tolist()],
                "H_norm": H_norm,
                "top1_sim": float(top_vals[0].item()),
                "top1_top2_margin": (
                    float((top_vals[0] - top_vals[1]).item()) if k_eff > 1 else None
                ),
                "tau": float(tau),
                "dedup_pair_count": len(pair_list),
            },
        }

    @staticmethod
    def _shared_prompt_for_role(role: str) -> str:
        """Return the generic shared prompt for a role (no category name)."""
        if role == "normal":
            return "X normal object"
        if role == "abnormal":
            return "X anomalous object"
        return "X object"

    @staticmethod
    def _adapt_prompt_to_target(
        prompt: str, src_name: str, tgt_name: str,
    ) -> Optional[str]:
        """Replace the source-domain category name in a prompt with the target-domain one, keeping the template structure.

        "X the flawed hazelnut" + (hazelnut → cashew) → "X the flawed cashew"
        Category-agnostic templates (which contain no category name) are returned unchanged.
        """
        if not src_name or not tgt_name:
            return prompt
        # case-insensitive lookup of the source category name
        lower = prompt.lower()
        src_lower = src_name.lower()
        idx = lower.find(src_lower)
        if idx >= 0:
            return prompt[:idx] + tgt_name + prompt[idx + len(src_name):]
        # retry with underscores replaced by spaces (e.g. metal_nut -> metal nut)
        src_spaced = src_lower.replace("_", " ")
        if src_spaced != src_lower:
            idx = lower.find(src_spaced)
            if idx >= 0:
                return prompt[:idx] + tgt_name + prompt[idx + len(src_spaced):]
        # no source category name (an agnostic template): return the prompt unchanged
        return prompt

    def _extract_name(self, prompt: str) -> str:
        """Extract 'xxx' from 'X xxx'."""
        p = prompt.strip()
        if p.startswith("X "):
            return p[2:].strip()
        return p

    def _normalize_name(self, name: str) -> str:
        """Normalize the category-name format to improve rule hit rate."""
        if name is None:
            return ""
        n = str(name).strip().lower()
        if n.startswith("x "):
            n = n[2:].strip()
        return n

    def _candidate_names(self, name: str) -> List[str]:
        """Generate equivalent category-name candidates (space / underscore / hyphen variants)."""
        base = self._normalize_name(name)
        raw = [base]
        if base:
            raw.extend([
                base.replace(" ", "_"),
                base.replace("_", " "),
                base.replace("-", "_"),
                base.replace("-", " "),
            ])
        out = []
        seen = set()
        for n in raw:
            if n and n not in seen:
                out.append(n)
                seen.add(n)
        return out

    @staticmethod
    def _tokenize_prompt_text(text: str) -> List[str]:
        """Normalize a prompt fragment into a token sequence, preserving underscored/hyphenated words."""
        if text is None:
            return []
        return re.findall(r"[a-z0-9]+(?:[_-][a-z0-9]+)*", str(text).lower())

    _DONOR_ATTRIBUTE_TOKENS = {
        "normal": frozenset({
            "clean", "intact", "pristine", "flawless", "perfect", "uniform",
            "regular", "standard", "typical", "unblemished", "defect-free",
            "smooth", "whole", "quality", "fine",
        }),
        "abnormal": frozenset({
            "scratch", "scratched", "rough", "broken", "contaminated",
            "discolored", "missing", "crack", "cracked", "faulty",
            "damaged", "defective", "anomalous", "irregular", "stained",
            "corroded", "chipped", "uneven", "flawed",
        }),
    }
    _DONOR_TARGET_PRIORS = {
        "normal": frozenset({"clean", "intact", "uniform"}),
        "abnormal": frozenset({"damaged", "defective", "irregular"}),
    }

    def _extract_prompt_attributes(self, prompt: str, role: str) -> Set[str]:
        role_key = role or "shared"
        tokens = set(self._tokenize_prompt_text(prompt))
        tags = set(self._DONOR_ATTRIBUTE_TOKENS.get(role_key, set())) & tokens
        if role_key == "normal" and {"whole", "item"} <= tokens:
            tags.add("whole")
        return tags

    def _target_attribute_prior(self, target_name: str, role: str) -> Set[str]:
        role_key = role or "shared"
        prior = set(self._DONOR_TARGET_PRIORS.get(role_key, set()))
        target_tokens = set(self._tokenize_prompt_text(target_name))
        if role_key == "normal":
            if {"pipe", "cable", "wire"} & target_tokens:
                prior.update({"uniform", "smooth"})
            if {"capsule", "capsules", "pill"} & target_tokens:
                prior.update({"clean", "intact"})
        elif role_key == "abnormal":
            if {"pcb", "transistor"} & target_tokens:
                prior.update({"damaged", "faulty"})
            if {"wood", "leather", "carpet", "grid"} & target_tokens:
                prior.update({"rough", "stained"})
        return prior

    def _attribute_match_score(self, role: str, donor_tags: Set[str], target_prior: Set[str]) -> float:
        donor_tags = set(donor_tags or set())
        target_prior = set(target_prior or set())
        if not donor_tags or not target_prior:
            return 0.5
        role_key = role or "shared"
        opposite_role = "abnormal" if role_key == "normal" else "normal"
        opposite = set(self._DONOR_ATTRIBUTE_TOKENS.get(opposite_role, set()))
        matches = len(donor_tags & target_prior)
        contradictions = len(donor_tags & opposite)
        score = 0.5
        score += 0.35 * (matches / max(1, len(target_prior)))
        score -= 0.35 * (contradictions / max(1, len(donor_tags)))
        return _clip01(score)

    def _behavior_consistency_score(self, source_name: str, role: str) -> float:
        role_key = role or "shared"
        meta = self.rule_metadata.get((role_key, source_name))
        if meta is None and role_key != "shared":
            meta = self.rule_metadata.get(("shared", source_name))
        if not meta:
            return 0.5
        gain_src = float(meta.get("gain_src", 0.0) or 0.0)
        gain_cross = float(meta.get("gain_cross", 0.0) or 0.0)
        score_std = float(meta.get("score_std", 0.0) or 0.0)
        gain_src_norm = _clip01(0.5 + gain_src / 0.1)
        gain_cross_norm = _clip01(0.5 + gain_cross / 0.1)
        std_norm = _clip01(1.0 - score_std / 0.1)
        return _clip01(0.45 * gain_src_norm + 0.30 * gain_cross_norm + 0.25 * std_norm)

    def _donor_score(
        self,
        target_name: str,
        source_name: str,
        role: str,
        semantic_sim: float,
        prompt: str,
    ) -> Dict[str, float]:
        semantic_norm = _clip01(float(semantic_sim))
        donor_tags = self._extract_prompt_attributes(prompt, role)
        target_prior = self._target_attribute_prior(target_name, role)
        attribute_match = self._attribute_match_score(role, donor_tags, target_prior)
        behavior_consistency = self._behavior_consistency_score(source_name, role)
        donor_score = (
            self.donor_weight_sem * semantic_norm
            + self.donor_weight_attr * attribute_match
            + self.donor_weight_beh * behavior_consistency
        )
        return {
            "semantic_sim": semantic_norm,
            "attribute_match": _clip01(attribute_match),
            "behavior_consistency": _clip01(behavior_consistency),
            "donor_score": _clip01(donor_score),
        }

    @staticmethod
    def _template_structure(template: str) -> str:
        return str(template).replace("{name}", "").strip()

    def _extract_descriptors(self, prompt: str, name: str) -> List[str]:
        """Extract descriptor tokens from a prompt, relative to the category name."""
        name_tokens = set(self._tokenize_prompt_text(name))
        tokens = self._tokenize_prompt_text(prompt)
        out = []
        for tok in tokens:
            if tok == "x" or tok in name_tokens or tok in self._GENERIC_TEMPLATE_TOKENS:
                continue
            if tok not in out:
                out.append(tok)
        return out

    def _descriptor_pool(self, role: Optional[str], name: str) -> List[str]:
        role_key = role or "shared"
        base = []
        if role_key == "normal":
            base.extend(self.normal_adjectives)
        elif role_key == "abnormal":
            base.extend(self.abnormal_adjectives)
        else:
            base.extend(self.adjectives)

        counter = Counter()
        for key, prompt in self.cache.items():
            if not (isinstance(key, tuple) and len(key) == 2):
                continue
            cache_role, cache_name = key
            if cache_role != role_key:
                continue
            for desc in self._extract_descriptors(prompt, cache_name):
                counter[desc] += 1

        pool = list(dict.fromkeys(base + [d for d, _ in counter.most_common(12)]))
        return [p for p in pool if p and p not in set(self._tokenize_prompt_text(name))]

    def _compose_factorized_prompt(
        self,
        template: str,
        name: str,
        descriptors: Optional[List[str]] = None,
    ) -> str:
        # Category-agnostic template (no {name}): return directly, optionally appending a descriptor
        if "{name}" not in template:
            if descriptors:
                desc_phrase = " ".join(d for d in descriptors if d).strip()
                if desc_phrase:
                    return f"{template} {desc_phrase}".strip()
            return template.strip()

        # deduplicate the descriptors and drop adjectives already present in the template
        if descriptors:
            tpl_content_words = set(
                template.replace("{name}", "").lower().split()
            ) - {"x", "a", "an", "the", "of", "in", "for", "with"}
            seen = set()
            unique_desc = []
            for d in descriptors:
                d_lower = d.lower()
                if d_lower not in seen and d_lower not in tpl_content_words:
                    seen.add(d_lower)
                    unique_desc.append(d)
            descriptors = unique_desc
        desc_phrase = " ".join([d for d in (descriptors or []) if d]).strip()
        if desc_phrase:
            slot_name = f"{desc_phrase} {name}".strip()
        else:
            slot_name = name
        return template.format(name=slot_name).strip()

    def _generation_space(
        self,
        name: str,
        role: Optional[str] = None,
    ) -> Tuple[List[str], List[str]]:
        """Return role-conditioned templates and descriptors for control rows."""
        if role == 'normal':
            templates_to_use = self.normal_templates + self.templates
            adjectives_to_use = self.normal_adjectives + self.adjectives
        elif role == 'abnormal':
            templates_to_use = self.abnormal_templates + [
                t for t in self.templates
                if not any(w in t.lower().split() for w in _NORMAL_SEMANTIC_WORDS)
            ]
            adjectives_to_use = [
                a for a in (self.abnormal_adjectives + self.adjectives)
                if a.lower() not in _NORMAL_SEMANTIC_WORDS
            ]
        else:
            templates_to_use = self.templates
            adjectives_to_use = self.adjectives

        descriptor_pool = self._descriptor_pool(role, name)
        descriptors = list(dict.fromkeys(descriptor_pool + adjectives_to_use))
        if role == 'abnormal':
            descriptors = [
                d for d in descriptors
                if not (set(self._tokenize_prompt_text(d)) & _NORMAL_SEMANTIC_WORDS)
            ]
        return templates_to_use, descriptors

    def _sample_random_prompt(self, name: str, role: Optional[str] = None) -> str:
        templates_to_use, descriptor_pool = self._generation_space(name, role=role)
        agnostic_pool = (self.agnostic_normal_templates if role == "normal"
                         else self.agnostic_abnormal_templates
                         if role == "abnormal" else [])
        if agnostic_pool and random.random() < 0.15:
            tpl = random.choice(agnostic_pool)
        else:
            tpl = random.choice(templates_to_use)

        if not descriptor_pool or random.random() < 0.20:
            chosen: List[str] = []
        else:
            desc_n = 1 if random.random() < 0.75 else 2
            chosen = random.sample(descriptor_pool, k=min(desc_n, len(descriptor_pool)))
        return self._compose_factorized_prompt(tpl, name, descriptors=chosen)

    def _random_search_pool(
        self,
        name: str,
        role: Optional[str] = None,
        target_size: Optional[int] = None,
    ) -> List[str]:
        """Generate a source-only random prompt pool without mutation or selection."""
        target = int(target_size or self.population_size * (self.generations + 1))
        target = max(1, target)
        rng_state = None
        if self.evo_random_search_seed is not None:
            seed_material = f"{self.evo_random_search_seed}|{name}|{role or 'shared'}|{target}"
            seed_value = int(hashlib.sha256(seed_material.encode("utf-8")).hexdigest()[:16], 16)
            rng_state = random.getstate()
            random.seed(seed_value)

        try:
            pool: List[str] = []
            seen = set()

            def add(candidate: str) -> None:
                if candidate and candidate not in seen and len(pool) < target:
                    pool.append(candidate)
                    seen.add(candidate)

            for candidate in self._init_population(name, role=role):
                add(candidate)

            for _ in range(max(target * 40, 40)):
                if len(pool) >= target:
                    break
                add(self._sample_random_prompt(name, role=role))

            if len(pool) < target:
                templates_to_use, descriptor_pool = self._generation_space(name, role=role)
                for tpl in templates_to_use:
                    add(self._compose_factorized_prompt(tpl, name, descriptors=[]))
                    for desc in descriptor_pool:
                        add(self._compose_factorized_prompt(tpl, name, descriptors=[desc]))
                        if len(pool) >= target:
                            break
                    if len(pool) >= target:
                        break

            if len(pool) < target:
                logger.warning(
                    "Random prompt pool underfilled: %d/%d for '%s' role=%s",
                    len(pool), target, name, role or "shared",
                )
            return pool[:target]
        finally:
            if rng_state is not None:
                random.setstate(rng_state)

    def _generic_crossover(
        self,
        parent_a: str,
        parent_b: str,
        name: str,
        role: Optional[str] = None,
    ) -> str:
        """Standard GA crossover for EvoPrompt controls; not CoEvo role-safe."""
        templates_to_use, descriptor_pool = self._generation_space(name, role=role)
        mixed_desc: List[str] = []
        for parent in (parent_a, parent_b):
            for desc in self._extract_descriptors(parent, name):
                if desc and desc not in mixed_desc:
                    mixed_desc.append(desc)
        random.shuffle(mixed_desc)

        if mixed_desc and random.random() < 0.80:
            chosen = mixed_desc[: random.randint(1, min(2, len(mixed_desc)))]
        elif descriptor_pool:
            chosen = [random.choice(descriptor_pool)]
        else:
            chosen = []
        return self._compose_factorized_prompt(
            random.choice(templates_to_use),
            name,
            descriptors=chosen,
        )

    def _spawn_standard_offspring(
        self,
        elite: List[str],
        name: str,
        role: Optional[str],
        op_counts: Dict[str, int],
    ) -> str:
        if (
            self.evo_crossover_rate > 0.0
            and len(elite) >= 2
            and random.random() < self.evo_crossover_rate
        ):
            parent_a, parent_b = random.sample(list(elite), 2)
            op_counts["crossover"] = op_counts.get("crossover", 0) + 1
            return self._generic_crossover(parent_a, parent_b, name, role=role)

        op_counts["mutation"] = op_counts.get("mutation", 0) + 1
        return self._mutate(random.choice(elite), name, role=role)

    @staticmethod
    def _extract_score_list(score_result: Any, expected_len: int) -> List[float]:
        """Accept List / Tuple / Dict outputs from scoring_callback."""
        if isinstance(score_result, dict):
            scores = score_result.get("scores", None)
            if scores is None:
                raise ValueError("scoring_callback dict is missing the 'scores' field")
        elif isinstance(score_result, tuple):
            if len(score_result) == 0:
                raise ValueError("scoring_callback returned an empty tuple")
            scores = score_result[0]
        else:
            scores = score_result
        if len(scores) != expected_len:
            raise ValueError(
                f"scoring_callback returned a mismatched length: got {len(scores)} expected {expected_len}"
            )
        return [float(s) for s in scores]

    def resolve_cached_prompt(
        self,
        name: str,
        role: Optional[str] = None,
        default_prompt: Optional[str] = None,
        allow_role_fallback: bool = False,
    ) -> Tuple[Optional[str], Optional[Tuple[str, str]], str]:
        """
        Resolve prompts from the cache only; no evolutionary search.

        Returns: (prompt, hit_key, source)
        source:
        - exact: direct hit on (role, name)
        - alias: hit on an equivalent name, or the shared fallback
        - semantic_fallback: fall back to the semantically closest category within the same role
        - template_transfer: build the target-category prompt by transferring a source-domain template
        - shared_fallback: fall back to the shared prompt after both semantic fallback and template transfer fail
        - role_fallback: hit any existing rule under the same role (cross-category fallback)
        - missing: no usable rule in the cache
        - default: use the default prompt
        """
        role_key = role or "shared"
        name_candidates = self._candidate_names(name)

        # 1) exact / alias hit: prefer the role, allowing the shared fallback when that role is missing
        role_search = [role_key]
        if role_key != "shared":
            role_search.append("shared")
        elif role is None:
            role_search.extend(["normal", "abnormal"])

        for r in role_search:
            for cand_name in name_candidates:
                key = (r, cand_name)
                if key in self.cache:
                    if r == role_key and cand_name == self._normalize_name(name):
                        return self.cache[key], key, "exact"
                    return self.cache[key], key, "alias"

        # 2) semantic fallback: match the closest category in the same-role cache by CLIP text similarity
        fallback_roles = [role_key]
        if role_key != "shared":
            fallback_roles.append("shared")
        else:
            fallback_roles.extend(["normal", "abnormal"])

        semantic_hit = self._semantic_fallback_lookup(name_candidates, fallback_roles)
        if semantic_hit is not None:
            key, prompt = semantic_hit
            src_name = key[1]  # source category name (e.g. "hazelnut")
            tgt_name = self._normalize_name(name)  # target category name (e.g. "cashew")
            # for cross-domain runs, swap the source category name for the target one, keeping the template structure
            adapted = self._adapt_prompt_to_target(prompt, src_name, tgt_name)
            if adapted is not None and adapted != prompt:
                logger.info(
                    "Semantic transfer for '%s' (role=%s): '%s' -> '%s' [src=%s]",
                    name, role_key, prompt, adapted, src_name,
                )
                return adapted, key, "semantic_transfer"
            return prompt, key, "semantic_fallback"

        # 2.5) template transfer: extract a template from the source rules and apply the target category name
        if self.enable_template_transfer:
            template_donor = None
            semantic_seed = self._topk_semantic_fallback_lookup(
                name_candidates,
                fallback_roles,
                topk=1,
                tau=1.0,
            )
            if semantic_seed:
                template_donor = semantic_seed[0][0][1]
            result = self.resolve_via_template_transfer(name, role_key, donor_source=template_donor)
            if result is not None:
                prompt, src_info = result
                logger.info(
                    "Template transfer for '%s' (role=%s): '%s' [%s]",
                    name, role_key, prompt, src_info,
                )
                hit_key = (role_key, template_donor) if template_donor else None
                return prompt, hit_key, "template_transfer"

        # 2.8) semantic fallback was enabled but did not hit (rejected by threshold/margin, or no candidate):
        #      if template transfer also fails, fall back to the generic shared prompt.
        if self.enable_semantic_fallback and self.semantic_embed_fn is not None:
            shared = self._shared_prompt_for_role(role_key)
            logger.info(
                "Semantic fallback not matched for '%s' (role=%s); "
                "template transfer unavailable, using shared prompt: '%s'",
                name, role_key, shared,
            )
            return shared, None, "shared_fallback"

        # 3) role-level fallback: take any stable rule under the same role (test-time safety net only)
        if allow_role_fallback:
            for r in fallback_roles:
                role_items = [
                    (k, v) for k, v in self.cache.items()
                    if isinstance(k, tuple) and len(k) == 2 and k[0] == r
                ]
                if role_items:
                    role_items.sort(key=lambda kv: kv[0][1])
                    key, prompt = role_items[0]
                    return prompt, key, "role_fallback"

        # 4) default fallback (if one was provided)
        if default_prompt is not None:
            return default_prompt, None, "default"

        return None, None, "missing"

    def _init_population(self, name: str, role: Optional[str] = None) -> List[str]:
        """Initialize a batch of candidates from factorized templates plus descriptors.
        
        :param name: category name
        :param role: role ('normal', 'abnormal', or None)
        """
        cand = []

        # pick templates and adjectives according to the role
        if role == 'normal':
            templates_to_use = self.normal_templates + self.templates
            adjectives_to_use = self.normal_adjectives + self.adjectives
        elif role == 'abnormal':
            # drop generic templates and adjectives that carry normal semantics
            templates_to_use = self.abnormal_templates + [
                t for t in self.templates
                if not any(w in t.lower().split() for w in _NORMAL_SEMANTIC_WORDS)
            ]
            adjectives_to_use = [
                a for a in (self.abnormal_adjectives + self.adjectives)
                if a.lower() not in _NORMAL_SEMANTIC_WORDS
            ]
        else:
            templates_to_use = self.templates
            adjectives_to_use = self.adjectives

        descriptor_pool = self._descriptor_pool(role, name)

        # -- category-agnostic candidates --
        # only 1 (~6%) on the normal side to keep text_delta category-directed; ~20% on the abnormal side for generalization
        agnostic_pool = (self.agnostic_normal_templates if role == "normal"
                         else self.agnostic_abnormal_templates
                         if role == "abnormal" else [])
        if role == "normal":
            n_agnostic = min(len(agnostic_pool), 1)
        else:
            n_agnostic = min(len(agnostic_pool), max(1, self.population_size * 2 // 10))
        for ag_tpl in random.sample(agnostic_pool, k=min(n_agnostic, len(agnostic_pool))):
            cand.append(ag_tpl)

        # base template: keep the class slot
        for t in templates_to_use:
            cand.append(self._compose_factorized_prompt(t, name, descriptors=[]))

        # descriptor-factorized variant: emphasize the descriptor slot
        for desc in descriptor_pool[: max(2, self.population_size // 2)]:
            tpl = random.choice(templates_to_use)
            cand.append(self._compose_factorized_prompt(tpl, name, descriptors=[desc]))

        for _ in range(max(0, self.population_size - len(cand))):
            tpl = random.choice(templates_to_use)
            if descriptor_pool:
                desc_n = 1 if random.random() < 0.7 else 2
                chosen = random.sample(descriptor_pool, k=min(desc_n, len(descriptor_pool)))
            else:
                chosen = [random.choice(adjectives_to_use)]
            cand.append(self._compose_factorized_prompt(tpl, name, descriptors=chosen))
        
        # deduplicate
        uniq = []
        seen = set()
        for c in cand:
            if c not in seen:
                uniq.append(c)
                seen.add(c)
        return uniq[: self.population_size]

    def _score(self, candidates: List[str]) -> List[float]:
        """
        Text-prior scoring (a lightweight placeholder used when no scoring_callback is given):
        - prefer moderate length (too long risks truncation; too short carries little information)
        - penalize repeated words
        - keep the 'X ' prefix
        """
        scores = []
        for s in candidates:
            sc = 0.0
            sc += 1.0 if s.startswith("X ") else -1.0
            L = len(s.split())
            if 2 <= L <= 8:
                sc += 1.0
            sc -= max(0, L - 10) * 0.1
            # penalize repeated words
            toks = s.lower().split()
            sc -= (len(toks) - len(set(toks))) * 0.2
            scores.append(sc)
        return scores

    def _select(
        self, 
        candidates: List[str], 
        scoring_callback: Optional[Callable[[List[str], Optional[str]], List[float]]] = None,
        role: Optional[str] = None
    ) -> List[str]:
        """Select the top-k candidates.
        
        :param candidates: list of candidate prompts
        :param scoring_callback: scoring callback (preferred when given; otherwise the internal _score is used)
        :param role: role (passed through to scoring_callback)
        """
        if scoring_callback is not None:
            # use the external scoring callback (e.g. scoring based on anomaly scores)
            scores = scoring_callback(candidates, role=role)
            if len(scores) != len(candidates):
                raise ValueError(f"scoring_callback returned {len(scores)} scores for {len(candidates)} candidates")
        else:
            # use the internal text-prior scoring
            scores = self._score(candidates)
        
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [c for c, _ in ranked[: self.topk]]
    
    def _mmr_select(
        self,
        candidates: List[str],
        base_scores: List[float],
        k: int,
        role: Optional[str] = None
    ) -> List[str]:
        """Diversity-aware selection via Maximal Marginal Relevance (MMR).
        
        :param candidates: list of candidate prompts
        :param base_scores: list of base scores
        :param k: number of candidates to select
        :param role: role (used to fetch features from text_feat_cache)
        :return: the list of selected candidates
        """
        if len(candidates) <= k:
            # sort by score and return the first k
            order = sorted(range(len(candidates)), key=lambda i: base_scores[i], reverse=True)
            return [candidates[i] for i in order[:k]]
        
        if not hasattr(self, 'text_feat_cache') or len(self.text_feat_cache) == 0:
            # Without a text-feature cache, take the top-k by base_scores (consistent with this generation's model scores)
            order = sorted(range(len(candidates)), key=lambda i: base_scores[i], reverse=True)
            return [candidates[i] for i in order[:k]]
        
        # fetch text features from the cache
        selected = []
        remaining = list(range(len(candidates)))
        
        # step 1: take the highest-scoring candidate
        best_idx = max(remaining, key=lambda i: base_scores[i])
        selected.append(best_idx)
        remaining.remove(best_idx)
        
        # greedily select the remaining k-1
        for _ in range(k - 1):
            if len(remaining) == 0:
                break
            
            best_mmr = -float('inf')
            best_cand_idx = remaining[0]
            
            role_key = role or "shared"
            selected_feats = []
            for sel_idx in selected:
                cand = candidates[sel_idx]
                key = (role_key, cand)
                if key in self.text_feat_cache:
                    selected_feats.append(self.text_feat_cache[key])
            
            if len(selected_feats) == 0:
                # diversity cannot be computed; fall back to selecting by score
                best_cand_idx = max(remaining, key=lambda i: base_scores[i])
            else:
                # compute the MMR score of each remaining candidate
                if torch is not None:
                    # convert the feature to a tensor if it is not one already
                    selected_tensors = []
                    for feat in selected_feats:
                        if isinstance(feat, torch.Tensor):
                            selected_tensors.append(feat)
                        else:
                            # for numpy or other formats, try to convert
                            selected_tensors.append(torch.tensor(feat, dtype=torch.float32))
                    
                    if len(selected_tensors) > 0:
                        selected_tensor = torch.stack(selected_tensors)  # [|selected|, D]
                        
                        for cand_idx in remaining:
                            cand = candidates[cand_idx]
                            key = (role_key, cand)
                            if key in self.text_feat_cache:
                                cand_feat = self.text_feat_cache[key]  # [D]
                                # convert to a tensor if it is not one already
                                if not isinstance(cand_feat, torch.Tensor):
                                    cand_feat = torch.tensor(cand_feat, dtype=torch.float32)
                                
                                # maximum cosine similarity against the already-selected candidates
                                similarities = torch.nn.functional.cosine_similarity(
                                    cand_feat.unsqueeze(0), selected_tensor, dim=1
                                )
                                max_sim = similarities.max().item()
                                # MMR score = relevance - lambda * redundancy
                                mmr_score = base_scores[cand_idx] - self.lambda_diversity * max_sim
                                if mmr_score > best_mmr:
                                    best_mmr = mmr_score
                                    best_cand_idx = cand_idx
                            else:
                                # not in the cache: use the base score
                                if base_scores[cand_idx] > best_mmr:
                                    best_mmr = base_scores[cand_idx]
                                    best_cand_idx = cand_idx
                    else:
                        # features cannot be handled; fall back to selecting by score
                        best_cand_idx = max(remaining, key=lambda i: base_scores[i])
                else:
                    # torch unavailable; fall back to selecting by score
                    best_cand_idx = max(remaining, key=lambda i: base_scores[i])
            
            selected.append(best_cand_idx)
            remaining.remove(best_cand_idx)
        
        return [candidates[i] for i in selected]

    # ── LLM Mutation Methods ──

    _LLM_REPHRASE_TMPL = (
        "Rephrase this industrial inspection prompt in a different way. "
        "Keep under 8 words. Category: {name}. Role: {role}.\n"
        "Current: {current}\n"
        "Reply with ONLY the new prompt, nothing else."
    )

    _LLM_CREATIVE_TMPL = (
        "Write a short CLIP text prompt (under 8 words) for {role} {name} "
        "in manufacturing quality inspection. Be specific and creative.\n"
        "Reply with ONLY the prompt, nothing else."
    )

    _ABNORMAL_SEMANTIC_WORDS = frozenset({
        "defective", "damaged", "broken", "faulty", "cracked", "scratched",
        "flawed", "anomaly", "anomalous", "abnormal",
    })

    def _ensure_llm_loaded(self) -> bool:
        if self._llm_model is not None:
            return True
        if not self.llm_model_id:
            return False
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self._llm_tokenizer = AutoTokenizer.from_pretrained(
                self.llm_model_id, trust_remote_code=True)
            _device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            _dtype = torch.float16 if _device.type == "cuda" else torch.float32
            self._llm_model = AutoModelForCausalLM.from_pretrained(
                self.llm_model_id, torch_dtype=_dtype,
                trust_remote_code=True).to(_device)
            self._llm_device = _device
            self._llm_model.eval()
            logger.info("LLM mutation model loaded: %s", self.llm_model_id)
            return True
        except Exception as e:
            logger.warning("LLM mutation init failed: %s", e)
            self.llm_mutation_enabled = False
            return False

    _LLM_SYSTEM_PROMPT = (
        "You are a CLIP text prompt engineer for industrial anomaly detection. "
        "Generate short, precise prompts that start with 'X '. "
        "Reply with ONLY the prompt text, no explanation or formatting."
    )

    def _llm_generate(self, user_prompt: str) -> str:
        """Generate text from LLM. No caching here — cache at caller after validation."""
        if not self._ensure_llm_loaded():
            return ""
        try:
            messages = [
                {"role": "system", "content": self._LLM_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
            text = self._llm_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            inputs = self._llm_tokenizer(text, return_tensors="pt")
            inputs = {k: v.to(self._llm_device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self._llm_model.generate(
                    **inputs,
                    max_new_tokens=self.llm_mutation_max_tokens,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9,
                )
            generated = outputs[0][inputs["input_ids"].shape[1]:]
            return self._llm_tokenizer.decode(generated, skip_special_tokens=True).strip()
        except Exception as e:
            logger.debug("LLM generate failed: %s", e)
            return ""

    def _validate_llm_output(self, text: str, name: str, role: Optional[str]) -> Optional[str]:
        if not text:
            return None
        text = text.strip().strip('"').strip("'").strip()
        if not text:
            return None
        if not text.startswith("X "):
            text = "X " + text
        words = text.split()
        if len(words) < 2 or len(words) > 10:  # "X " + up to 8 content words
            return None
        body_words = text[2:].lower().split()
        if role == "abnormal" and any(w in body_words for w in _NORMAL_SEMANTIC_WORDS):
            return None
        if role == "normal" and any(w in body_words for w in self._ABNORMAL_SEMANTIC_WORDS):
            return None
        stopwords = {"a", "an", "the", "of", "with", "is", "in", "on", "for", "and", "or"}
        meaningful = [w for w in body_words if w not in stopwords and len(w) > 1]
        if len(meaningful) < 1:
            return None
        return text

    def _llm_rephrase(self, s: str, name: str, role: Optional[str]) -> Optional[str]:
        current = s[2:].strip() if s.startswith("X ") else s.strip()
        cache_key = f"rephrase|{role}|{name}|{current}"
        if cache_key in self._llm_cache:
            return self._llm_cache[cache_key]
        prompt = self._LLM_REPHRASE_TMPL.format(
            name=name, role=role or "shared", current=current)
        raw = self._llm_generate(prompt)
        result = self._validate_llm_output(raw, name, role)
        if result is not None:
            self._llm_cache[cache_key] = result
        return result

    def _llm_creative(self, name: str, role: Optional[str]) -> Optional[str]:
        cache_key = f"creative|{role}|{name}|{random.randint(0, 999)}"
        prompt = self._LLM_CREATIVE_TMPL.format(
            name=name, role=role or "shared")
        raw = self._llm_generate(prompt)
        result = self._validate_llm_output(raw, name, role)
        if result is not None:
            self._llm_cache[cache_key] = result
        return result

    def _mutate(self, s: str, name: str, role: Optional[str] = None) -> str:
        """Descriptor-factorized mutation strategy.

        :param s: the current prompt
        :param name: category name
        :param role: role ('normal', 'abnormal', or None)
        """
        # pick templates and adjectives according to the role
        if role == 'normal':
            templates_to_use = self.normal_templates + self.templates
            adjectives_to_use = self.normal_adjectives + self.adjectives
        elif role == 'abnormal':
            # drop generic templates and adjectives that carry normal semantics
            templates_to_use = self.abnormal_templates + [
                t for t in self.templates
                if not any(w in t.lower().split() for w in _NORMAL_SEMANTIC_WORDS)
            ]
            adjectives_to_use = [
                a for a in (self.abnormal_adjectives + self.adjectives)
                if a.lower() not in _NORMAL_SEMANTIC_WORDS
            ]
        else:
            templates_to_use = self.templates
            adjectives_to_use = self.adjectives

        descriptor_pool = self._descriptor_pool(role, name)
        current_desc = self._extract_descriptors(s, name)
        mutation_ops = list(self.evo_mutation_ops)
        if self.llm_mutation_enabled and not self._custom_mutation_ops:
            mutation_ops.extend(LLM_MUTATION_OPS)

        if self.llm_mutation_enabled and not self._custom_mutation_ops:
            op = random.choices(
                mutation_ops,
                weights=[1, 1, 1, 1, 1, 1, 0.5, 0.5],
            )[0]
        else:
            op = random.choice(mutation_ops)

        # LLM ops (with fallback to descriptor_replace on failure)
        if op == "llm_rephrase":
            result = self._llm_rephrase(s, name, role)
            if result is not None:
                logger.info("LLM_MUTATE|op=rephrase|role=%s|cat=%s|result=%s", role, name, result)
                self._llm_stats["llm_rephrase"] += 1
                return result
            self._llm_stats["llm_fallback"] += 1
            op = "descriptor_replace"  # fallback

        if op == "llm_creative":
            result = self._llm_creative(name, role)
            if result is not None:
                logger.info("LLM_MUTATE|op=creative|role=%s|cat=%s|result=%s", role, name, result)
                self._llm_stats["llm_creative"] += 1
                return result
            self._llm_stats["llm_fallback"] += 1
            op = "descriptor_replace"  # fallback

        self._llm_stats["rule_ops"] += 1

        if op == "descriptor_replace":
            tpl = random.choice(templates_to_use)
            chosen = [random.choice(descriptor_pool)] if descriptor_pool else [random.choice(adjectives_to_use)]
            return self._compose_factorized_prompt(tpl, name, descriptors=chosen)

        if op == "descriptor_insert":
            tpl = random.choice(templates_to_use)
            chosen = list(current_desc[:2]) if current_desc else []
            if descriptor_pool:
                chosen.append(random.choice(descriptor_pool))
            elif not chosen:
                chosen.append(random.choice(adjectives_to_use))
            chosen = list(dict.fromkeys(chosen))[:2]
            return self._compose_factorized_prompt(tpl, name, descriptors=chosen)

        if op == "descriptor_drop":
            tpl = random.choice(templates_to_use)
            if len(current_desc) > 1:
                chosen = current_desc[:-1]
            elif role is not None and current_desc:
                # When a role is set (normal/abnormal), keep at least one descriptor so normal never equals abnormal
                chosen = current_desc
            else:
                chosen = []
            return self._compose_factorized_prompt(tpl, name, descriptors=chosen)

        if op == "template_swap":
            # Category-agnostic template probability: 10% on the normal side (protects pixel), 20% on the abnormal side (aids generalization)
            agnostic_pool = (self.agnostic_normal_templates if role == "normal"
                             else self.agnostic_abnormal_templates
                             if role == "abnormal" else [])
            _agnostic_prob = 0.1 if role == "normal" else 0.2
            if agnostic_pool and random.random() < _agnostic_prob:
                return random.choice(agnostic_pool)
            tpl = random.choice(templates_to_use)
            chosen = current_desc[:2]
            return self._compose_factorized_prompt(tpl, name, descriptors=chosen)

        if op == "descriptor_mix":
            tpl = random.choice(templates_to_use)
            chosen = []
            if current_desc:
                chosen.append(random.choice(current_desc))
            if descriptor_pool:
                chosen.append(random.choice(descriptor_pool))
            chosen = list(dict.fromkeys(chosen))[:2]
            if not chosen:
                chosen = [random.choice(adjectives_to_use)]
            return self._compose_factorized_prompt(tpl, name, descriptors=chosen)

        synonyms = {
            "photo": ["image", "picture", "view"],
            "object": ["item", "thing", "sample"],
        }
        if role == 'normal':
            synonyms.update({
                "typical": ["standard", "normal", "common"],
                "clean": ["pristine", "perfect", "flawless"],
            })
        elif role == 'abnormal':
            synonyms.update({
                "defective": ["faulty", "damaged", "broken"],
            })
        else: # role is None (shared) or something else
            synonyms.update({
                "defective": ["faulty", "damaged", "broken"],
                "typical": ["standard", "normal", "common"],
                "clean": ["pristine", "perfect", "flawless"],
            })

        # for the abnormal role, skip synonym groups carrying normal semantics
        if role == 'abnormal':
            synonyms = {k: v for k, v in synonyms.items()
                        if k not in _NORMAL_SEMANTIC_WORDS}
            # also filter normal-semantics words out of the replacement candidates
            synonyms = {k: [w for w in v if w not in _NORMAL_SEMANTIC_WORDS]
                        for k, v in synonyms.items()}
            synonyms = {k: v for k, v in synonyms.items() if v}  # drop empty lists
        lowered = s.lower()
        for word, syns in synonyms.items():
            idx = lowered.find(word)
            if idx >= 0:
                replacement = random.choice(syns)
                return s[:idx] + replacement + s[idx + len(word):]

        tpl = random.choice(templates_to_use)
        return self._compose_factorized_prompt(tpl, name, descriptors=current_desc[:2])

    def optimize(
        self,
        prompts: List[str],
        scoring_callback: Optional[Callable[[List[str], Optional[str]], List[float]]] = None,
        role: Optional[str] = None,
        rerank_callback: Optional[Callable[[List[str], Optional[str]], List[float]]] = None,
        rerank_topk: int = 0,
        qd_archive: Optional[Any] = None,
        qd_bd_names: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Optimize prompts (text-level substitution).
        
        :param prompts: input list such as ['X bottle', 'X cable']
        :param scoring_callback: scoring callback with signature (candidates: List[str], role: Optional[str]) -> List[float]
        :param role: role ('normal', 'abnormal', or None)
        :return: a list of the same length, whose elements are the substituted texts
        """
        # Clear the text-feature cache at the start of each optimization run (avoids stale cross-batch features)
        self.text_feat_cache.clear()
        
        outs: List[str] = []
        role_key = role or "shared"
        
        for p in prompts:
            name = self._extract_name(p)
            cache_name = self._normalize_name(name)
            cache_key = (role_key, cache_name)

            # check the cache
            if cache_key in self.cache:
                outs.append(self.cache[cache_key])
                continue

            # At inference time, resolve missing categories from the existing cache first
                # (avoids pointless re-optimization in test mode)
            if self.inference_resolve_missing_from_cache:
                hit_prompt, _, src = self.resolve_cached_prompt(
                    cache_name,
                    role=role,
                    default_prompt=None,
                    allow_role_fallback=self.inference_allow_role_fallback,
                )
                if hit_prompt is not None and src in {"exact", "alias", "semantic_fallback", "template_transfer", "role_fallback", "shared_fallback"}:
                    # only exact and semantic matches are written back to the cache
                    # template_transfer / role_fallback / shared_fallback are not written back, to avoid contamination
                    if src in {"exact", "alias", "semantic_fallback"}:
                        self.cache[cache_key] = hit_prompt
                    outs.append(hit_prompt)
                    continue

            if self.evo_random_search:
                reference_budget = max(
                    self.population_size,
                    self.population_size * (self.generations + 1),
                )
                pool = self._random_search_pool(
                    cache_name,
                    role=role,
                    target_size=reference_budget,
                )
                stage2_logger.info(
                    "EQUAL_BUDGET_AUDIT|family=random_prompt_search|cat=%s|role=%s|evaluated_candidates=%d|reference_budget=%d|population=%d|generations=%d|mutation=0|crossover=0",
                    cache_name,
                    role_key,
                    len(pool),
                    reference_budget,
                    self.population_size,
                    self.generations,
                )
                if scoring_callback is not None:
                    _cb_result = scoring_callback(pool, role=role)
                    final_scores = self._extract_score_list(_cb_result, expected_len=len(pool))
                    if rerank_callback is not None and rerank_topk > 1:
                        topk_idx = sorted(
                            range(len(pool)),
                            key=lambda i: final_scores[i],
                            reverse=True,
                        )[: min(rerank_topk, len(pool))]
                        rerank_pool = [pool[i] for i in topk_idx]
                        rerank_scores = self._extract_score_list(
                            rerank_callback(rerank_pool, role=role),
                            expected_len=len(rerank_pool),
                        )
                        best = rerank_pool[max(range(len(rerank_pool)), key=lambda i: rerank_scores[i])]
                    else:
                        best = pool[max(range(len(pool)), key=lambda i: final_scores[i])]
                else:
                    local_scores = self._score(pool)
                    best = pool[max(range(len(pool)), key=lambda i: local_scores[i])]
                self.cache[cache_key] = best
                outs.append(best)
                continue

            # initialize the population
            pop = self._init_population(cache_name, role=role)
            pool = pop[:]
            
            _bd_names = qd_bd_names or ["image_auroc", "pixel_f1"]

            # evolutionary iterations
            for gen in range(self.generations):
                # selection (uses scoring_callback when provided)
                if scoring_callback is not None:
                    # call scoring_callback first to obtain scores (it also fills text_feat_cache)
                    _cb_result = scoring_callback(pool, role=role)
                    scores = self._extract_score_list(_cb_result, expected_len=len(pool))

                    # QD archive: insert scored candidates
                    # BD uses source-domain metrics; quality uses blended score
                    if qd_archive is not None:
                        _qd_metrics = _cb_result.get("src_metrics", []) if isinstance(_cb_result, dict) else []
                        if len(_qd_metrics) == len(pool):
                            for qi, (qc, qs) in enumerate(zip(pool, scores)):
                                qm = _qd_metrics[qi]
                                bd = tuple(float(qm.get(n, 0.0)) for n in _bd_names)
                                qd_archive.try_add(
                                    prompt=qc, quality=qs, descriptor=bd,
                                    metadata={"role": role_key, "name": cache_name, "gen": gen},
                                )

                    # Parent selection: QD archive sampling vs MMR
                    if qd_archive is not None and qd_archive.size >= self.topk:
                        elite = qd_archive.sample_parents(k=self.topk)
                    elif hasattr(self, 'text_feat_cache') and len(self.text_feat_cache) > 0:
                        elite = self._mmr_select(pool, scores, k=self.topk, role=role)
                    else:
                        ranked = sorted(zip(pool, scores), key=lambda x: x[1], reverse=True)
                        elite = [c for c, _ in ranked[:self.topk]]
                else:
                    # plain selection (internal text-prior scoring)
                    elite = self._select(pool, scoring_callback=None, role=role)
                
                # mutation / standard crossover to produce new candidates
                op_counts = {"mutation": 0, "crossover": 0}
                newc = [
                    self._spawn_standard_offspring(elite, cache_name, role, op_counts)
                    for _ in range(self.population_size - len(elite))
                ]
                pool = elite + newc

                # refill after deduplication to avoid population collapse
                pool = list(dict.fromkeys(pool))
                for _ in range(self.population_size * 3):
                    if len(pool) >= self.population_size:
                        break
                    pool.append(self._spawn_standard_offspring(elite, cache_name, role, op_counts))
                    pool = list(dict.fromkeys(pool))
                if len(pool) < self.population_size:
                    existing = set(pool)
                    for c in self._init_population(cache_name, role=role):
                        if c not in existing:
                            pool.append(c)
                            existing.add(c)
                        if len(pool) >= self.population_size:
                            break
                if len(pool) < self.population_size:
                    logger.warning("EvoPrompt pop backfill incomplete: %d/%d for '%s'",
                                   len(pool), self.population_size, cache_name)
                pool = pool[:self.population_size]
                stage2_logger.info(
                    "EVO_OP_COUNTS|gen=%d|cat=%s|role=%s|mutation=%d|crossover=%d|rate=%.3f",
                    gen,
                    cache_name,
                    role_key,
                    op_counts.get("mutation", 0),
                    op_counts.get("crossover", 0),
                    self.evo_crossover_rate,
                )

            # pick the final best candidate
            if scoring_callback is not None:
                _cb_final = scoring_callback(pool, role=role)
                final_scores = self._extract_score_list(_cb_final, expected_len=len(pool))
                # QD: also insert final-round candidates
                if qd_archive is not None:
                    _qd_final = _cb_final.get("src_metrics", []) if isinstance(_cb_final, dict) else []
                    if len(_qd_final) == len(pool):
                        for qi, (qc, qs) in enumerate(zip(pool, final_scores)):
                            qm = _qd_final[qi]
                            bd = tuple(float(qm.get(n, 0.0)) for n in _bd_names)
                            qd_archive.try_add(
                                prompt=qc, quality=qs, descriptor=bd,
                                metadata={"role": role_key, "name": cache_name, "gen": "final"},
                            )

                if rerank_callback is not None and rerank_topk > 1:
                    topk_idx = sorted(
                        range(len(pool)),
                        key=lambda i: final_scores[i],
                        reverse=True,
                    )[: min(rerank_topk, len(pool))]
                    rerank_pool = [pool[i] for i in topk_idx]
                    rerank_scores = self._extract_score_list(
                        rerank_callback(rerank_pool, role=role),
                        expected_len=len(rerank_pool),
                    )
                    best = rerank_pool[max(range(len(rerank_pool)), key=lambda i: rerank_scores[i])]
                else:
                    best_idx = max(range(len(pool)), key=lambda i: final_scores[i])
                    best = pool[best_idx]
            else:
                best = self._select(pool, scoring_callback=scoring_callback, role=role)[0]

            # QD: log archive summary (archive used for parent selection only;
            # final best comes from rerank/score pipeline to respect rerank and constraints)
            if qd_archive is not None and qd_archive.size > 0:
                logger.info(
                    "QD archive for '%s' (role=%s): %d/%d cells (%.0f%%), best_quality=%.4f",
                    cache_name, role_key, qd_archive.size, qd_archive.max_cells,
                    qd_archive.coverage * 100,
                    qd_archive.get_best().quality if qd_archive.get_best() else 0.0,
                )
            
            # cache the result
            self.cache[cache_key] = best
            outs.append(best)
        
        return outs
    
    def optimize_dual(
        self,
        prompts: List[str],
        scoring_callback: Optional[Callable[[List[str], Optional[str]], List[float]]] = None,
        rerank_callback: Optional[Callable[[List[str], Optional[str]], List[float]]] = None,
        rerank_topk: int = 0,
    ) -> Tuple[List[str], List[str]]:
        """
        Dual-branch optimization: optimize prompts separately for the normal and abnormal branches.
        
        :param prompts: input list such as ['X bottle', 'X cable']
        :param scoring_callback: scoring callback with signature (candidates: List[str], role: Optional[str]) -> List[float]
        :return: the tuple (optimized_normal_prompts, optimized_abnormal_prompts)
        """
        # optimize the normal branch first
        optimized_normal = self.optimize(
            prompts,
            scoring_callback=scoring_callback,
            role='normal',
            rerank_callback=rerank_callback,
            rerank_topk=rerank_topk,
        )
        
        # then optimize the abnormal branch
        optimized_abnormal = self.optimize(
            prompts,
            scoring_callback=scoring_callback,
            role='abnormal',
            rerank_callback=rerank_callback,
            rerank_topk=rerank_topk,
        )
        
        return optimized_normal, optimized_abnormal

    # ─── Template Transfer ───────────────────────────────────────────

    # Strongly normal-leaning words, used to filter out unreliable abnormal rules
    _NORMAL_INDICATOR_TOKENS = frozenset({
        "normal", "clean", "pristine", "flawless", "perfect", "good",
        "standard", "typical", "plain", "basic", "regular", "pure",
        "intact", "proper", "fine", "healthy", "unblemished",
        "defect-free", "quality",
    })
    _GENERIC_TEMPLATE_TOKENS = frozenset({
        "a", "an", "the", "of", "photo", "image", "view", "sample",
        "item", "object", "product",
    })

    def extract_transfer_templates(self) -> Dict[str, Dict[str, Any]]:
        """Extract transferable templates and their frequency statistics from the cached rules.

        For the abnormal role, rules whose descriptors carry normal semantics are filtered out.

        Returns:
            {
                "normal": {
                    "raw_templates": [...],
                    "unique_templates": [...],
                    "template_counts": {...},
                    "structure_counts": {...},
                },
                ...
            }
        """
        stats_by_role: Dict[str, Dict[str, Any]] = {
            "normal": {"raw_templates": []},
            "abnormal": {"raw_templates": []},
            "shared": {"raw_templates": []},
        }

        for key, prompt in self.cache.items():
            if not (isinstance(key, tuple) and len(key) == 2):
                continue
            role, source_name = key
            if role not in stats_by_role:
                continue
            prompt = str(prompt)

            # replace the source category name in the prompt with {name}
            matched_name = None
            prompt_lower = prompt.lower()
            for cand_name in self._candidate_names(source_name):
                if cand_name in prompt_lower:
                    matched_name = cand_name
                    break
            if matched_name is None:
                continue
            idx = prompt_lower.index(matched_name)
            template = prompt[:idx] + "{name}" + prompt[idx + len(matched_name):]

            # filter out unreliable abnormal templates
            if role == "abnormal":
                name_tokens = set(self._tokenize_prompt_text(matched_name))
                content_tokens = [
                    tok for tok in self._tokenize_prompt_text(prompt_lower)
                    if tok != "x" and tok not in name_tokens
                ]
                semantic_tokens = [
                    tok for tok in content_tokens
                    if tok not in self._GENERIC_TEMPLATE_TOKENS
                ]
                if set(semantic_tokens) & self._NORMAL_INDICATOR_TOKENS:
                    logger.debug("Filtered unreliable abnormal template: %s (source: %s)", prompt, source_name)
                    continue
                # drop bare template shells (e.g. "X capsule" / "X a photo of leather")
                if not semantic_tokens:
                    continue

            stats_by_role[role]["raw_templates"].append(template)

        for role, stats in stats_by_role.items():
            raw_templates = stats["raw_templates"]
            unique_templates = list(dict.fromkeys(raw_templates))
            template_counts = Counter(raw_templates)
            structure_counts = Counter(self._template_structure(t) for t in raw_templates)
            stats["unique_templates"] = unique_templates
            stats["template_counts"] = dict(template_counts)
            stats["structure_counts"] = dict(structure_counts)

        logger.info(
            "[TemplateTransfer] extracted templates: normal=%d(%d unique), abnormal=%d(%d unique), shared=%d(%d unique)",
            len(stats_by_role["normal"]["raw_templates"]),
            len(stats_by_role["normal"]["unique_templates"]),
            len(stats_by_role["abnormal"]["raw_templates"]),
            len(stats_by_role["abnormal"]["unique_templates"]),
            len(stats_by_role["shared"]["raw_templates"]),
            len(stats_by_role["shared"]["unique_templates"]),
        )
        return stats_by_role

    def resolve_via_template_transfer(
        self,
        name: str,
        role: str,
        donor_source: Optional[str] = None,
    ) -> Optional[Tuple[str, str]]:
        """Generate a prompt for a target-domain category via template transfer.

        A template pattern is extracted from the source-domain rules and applied to the target category name.
        A consensus strategy is used: templates are ranked by frequency and the most frequent one is chosen.

        Args:
            name: target category name
            role: role ('normal', 'abnormal', 'shared')

        Returns:
            (prompt, source_info), or None
        """
        if self._transfer_templates_cache is None:
            self._transfer_templates_cache = self.extract_transfer_templates()

        role_key = role or "shared"
        if donor_source:
            donor_prompt = self.cache.get((role_key, donor_source))
            if donor_prompt:
                for cand_name in self._candidate_names(donor_source):
                    donor_lower = str(donor_prompt).lower()
                    if cand_name in donor_lower:
                        idx = donor_lower.index(cand_name)
                        donor_template = (
                            str(donor_prompt)[:idx]
                            + "{name}"
                            + str(donor_prompt)[idx + len(cand_name):]
                        )
                        return donor_template.format(name=name), f"template_transfer(donor={donor_source})"
        role_stats = self._transfer_templates_cache.get(role_key, {})
        unique_templates = role_stats.get("unique_templates", [])
        raw_templates = role_stats.get("raw_templates", [])
        structure_counts = role_stats.get("structure_counts", {})
        if not unique_templates:
            return None

        # Pick the top template by true frequency; ties keep first-appearance order (unique_templates is already stable)
        valid_templates = [template for template in unique_templates if "{name}" in template]
        if not valid_templates:
            logger.warning("Template transfer skipped: no valid templates with {name} for role=%s", role_key)
            return None

        best_template = valid_templates[0]
        best_structure = self._template_structure(best_template)
        best_count = int(structure_counts.get(best_structure, 0))
        for template in valid_templates[1:]:
            structure = self._template_structure(template)
            count = int(structure_counts.get(structure, 0))
            if count > best_count:
                best_template = template
                best_structure = structure
                best_count = count

        prompt = best_template.format(name=name)
        raw_total = max(1, len(raw_templates))
        source = f"template_transfer(consensus={best_count}/{raw_total})"
        if donor_source:
            source = f"template_transfer(consensus_fallback={best_count}/{raw_total}, donor={donor_source})"
        return prompt, source

    def save_optimized_rules(self, save_path: str) -> None:
        """Save the optimized prompt rules to a file.
        
        :param save_path: output path (JSON format)
        """
        rules = {
            "cache": {f"{role}_{name}": prompt for (role, name), prompt in self.cache.items()},
            "metadata": {f"{role}_{name}": meta for (role, name), meta in self.rule_metadata.items()},
            "templates": self.templates,
            "normal_templates": self.normal_templates,
            "abnormal_templates": self.abnormal_templates,
            "adjectives": self.adjectives,
            "normal_adjectives": self.normal_adjectives,
            "abnormal_adjectives": self.abnormal_adjectives,
            "population_size": self.population_size,
            "generations": self.generations,
            "topk": self.topk,
            "lambda_diversity": self.lambda_diversity,
        }
        
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(rules, f, indent=2, ensure_ascii=False)
        print(f"[EvoPrompt] saved {len(self.cache)} optimized rules to {save_path}")

    def load_optimized_rules(self, load_path: str) -> None:
        """Load optimized prompt rules from a file.
        
        :param load_path: input path (JSON format)
        """
        if not os.path.exists(load_path):
            print(f"[EvoPrompt] rule file not found: {load_path}; falling back to the default templates")
            return
        
        with open(load_path, 'r', encoding='utf-8') as f:
            rules = json.load(f)
        
        # restore the cache (convert the "role_name" form back into a (role, name) tuple)
        cache_data = rules.get("cache", {})
        for key, prompt in cache_data.items():
            # parse the "role_name" form, e.g. "normal_bottle" -> ("normal", "bottle")
            parts = key.split("_", 1)
            if len(parts) == 2:
                role, name = parts
                self.cache[(role, self._normalize_name(name))] = prompt

        metadata_data = rules.get("metadata", {})
        for key, meta in metadata_data.items():
            parts = key.split("_", 1)
            if len(parts) != 2 or not isinstance(meta, dict):
                continue
            role, name = parts
            self.rule_metadata[(role, self._normalize_name(name))] = meta
        
        # optional: refresh the template pool if one was saved
        if "templates" in rules:
            self.templates = rules["templates"]
        if "normal_templates" in rules:
            self.normal_templates = rules["normal_templates"]
        if "abnormal_templates" in rules:
            self.abnormal_templates = rules["abnormal_templates"]
        self._transfer_templates_cache = None
        
        print(f"[EvoPrompt] loaded {len(self.cache)} optimized rules from {load_path}")

    def set_rule_metadata(self, name: str, role: Optional[str], metadata: Dict[str, Any]) -> None:
        role_key = role or "shared"
        self.rule_metadata[(role_key, self._normalize_name(name))] = dict(metadata)

    def get_rule_metadata(self, name: str, role: Optional[str] = None) -> Optional[Dict[str, Any]]:
        role_key = role or "shared"
        return self.rule_metadata.get((role_key, self._normalize_name(name)))

    def clear_cache(self) -> None:
        """Clear the cache."""
        self.cache.clear()
        self.rule_metadata.clear()
        self.text_feat_cache.clear()
        self._transfer_templates_cache = None
        print("[EvoPrompt] cache cleared")


def build_evo_prompt_optimizer(**kwargs) -> EvoPromptOptimizer:
    """
    Build the text-level EvoPrompt optimizer.
    
    :return: an EvoPromptOptimizer instance
    """
    return EvoPromptOptimizer(**kwargs)
