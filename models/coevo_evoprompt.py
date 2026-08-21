"""
Co-evolutionary EvoPrompt Optimizer (A-lite)

Co-evolution: joint optimization of normal and abnormal prompts.

Key ideas:
1. Evolve two populations (normal, abnormal) simultaneously.
2. Fitness combines paired AUROC with inter-branch contrast (consistency optional):
   Fitness = alpha*AUROC + beta*Contrast (+ gamma*Consistency)
3. Lightweight: score each branch once, then derive contrast from cached features only
   (no extra image inference).
4. Each normal is paired with only K abnormals, avoiding an O(N*M) blowup.

Extends EvoPrompt.
"""

import json
import os
import hashlib
import math
import random
import logging
from typing import Any, List, Tuple, Dict, Optional
from collections import defaultdict

try:
    import torch
except ImportError:
    torch = None

try:
    from .evoprompt import EvoPromptOptimizer, _NORMAL_SEMANTIC_WORDS
except ImportError:
    from evoprompt import EvoPromptOptimizer, _NORMAL_SEMANTIC_WORDS


# Keep CoEvo diagnostics on the main stage2 logger so INFO logs are visible
# under optimize_universal.py's logger setup.
logger = logging.getLogger("optimize_universal")


class CoEvoPromptOptimizer(EvoPromptOptimizer):
    """Co-evolutionary EvoPrompt optimizer.

    :param coevo_pair_k: number of abnormal prompts paired with each normal one (default 3)
    :param coevo_alpha_auroc: weight on AUROC (default 0.85)
    :param coevo_beta_contrast: weight on contrast (default 0.15)
    :param enable_consistency_guidance: whether to enable consistency guidance (default False)
    :param coevo_gamma_consistency: weight on consistency (default 0.10)
    """

    def __init__(
        self,
        population_size: int = 8,
        generations: int = 3,
        topk: int = 4,
        coevo_pair_k: int = 3,
        coevo_alpha_auroc: float = 0.85,
        coevo_beta_contrast: float = 0.25,
        enable_consistency_guidance: bool = False,
        coevo_gamma_consistency: float = 0.10,
        game_metrics_enable: bool = False,
        coevo_crossover_rate: float = 0.0,
        coevo_final_pair_select: str = "marginal",
        record_population_trace: bool = False,
        **kwargs
    ):
        super().__init__(
            population_size=population_size,
            generations=generations,
            topk=topk,
            **kwargs
        )

        self.coevo_pair_k = coevo_pair_k
        self.coevo_alpha_auroc = coevo_alpha_auroc
        self.coevo_beta_contrast = coevo_beta_contrast
        self.enable_consistency_guidance = enable_consistency_guidance
        self.coevo_gamma_consistency = coevo_gamma_consistency
        self.game_metrics_enable = game_metrics_enable
        self.coevo_crossover_rate = float(max(0.0, min(1.0, coevo_crossover_rate)))
        if coevo_final_pair_select not in {"marginal", "global_argmax"}:
            raise ValueError(
                "coevo_final_pair_select must be 'marginal' or 'global_argmax', "
                f"got {coevo_final_pair_select!r}"
            )
        self.coevo_final_pair_select = coevo_final_pair_select
        self.record_population_trace = bool(record_population_trace)
        self.population_trace: List[Dict[str, Any]] = []
        self.final_pair_selection_audit: List[Dict[str, Any]] = []
        self._warned_empty_text_feat_cache = False

        print("[CoEvo] co-evolutionary EvoPrompt initialized")
        print(f"  - pairing samples K: {coevo_pair_k}")
        print(f"  - alpha (AUROC weight): {coevo_alpha_auroc}")
        print(f"  - beta (contrast weight): {coevo_beta_contrast}")
        print(f"  - crossover rate: {self.coevo_crossover_rate:.3f}")
        print(f"  - final pair selector: {self.coevo_final_pair_select}")
        logger.info(
            "coevo_crossover_rate=%.3f coevo_final_pair_select=%s",
            self.coevo_crossover_rate,
            self.coevo_final_pair_select,
        )
        if self.record_population_trace:
            print("  - population trace: enabled")
            logger.info("coevo_record_population_trace=True")
        if self.enable_consistency_guidance:
            print(f"  - gamma (consistency weight): {coevo_gamma_consistency}")
            print("  - consistency guidance: enabled")
        else:
            print("  - consistency guidance: disabled")

    def _normal_semantic_tokens(self) -> set:
        tokens = set(_NORMAL_SEMANTIC_WORDS)
        tokens.update(getattr(self, "_NORMAL_INDICATOR_TOKENS", set()))
        return tokens

    def _has_normal_semantics(self, text: str) -> bool:
        return bool(set(self._tokenize_prompt_text(text)) & self._normal_semantic_tokens())

    def _score_prompt_pair(
        self,
        *,
        normal_prompt: str,
        abnormal_prompt: str,
        normal_score: float,
        abnormal_score: float,
        normal_consistency: float = 0.5,
        abnormal_consistency: float = 0.5,
    ) -> Tuple[float, Dict[str, float]]:
        """Score one normal/abnormal pair with the same CoEvo payoff terms."""
        if normal_prompt.strip() == abnormal_prompt.strip():
            return -1.0, {
                "auroc": 0.0,
                "contrast": 0.0,
                "consistency": 0.0,
                "penalty": -1.0,
            }

        cache = getattr(self, "text_feat_cache", {})
        n_feat = cache.get(("normal", normal_prompt))
        a_feat = cache.get(("abnormal", abnormal_prompt))
        cos_dist = (
            self._cosine_distance(n_feat, a_feat)
            if (n_feat is not None and a_feat is not None)
            else 0.0
        )
        a_term = self.coevo_alpha_auroc * 0.5 * (float(normal_score) + float(abnormal_score))
        c_term = self.coevo_beta_contrast * cos_dist
        cs_term = 0.0
        if self.enable_consistency_guidance:
            cs_term = self.coevo_gamma_consistency * 0.5 * (
                float(normal_consistency) + float(abnormal_consistency)
            )
        pair_score = a_term + c_term + cs_term
        return float(pair_score), {
            "auroc": float(a_term),
            "contrast": float(c_term),
            "consistency": float(cs_term),
            "penalty": 0.0,
        }

    def _select_global_pair_argmax(
        self,
        *,
        normal_pop: List[str],
        abnormal_pop: List[str],
        scores_n: List[float],
        scores_a: List[float],
        consistencies_n: Optional[List[float]] = None,
        consistencies_a: Optional[List[float]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Select the best pair over the full normal x abnormal cross-product."""
        if not normal_pop or not abnormal_pop:
            return None
        consistencies_n = consistencies_n or [0.5] * len(normal_pop)
        consistencies_a = consistencies_a or [0.5] * len(abnormal_pop)

        best: Optional[Dict[str, Any]] = None
        for ni, n_prompt in enumerate(normal_pop):
            for ai, a_prompt in enumerate(abnormal_pop):
                pair_score, pair_terms = self._score_prompt_pair(
                    normal_prompt=n_prompt,
                    abnormal_prompt=a_prompt,
                    normal_score=scores_n[ni],
                    abnormal_score=scores_a[ai],
                    normal_consistency=consistencies_n[ni],
                    abnormal_consistency=consistencies_a[ai],
                )
                record = {
                    "normal_index": int(ni),
                    "abnormal_index": int(ai),
                    "normal_prompt": n_prompt,
                    "abnormal_prompt": a_prompt,
                    "normal_score": float(scores_n[ni]),
                    "abnormal_score": float(scores_a[ai]),
                    "pair_score": float(pair_score),
                    "pair_terms": pair_terms,
                    "pair_matrix_size": int(len(normal_pop) * len(abnormal_pop)),
                }
                if best is None or record["pair_score"] > best["pair_score"]:
                    best = record
        return best

    def _role_safe_generation_space(self, name: str, role: Optional[str]):
        """Return templates and descriptor fallbacks that respect branch role."""
        role_key = role or "shared"
        if role_key == "normal":
            templates = self.normal_templates + self.templates
            descriptors = self._descriptor_pool("normal", name)
            fallbacks = self.normal_adjectives + self.adjectives
        elif role_key == "abnormal":
            templates = self.abnormal_templates + [
                t for t in self.templates
                if not self._has_normal_semantics(t)
            ]
            descriptors = [
                d for d in self._descriptor_pool("abnormal", name)
                if not self._has_normal_semantics(d)
            ]
            fallbacks = [
                a for a in (self.abnormal_adjectives + self.adjectives)
                if not self._has_normal_semantics(a)
            ]
        else:
            templates = self.templates
            descriptors = self._descriptor_pool(None, name)
            fallbacks = self.adjectives

        descriptors = list(dict.fromkeys([d for d in descriptors + fallbacks if d]))
        if role_key == "abnormal":
            descriptors = [d for d in descriptors if not self._has_normal_semantics(d)]
        return templates, descriptors

    def _role_safe_crossover(
        self,
        parent_a: str,
        parent_b: str,
        name: str,
        role: Optional[str] = None,
    ) -> str:
        """Same-role crossover: mix parent descriptors, then compose a role-safe child."""
        role_key = role or "shared"
        templates, descriptor_pool = self._role_safe_generation_space(name, role_key)

        mixed_desc: List[str] = []
        for parent in (parent_a, parent_b):
            for desc in self._extract_descriptors(parent, name):
                if role_key == "abnormal" and self._has_normal_semantics(desc):
                    continue
                if desc and desc not in mixed_desc:
                    mixed_desc.append(desc)

        random.shuffle(mixed_desc)
        if mixed_desc:
            chosen = mixed_desc[: random.randint(1, min(2, len(mixed_desc)))]
        elif descriptor_pool:
            chosen = [random.choice(descriptor_pool)]
        elif role_key == "abnormal":
            chosen = ["defective"]
        elif role_key == "normal":
            chosen = ["clean"]
        else:
            chosen = []

        template = random.choice(templates)
        child = self._compose_factorized_prompt(template, name, descriptors=chosen)
        if role_key == "abnormal" and self._has_normal_semantics(child):
            safe_desc = [d for d in descriptor_pool if not self._has_normal_semantics(d)]
            child = self._compose_factorized_prompt(
                random.choice(self.abnormal_templates),
                name,
                descriptors=[random.choice(safe_desc)] if safe_desc else ["defective"],
            )
        return child

    def _spawn_offspring(
        self,
        elite: List[str],
        name: str,
        role: str,
        op_counts: Dict[str, int],
    ) -> str:
        """Create one budget-matched offspring via mutation or same-role crossover."""
        if (
            self.coevo_crossover_rate > 0.0
            and len(elite) >= 2
            and random.random() < self.coevo_crossover_rate
        ):
            parent_a, parent_b = random.sample(list(elite), 2)
            op_counts["crossover"] = op_counts.get("crossover", 0) + 1
            return self._role_safe_crossover(parent_a, parent_b, name, role=role)

        op_counts["mutation"] = op_counts.get("mutation", 0) + 1
        return self._mutate(random.choice(elite), name, role=role)

    def _split_scores_result(self, result, candidates, role: str):
        """Parse the scoring_callback return value, keeping only scores / consistencies."""
        consistencies = None
        if isinstance(result, dict):
            scores = result.get("scores", None)
            consistencies = result.get("consistencies", None)
            if scores is None:
                raise ValueError(f"scoring_callback dict is missing the 'scores' field, role={role}")
        else:
            scores = result

        if len(scores) != len(candidates):
            raise ValueError(
                f"scoring_callback returned a mismatched length: role={role}, "
                f"scores={len(scores)} vs candidates={len(candidates)}"
            )
        scores = [float(s) for s in scores]

        if consistencies is not None:
            if len(consistencies) != len(candidates):
                raise ValueError(
                    f"consistencies length mismatch: role={role}, "
                    f"consistencies={len(consistencies)} vs candidates={len(candidates)}"
                )
            consistencies = [float(c) for c in consistencies]

        if not self.enable_consistency_guidance:
            consistencies = None

        return scores, consistencies

    def _cosine_distance(self, feat_a, feat_b) -> float:
        """Cosine distance 1 - cos_sim; larger means less similar."""
        if torch is None or feat_a is None or feat_b is None:
            return 0.0

        a = feat_a if isinstance(feat_a, torch.Tensor) else torch.tensor(feat_a, dtype=torch.float32)
        b = feat_b if isinstance(feat_b, torch.Tensor) else torch.tensor(feat_b, dtype=torch.float32)

        a = a.detach().float().flatten()
        b = b.detach().float().flatten()
        if a.numel() == 0 or b.numel() == 0:
            return 0.0

        if b.device != a.device:
            b = b.to(a.device)

        sim = torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=1).item()
        return 1.0 - sim

    def _diversity_sampling(
        self,
        pool: List[str],
        scores: List[float],
        role: str,
        exclude: Optional[List[str]] = None,
        k: int = 3
    ) -> List[int]:
        """Diversity sampling: pick k indices from the pool, balancing score and diversity."""
        exclude = exclude or []
        n = len(pool)
        if n <= k:
            return list(range(n))

        cache = getattr(self, "text_feat_cache", {})
        if not cache and not self._warned_empty_text_feat_cache:
            logger.warning(
                "CoEvo text_feat_cache is empty; diversity sampling falls back to score-only selection."
            )
            self._warned_empty_text_feat_cache = True

        # Sort by score, then start from a random top candidate so one individual never dominates
        idx_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        top_seed_n = min(max(3, k), len(idx_scores))
        seed_idx = random.choice([idx_scores[i][0] for i in range(top_seed_n)])
        selected = [seed_idx]
        remaining = [i for i in range(n) if i != selected[0]]

        for _ in range(k - 1):
            if not remaining:
                break
            best_idx = remaining[0]
            best_mmr = -float("inf")

            for idx in remaining:
                cand = pool[idx]
                key = (role, cand)
                if key not in getattr(self, "text_feat_cache", {}):
                    # no cache; select by score
                    if scores[idx] > best_mmr:
                        best_mmr = scores[idx]
                        best_idx = idx
                    continue

                cand_feat = self.text_feat_cache[key]
                min_sim = 1.0
                for sel_idx in selected:
                    sel_cand = pool[sel_idx]
                    sel_key = (role, sel_cand)
                    if sel_key in self.text_feat_cache:
                        d = self._cosine_distance(cand_feat, self.text_feat_cache[sel_key])
                        min_sim = min(min_sim, 1.0 - d)  # sim = 1 - dist
                dist = 1.0 - min_sim
                mmr = scores[idx] + 0.2 * dist
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = idx

            selected.append(best_idx)
            remaining.remove(best_idx)

        return selected[:k]

    def _aggregate_pair_scores(
        self,
        normal_pop: List[str],
        abnormal_pop: List[str],
        scores_n: List[float],
        scores_a: List[float],
        consistencies_n: Optional[List[float]] = None,
        consistencies_a: Optional[List[float]] = None,
        pair_records_out: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[Dict[str, float], Dict[str, float],
               Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
        """
        Lightweight co-pairing: each normal is paired with only K abnormals, and contrast
        is computed from the cache. scoring_callback is not called here (the caller already
        computed scores_n and scores_a once).

        Returns:
            (norm_n, norm_a, decomp_n, decomp_a)
            decomp_*: per-prompt decomposed terms {"auroc": ..., "contrast": ..., "consistency": ...}
        """
        agg_n = defaultdict(float)
        agg_a = defaultdict(float)
        count_n = defaultdict(int)
        count_a = defaultdict(int)
        # Decomposed term accumulators
        auroc_n = defaultdict(float)
        auroc_a = defaultdict(float)
        contrast_n = defaultdict(float)
        contrast_a = defaultdict(float)
        consistency_n_acc = defaultdict(float)
        consistency_a_acc = defaultdict(float)
        penalty_n = defaultdict(float)
        penalty_a = defaultdict(float)

        cache = getattr(self, "text_feat_cache", {})
        k = min(self.coevo_pair_k, len(abnormal_pop))
        adjusted_scores_n = [float(s) for s in scores_n]
        adjusted_scores_a = [float(s) for s in scores_a]
        if consistencies_n is None:
            consistencies_n = [0.5] * len(normal_pop)
        if consistencies_a is None:
            consistencies_a = [0.5] * len(abnormal_pop)

        # diversity-sample the abnormal indices
        abn_indices = self._diversity_sampling(
            abnormal_pop, adjusted_scores_a, "abnormal", k=k
        )

        for ni, n_prompt in enumerate(normal_pop):
            n_feat = cache.get(("normal", n_prompt))
            for ai in abn_indices:
                a_prompt = abnormal_pop[ai]
                a_feat = cache.get(("abnormal", a_prompt))

                # Core penalty: give a very low score when the normal and abnormal prompts are
                # identical, letting the GA weed them out instead of hard-coding string rules
                if n_prompt.strip() == a_prompt.strip():
                    pair_score = -1.0
                    a_term = 0.0
                    c_term = 0.0
                    cs_term = 0.0
                    p_term = -1.0
                else:
                    cos_dist = self._cosine_distance(n_feat, a_feat) if (n_feat is not None and a_feat is not None) else 0.0
                    a_term = self.coevo_alpha_auroc * 0.5 * (adjusted_scores_n[ni] + adjusted_scores_a[ai])
                    c_term = self.coevo_beta_contrast * cos_dist
                    cs_term = 0.0
                    p_term = 0.0
                    if self.enable_consistency_guidance:
                        cs_term = self.coevo_gamma_consistency * 0.5 * (consistencies_n[ni] + consistencies_a[ai])
                    pair_score = a_term + c_term + cs_term + p_term

                agg_n[n_prompt] += pair_score
                agg_a[a_prompt] += pair_score
                count_n[n_prompt] += 1
                count_a[a_prompt] += 1
                auroc_n[n_prompt] += a_term
                auroc_a[a_prompt] += a_term
                contrast_n[n_prompt] += c_term
                contrast_a[a_prompt] += c_term
                consistency_n_acc[n_prompt] += cs_term
                consistency_a_acc[a_prompt] += cs_term
                penalty_n[n_prompt] += p_term
                penalty_a[a_prompt] += p_term
                if pair_records_out is not None:
                    pair_records_out.append(
                        {
                            "normal_index": int(ni),
                            "abnormal_index": int(ai),
                            "normal_prompt": n_prompt,
                            "abnormal_prompt": a_prompt,
                            "normal_score": float(adjusted_scores_n[ni]),
                            "abnormal_score": float(adjusted_scores_a[ai]),
                            "point_score": float(pair_score),
                            "pair_score": float(pair_score),
                            "pair_terms": {
                                "auroc": float(a_term),
                                "contrast": float(c_term),
                                "consistency": float(cs_term),
                                "penalty": float(p_term),
                            },
                        }
                    )

        # Normalization: if a candidate was never sampled for pairing, fall back to its raw
        # AUROC score so it is not wrongly driven to 0
        norm_n: Dict[str, float] = {}
        decomp_n: Dict[str, Dict[str, float]] = {}
        for i, p in enumerate(normal_pop):
            if count_n[p] > 0:
                cnt = count_n[p]
                norm_n[p] = agg_n[p] / cnt
                decomp_n[p] = {
                    "auroc": auroc_n[p] / cnt,
                    "contrast": contrast_n[p] / cnt,
                    "consistency": consistency_n_acc[p] / cnt,
                    "penalty": penalty_n[p] / cnt,
                }
            elif self.enable_consistency_guidance:
                norm_n[p] = self.coevo_alpha_auroc * adjusted_scores_n[i] + self.coevo_gamma_consistency * consistencies_n[i]
                decomp_n[p] = {"auroc": self.coevo_alpha_auroc * adjusted_scores_n[i], "contrast": 0.0, "consistency": self.coevo_gamma_consistency * consistencies_n[i], "penalty": 0.0}
            else:
                norm_n[p] = adjusted_scores_n[i]
                decomp_n[p] = {"auroc": adjusted_scores_n[i], "contrast": 0.0, "consistency": 0.0, "penalty": 0.0}

        norm_a: Dict[str, float] = {}
        decomp_a: Dict[str, Dict[str, float]] = {}
        for i, p in enumerate(abnormal_pop):
            if count_a[p] > 0:
                cnt = count_a[p]
                norm_a[p] = agg_a[p] / cnt
                decomp_a[p] = {
                    "auroc": auroc_a[p] / cnt,
                    "contrast": contrast_a[p] / cnt,
                    "consistency": consistency_a_acc[p] / cnt,
                    "penalty": penalty_a[p] / cnt,
                }
            elif self.enable_consistency_guidance:
                norm_a[p] = self.coevo_alpha_auroc * adjusted_scores_a[i] + self.coevo_gamma_consistency * consistencies_a[i]
                decomp_a[p] = {"auroc": self.coevo_alpha_auroc * adjusted_scores_a[i], "contrast": 0.0, "consistency": self.coevo_gamma_consistency * consistencies_a[i], "penalty": 0.0}
            else:
                norm_a[p] = adjusted_scores_a[i]
                decomp_a[p] = {"auroc": adjusted_scores_a[i], "contrast": 0.0, "consistency": 0.0, "penalty": 0.0}

        return norm_n, norm_a, decomp_n, decomp_a

    def _population_member_records(
        self,
        population: List[str],
        scores: List[float],
        fitness: List[float],
        elite: List[str],
    ) -> List[Dict[str, Any]]:
        elite_set = set(elite or [])
        return [
            {
                "index": int(i),
                "prompt": prompt,
                "score": float(scores[i]) if i < len(scores) else None,
                "fitness": float(fitness[i]) if i < len(fitness) else None,
                "is_elite": prompt in elite_set,
            }
            for i, prompt in enumerate(population)
        ]

    def _record_population_snapshot(
        self,
        *,
        category: str,
        generation,
        stage: str,
        normal_pop: List[str],
        abnormal_pop: List[str],
        scores_n: List[float],
        scores_a: List[float],
        norm_fitness_n: List[float],
        norm_fitness_a: List[float],
        normal_elite: List[str],
        abnormal_elite: List[str],
        pair_records: List[Dict[str, Any]],
    ) -> None:
        if not self.record_population_trace:
            return

        gen_token = str(generation).replace(" ", "_")
        pair_payload: List[Dict[str, Any]] = []
        for idx, record in enumerate(pair_records):
            item = dict(record)
            item.setdefault("class_name", category)
            item.setdefault("category", category)
            item.setdefault("generation", generation)
            item.setdefault("stage", stage)
            item.setdefault(
                "candidate_id",
                (
                    f"{category}:{stage}:g{gen_token}:"
                    f"n{item.get('normal_index', idx)}:a{item.get('abnormal_index', idx)}:p{idx}"
                ),
            )
            pair_payload.append(item)

        self.population_trace.append(
            {
                "schema_version": "coevo_population_trace_v1",
                "category": category,
                "generation": generation,
                "stage": stage,
                "population_size": int(self.population_size),
                "topk": int(self.topk),
                "coevo_pair_k": int(self.coevo_pair_k),
                "normal_population": self._population_member_records(
                    normal_pop, scores_n, norm_fitness_n, normal_elite
                ),
                "abnormal_population": self._population_member_records(
                    abnormal_pop, scores_a, norm_fitness_a, abnormal_elite
                ),
                "pair_candidates": pair_payload,
            }
        )

    def _compute_game_metrics(
        self,
        gen: int,
        category: str,
        normal_pop: List[str],
        abnormal_pop: List[str],
        scores_n: List[float],
        scores_a: List[float],
        norm_fitness_n: List[float],
        norm_fitness_a: List[float],
        decomp_n: Dict[str, Dict[str, float]],
        decomp_a: Dict[str, Dict[str, float]],
        normal_elite: List[str],
        abnormal_elite: List[str],
        prev_elite_hash_n: Optional[str],
        prev_elite_hash_a: Optional[str],
        prev_payoff_n: Optional[float],
        prev_payoff_a: Optional[float],
        pop_size_n: Optional[int] = None,
        pop_size_a: Optional[int] = None,
    ) -> Dict:
        """Compute and log per-generation game-theory metrics."""
        # Elite payoff stats
        elite_fitness_n = [norm_fitness_n[i] for i, p in enumerate(normal_pop) if p in set(normal_elite)]
        elite_fitness_a = [norm_fitness_a[i] for i, p in enumerate(abnormal_pop) if p in set(abnormal_elite)]
        payoff_n = float(sum(elite_fitness_n) / max(1, len(elite_fitness_n)))
        payoff_a = float(sum(elite_fitness_a) / max(1, len(elite_fitness_a)))
        payoff_n_max = float(max(norm_fitness_n)) if norm_fitness_n else 0.0
        payoff_a_max = float(max(norm_fitness_a)) if norm_fitness_a else 0.0

        # Decomposed terms (population-level means)
        auroc_vals_n = [decomp_n[p]["auroc"] for p in normal_pop if p in decomp_n]
        contrast_vals_n = [decomp_n[p]["contrast"] for p in normal_pop if p in decomp_n]
        auroc_vals_a = [decomp_a[p]["auroc"] for p in abnormal_pop if p in decomp_a]
        contrast_vals_a = [decomp_a[p]["contrast"] for p in abnormal_pop if p in decomp_a]
        auroc_n_mean = float(sum(auroc_vals_n) / max(1, len(auroc_vals_n)))
        contrast_n_mean = float(sum(contrast_vals_n) / max(1, len(contrast_vals_n)))
        auroc_a_mean = float(sum(auroc_vals_a) / max(1, len(auroc_vals_a)))
        contrast_a_mean = float(sum(contrast_vals_a) / max(1, len(contrast_vals_a)))

        # Feature hit rate and role distance
        cache = getattr(self, "text_feat_cache", {})
        n_hits = sum(1 for p in normal_pop if ("normal", p) in cache)
        a_hits = sum(1 for p in abnormal_pop if ("abnormal", p) in cache)
        feat_hit_rate = float((n_hits + a_hits) / max(1, len(normal_pop) + len(abnormal_pop)))

        best_n = normal_pop[max(range(len(normal_pop)), key=lambda i: norm_fitness_n[i])] if normal_pop else ""
        best_a = abnormal_pop[max(range(len(abnormal_pop)), key=lambda i: norm_fitness_a[i])] if abnormal_pop else ""
        n_feat = cache.get(("normal", best_n))
        a_feat = cache.get(("abnormal", best_a))
        if n_feat is not None and a_feat is not None:
            role_dist = self._cosine_distance(n_feat, a_feat)
        else:
            role_dist = float("nan")

        # Elite change detection
        elite_hash_n = hashlib.md5("|".join(sorted(normal_elite)).encode()).hexdigest()[:8]
        elite_hash_a = hashlib.md5("|".join(sorted(abnormal_elite)).encode()).hexdigest()[:8]
        elite_chg_n = 0 if (prev_elite_hash_n is not None and elite_hash_n == prev_elite_hash_n) else 1
        elite_chg_a = 0 if (prev_elite_hash_a is not None and elite_hash_a == prev_elite_hash_a) else 1

        # Payoff delta
        if prev_payoff_n is not None and prev_payoff_a is not None:
            payoff_delta = abs(payoff_n - prev_payoff_n) + abs(payoff_a - prev_payoff_a)
        else:
            payoff_delta = 0.0

        _pop_n = pop_size_n if pop_size_n is not None else len(normal_pop)
        _pop_a = pop_size_a if pop_size_a is not None else len(abnormal_pop)

        logger.info(
            "GAME_METRICS|gen=%d|cat=%s|payoff_n=%.6f|payoff_a=%.6f"
            "|payoff_n_max=%.6f|payoff_a_max=%.6f"
            "|auroc_n=%.6f|contrast_n=%.6f|auroc_a=%.6f|contrast_a=%.6f"
            "|role_dist=%.4f|feat_hit=%.4f"
            "|elite_chg_n=%d|elite_chg_a=%d|payoff_delta=%.6f"
            "|pop_n=%d|pop_a=%d",
            gen, category,
            payoff_n, payoff_a, payoff_n_max, payoff_a_max,
            auroc_n_mean, contrast_n_mean, auroc_a_mean, contrast_a_mean,
            role_dist, feat_hit_rate,
            elite_chg_n, elite_chg_a, payoff_delta,
            _pop_n, _pop_a,
        )

        return {
            "payoff_n_mean": payoff_n,
            "payoff_a_mean": payoff_a,
            "elite_hash_n": elite_hash_n,
            "elite_hash_a": elite_hash_a,
        }

    def optimize_dual(
        self,
        prompts: List[str],
        scoring_callback,
        rerank_callback=None,
        rerank_topk: int = 0,
        qd_archive_normal=None,
        qd_archive_abnormal=None,
        qd_bd_names: Optional[List[str]] = None,
    ) -> Tuple[List[str], List[str]]:
        """
        Co-evolutionary dual-branch optimization.

        Key point: score each branch once (no repeated calls); contrast comes from
        text_feat_cache alone.
        """
        if hasattr(self, "text_feat_cache"):
            self.text_feat_cache.clear()
        self._warned_empty_text_feat_cache = False
        _bd_names = qd_bd_names or ["image_auroc", "pixel_f1"]

        normal_optimized = []
        abnormal_optimized = []

        for p in prompts:
            name = self._extract_name(p)
            cache_key_n = ("normal", name)
            cache_key_a = ("abnormal", name)

            if cache_key_n in self.cache and cache_key_a in self.cache:
                normal_optimized.append(self.cache[cache_key_n])
                abnormal_optimized.append(self.cache[cache_key_a])
                continue

            normal_pop = self._init_population(name, role="normal")
            abnormal_pop = self._init_population(name, role="abnormal")

            # Game metrics state across generations
            _prev_elite_hash_n: Optional[str] = None
            _prev_elite_hash_a: Optional[str] = None
            _prev_payoff_n: Optional[float] = None
            _prev_payoff_a: Optional[float] = None

            for gen in range(self.generations):
                # 1. Score each branch once, without repeated calls (scoring_callback fills text_feat_cache)
                _cb_raw_n = scoring_callback(normal_pop, role="normal")
                scores_n, consistencies_n = self._split_scores_result(
                    _cb_raw_n, normal_pop, "normal"
                )
                _qm_n = _cb_raw_n.get("src_metrics", []) if isinstance(_cb_raw_n, dict) else []

                _cb_raw_a = scoring_callback(abnormal_pop, role="abnormal")
                scores_a, consistencies_a = self._split_scores_result(
                    _cb_raw_a, abnormal_pop, "abnormal"
                )
                _qm_a = _cb_raw_a.get("src_metrics", []) if isinstance(_cb_raw_a, dict) else []

                # 2. Lightweight co-pairing (cache only, no extra inference)
                pair_records: List[Dict[str, Any]] = []
                agg_n, agg_a, decomp_n, decomp_a = self._aggregate_pair_scores(
                    normal_pop, abnormal_pop, scores_n, scores_a,
                    consistencies_n=consistencies_n, consistencies_a=consistencies_a,
                    pair_records_out=pair_records if self.record_population_trace else None,
                )

                norm_fitness_n = [agg_n.get(p, scores_n[i]) for i, p in enumerate(normal_pop)]
                norm_fitness_a = [agg_a.get(p, scores_a[i]) for i, p in enumerate(abnormal_pop)]

                # QD: insert scored candidates after coevo pairing so archive quality
                # matches the actual blended fitness used for selection.
                if qd_archive_normal is not None and len(_qm_n) == len(normal_pop):
                    for qi, (qc, qf) in enumerate(zip(normal_pop, norm_fitness_n)):
                        bd = tuple(float(_qm_n[qi].get(n, 0.0)) for n in _bd_names)
                        qd_archive_normal.try_add(
                            prompt=qc, quality=qf, descriptor=bd,
                            metadata={"role": "normal", "name": name, "gen": gen},
                        )
                if qd_archive_abnormal is not None and len(_qm_a) == len(abnormal_pop):
                    for qi, (qc, qf) in enumerate(zip(abnormal_pop, norm_fitness_a)):
                        bd = tuple(float(_qm_a[qi].get(n, 0.0)) for n in _bd_names)
                        qd_archive_abnormal.try_add(
                            prompt=qc, quality=qf, descriptor=bd,
                            metadata={"role": "abnormal", "name": name, "gen": gen},
                        )

                # 3. Select elites (QD archive sampling or MMR)
                if qd_archive_normal is not None and qd_archive_normal.size >= self.topk:
                    normal_elite = qd_archive_normal.sample_parents(k=self.topk)
                elif hasattr(self, "text_feat_cache") and len(self.text_feat_cache) > 0:
                    normal_elite = self._mmr_select(
                        normal_pop, norm_fitness_n, k=self.topk, role="normal"
                    )
                else:
                    rn = sorted(zip(normal_pop, norm_fitness_n), key=lambda x: x[1], reverse=True)
                    normal_elite = [c for c, _ in rn[: self.topk]]

                if qd_archive_abnormal is not None and qd_archive_abnormal.size >= self.topk:
                    abnormal_elite = qd_archive_abnormal.sample_parents(k=self.topk)
                elif hasattr(self, "text_feat_cache") and len(self.text_feat_cache) > 0:
                    abnormal_elite = self._mmr_select(
                        abnormal_pop, norm_fitness_a, k=self.topk, role="abnormal"
                    )
                else:
                    ra = sorted(zip(abnormal_pop, norm_fitness_a), key=lambda x: x[1], reverse=True)
                    abnormal_elite = [c for c, _ in ra[: self.topk]]

                self._record_population_snapshot(
                    category=name,
                    generation=gen,
                    stage="evolution",
                    normal_pop=normal_pop,
                    abnormal_pop=abnormal_pop,
                    scores_n=scores_n,
                    scores_a=scores_a,
                    norm_fitness_n=norm_fitness_n,
                    norm_fitness_a=norm_fitness_a,
                    normal_elite=normal_elite,
                    abnormal_elite=abnormal_elite,
                    pair_records=pair_records,
                )

                # Game-theory metrics (per-generation)
                if self.game_metrics_enable:
                    try:
                        gm = self._compute_game_metrics(
                            gen=gen, category=name,
                            normal_pop=normal_pop, abnormal_pop=abnormal_pop,
                            scores_n=scores_n, scores_a=scores_a,
                            norm_fitness_n=norm_fitness_n, norm_fitness_a=norm_fitness_a,
                            decomp_n=decomp_n, decomp_a=decomp_a,
                            normal_elite=normal_elite, abnormal_elite=abnormal_elite,
                            prev_elite_hash_n=_prev_elite_hash_n,
                            prev_elite_hash_a=_prev_elite_hash_a,
                            prev_payoff_n=_prev_payoff_n,
                            prev_payoff_a=_prev_payoff_a,
                            pop_size_n=len(normal_pop),
                            pop_size_a=len(abnormal_pop),
                        )
                        _prev_elite_hash_n = gm["elite_hash_n"]
                        _prev_elite_hash_a = gm["elite_hash_a"]
                        _prev_payoff_n = gm["payoff_n_mean"]
                        _prev_payoff_a = gm["payoff_a_mean"]
                    except Exception as exc:
                        # Surface instrumentation failures at INFO/WARNING level;
                        # debug logs are usually hidden in remote runs.
                        logger.warning("GAME_METRICS failed: %s", exc)

                if gen == 0 or gen == self.generations - 1:
                    bn_idx = max(range(len(normal_pop)), key=lambda i: norm_fitness_n[i])
                    ba_idx = max(range(len(abnormal_pop)), key=lambda i: norm_fitness_a[i])
                    print(
                        f"[CoEvo] generation {gen+1}/{self.generations}, category '{name}'"
                    )
                    if self.enable_consistency_guidance and consistencies_n is not None and consistencies_a is not None:
                        print(
                            f"  Normal Top1: AUROC={scores_n[bn_idx]:.4f}, "
                            f"Cons={consistencies_n[bn_idx]:.4f}, Fitness={norm_fitness_n[bn_idx]:.4f}"
                        )
                        print(
                            f"  Abnormal Top1: AUROC={scores_a[ba_idx]:.4f}, "
                            f"Cons={consistencies_a[ba_idx]:.4f}, Fitness={norm_fitness_a[ba_idx]:.4f}"
                        )
                    else:
                        print(f"  Normal Top1: AUROC={scores_n[bn_idx]:.4f}, Fitness={norm_fitness_n[bn_idx]:.4f}")
                        print(f"  Abnormal Top1: AUROC={scores_a[ba_idx]:.4f}, Fitness={norm_fitness_a[ba_idx]:.4f}")

                # 4. Mutation
                en = normal_elite if normal_elite else normal_pop[:1]
                ea = abnormal_elite if abnormal_elite else abnormal_pop[:1]
                op_counts_n = {"mutation": 0, "crossover": 0}
                op_counts_a = {"mutation": 0, "crossover": 0}

                normal_new = [
                    self._spawn_offspring(en, name, "normal", op_counts_n)
                    for _ in range(self.population_size - len(normal_elite))
                ]
                abnormal_new = [
                    self._spawn_offspring(ea, name, "abnormal", op_counts_a)
                    for _ in range(self.population_size - len(abnormal_elite))
                ]

                normal_pop = normal_elite + normal_new
                abnormal_pop = abnormal_elite + abnormal_new

                # Refill after deduplication to avoid population collapse
                normal_pop = list(dict.fromkeys(normal_pop))
                for _ in range(self.population_size * 3):
                    if len(normal_pop) >= self.population_size:
                        break
                    normal_pop.append(self._spawn_offspring(en, name, "normal", op_counts_n))
                    normal_pop = list(dict.fromkeys(normal_pop))
                if len(normal_pop) < self.population_size:
                    existing = set(normal_pop)
                    for c in self._init_population(name, role="normal"):
                        if c not in existing:
                            normal_pop.append(c)
                            existing.add(c)
                        if len(normal_pop) >= self.population_size:
                            break
                if len(normal_pop) < self.population_size:
                    logger.warning("CoEvo normal pop backfill incomplete: %d/%d for '%s' gen %d",
                                   len(normal_pop), self.population_size, name, gen)
                normal_pop = normal_pop[:self.population_size]

                abnormal_pop = list(dict.fromkeys(abnormal_pop))
                for _ in range(self.population_size * 3):
                    if len(abnormal_pop) >= self.population_size:
                        break
                    abnormal_pop.append(self._spawn_offspring(ea, name, "abnormal", op_counts_a))
                    abnormal_pop = list(dict.fromkeys(abnormal_pop))
                if len(abnormal_pop) < self.population_size:
                    existing = set(abnormal_pop)
                    for c in self._init_population(name, role="abnormal"):
                        if c not in existing:
                            abnormal_pop.append(c)
                            existing.add(c)
                        if len(abnormal_pop) >= self.population_size:
                            break
                if len(abnormal_pop) < self.population_size:
                    logger.warning("CoEvo abnormal pop backfill incomplete: %d/%d for '%s' gen %d",
                                   len(abnormal_pop), self.population_size, name, gen)
                abnormal_pop = abnormal_pop[:self.population_size]

                logger.info(
                    "COEVO_OP_COUNTS|gen=%d|cat=%s|role=normal|mutation=%d|crossover=%d|rate=%.3f",
                    gen,
                    name,
                    op_counts_n.get("mutation", 0),
                    op_counts_n.get("crossover", 0),
                    self.coevo_crossover_rate,
                )
                logger.info(
                    "COEVO_OP_COUNTS|gen=%d|cat=%s|role=abnormal|mutation=%d|crossover=%d|rate=%.3f",
                    gen,
                    name,
                    op_counts_a.get("mutation", 0),
                    op_counts_a.get("crossover", 0),
                    self.coevo_crossover_rate,
                )

            # 5. Final selection (score once more to refresh the cache)
            _cb_final_n = scoring_callback(normal_pop, role="normal")
            final_scores_n, final_cons_n = self._split_scores_result(
                _cb_final_n, normal_pop, "normal-final"
            )
            _qm_fn = _cb_final_n.get("src_metrics", []) if isinstance(_cb_final_n, dict) else []

            _cb_final_a = scoring_callback(abnormal_pop, role="abnormal")
            final_scores_a, final_cons_a = self._split_scores_result(
                _cb_final_a, abnormal_pop, "abnormal-final"
            )
            _qm_fa = _cb_final_a.get("src_metrics", []) if isinstance(_cb_final_a, dict) else []

            final_pair_records: List[Dict[str, Any]] = []
            agg_n, agg_a, _, _ = self._aggregate_pair_scores(
                normal_pop, abnormal_pop, final_scores_n, final_scores_a,
                consistencies_n=final_cons_n, consistencies_a=final_cons_a,
                pair_records_out=final_pair_records if self.record_population_trace else None,
            )
            norm_fitness_n = [agg_n.get(p, final_scores_n[i]) for i, p in enumerate(normal_pop)]
            norm_fitness_a = [agg_a.get(p, final_scores_a[i]) for i, p in enumerate(abnormal_pop)]

            if qd_archive_normal is not None and len(_qm_fn) == len(normal_pop):
                for qi, (qc, qf) in enumerate(zip(normal_pop, norm_fitness_n)):
                    bd = tuple(float(_qm_fn[qi].get(n, 0.0)) for n in _bd_names)
                    qd_archive_normal.try_add(
                        prompt=qc, quality=qf, descriptor=bd,
                        metadata={"role": "normal", "name": name, "gen": "final"},
                    )

            if qd_archive_abnormal is not None and len(_qm_fa) == len(abnormal_pop):
                for qi, (qc, qf) in enumerate(zip(abnormal_pop, norm_fitness_a)):
                    bd = tuple(float(_qm_fa[qi].get(n, 0.0)) for n in _bd_names)
                    qd_archive_abnormal.try_add(
                        prompt=qc, quality=qf, descriptor=bd,
                        metadata={"role": "abnormal", "name": name, "gen": "final"},
                    )

            best_n_idx = max(range(len(normal_pop)), key=lambda i: norm_fitness_n[i])
            best_a_idx = max(range(len(abnormal_pop)), key=lambda i: norm_fitness_a[i])
            best_n = normal_pop[best_n_idx]
            best_a = abnormal_pop[best_a_idx]
            marginal_best_n = best_n
            marginal_best_a = best_a
            final_elite_n = [
                normal_pop[i] for i in sorted(
                    range(len(normal_pop)),
                    key=lambda j: norm_fitness_n[j],
                    reverse=True,
                )[: self.topk]
            ]
            final_elite_a = [
                abnormal_pop[i] for i in sorted(
                    range(len(abnormal_pop)),
                    key=lambda j: norm_fitness_a[j],
                    reverse=True,
                )[: self.topk]
            ]
            self._record_population_snapshot(
                category=name,
                generation="final",
                stage="final",
                normal_pop=normal_pop,
                abnormal_pop=abnormal_pop,
                scores_n=final_scores_n,
                scores_a=final_scores_a,
                norm_fitness_n=norm_fitness_n,
                norm_fitness_a=norm_fitness_a,
                normal_elite=final_elite_n,
                abnormal_elite=final_elite_a,
                pair_records=final_pair_records,
            )

            global_pair_record: Optional[Dict[str, Any]] = None
            rerank_handling = "none"

            if rerank_callback is not None and rerank_topk > 1:
                top_n_idx = sorted(
                    range(len(normal_pop)),
                    key=lambda i: norm_fitness_n[i],
                    reverse=True,
                )[: min(rerank_topk, len(normal_pop))]
                top_a_idx = sorted(
                    range(len(abnormal_pop)),
                    key=lambda i: norm_fitness_a[i],
                    reverse=True,
                )[: min(rerank_topk, len(abnormal_pop))]

                rerank_normal = [normal_pop[i] for i in top_n_idx]
                rerank_abnormal = [abnormal_pop[i] for i in top_a_idx]
                rerank_scores_n = self._extract_score_list(
                    rerank_callback(rerank_normal, role="normal"),
                    expected_len=len(rerank_normal),
                )
                rerank_scores_a = self._extract_score_list(
                    rerank_callback(rerank_abnormal, role="abnormal"),
                    expected_len=len(rerank_abnormal),
                )
                if self.coevo_final_pair_select == "global_argmax":
                    global_pair_record = self._select_global_pair_argmax(
                        normal_pop=rerank_normal,
                        abnormal_pop=rerank_abnormal,
                        scores_n=rerank_scores_n,
                        scores_a=rerank_scores_a,
                    )
                    if global_pair_record is not None:
                        best_n = global_pair_record["normal_prompt"]
                        best_a = global_pair_record["abnormal_prompt"]
                    rerank_handling = "global_argmax_on_rerank_cross_product"
                else:
                    best_n = rerank_normal[max(range(len(rerank_normal)), key=lambda i: rerank_scores_n[i])]
                    best_a = rerank_abnormal[max(range(len(rerank_abnormal)), key=lambda i: rerank_scores_a[i])]
                    rerank_handling = "split_rerank"
            elif self.coevo_final_pair_select == "global_argmax":
                global_pair_record = self._select_global_pair_argmax(
                    normal_pop=normal_pop,
                    abnormal_pop=abnormal_pop,
                    scores_n=final_scores_n,
                    scores_a=final_scores_a,
                    consistencies_n=final_cons_n,
                    consistencies_a=final_cons_a,
                )
                if global_pair_record is not None:
                    best_n = global_pair_record["normal_prompt"]
                    best_a = global_pair_record["abnormal_prompt"]
                rerank_handling = "global_argmax_on_final_population"

            if self.coevo_final_pair_select == "global_argmax":
                audit = {
                    "category": name,
                    "selector_mode": self.coevo_final_pair_select,
                    "rerank_handling": rerank_handling,
                    "marginal_normal_prompt": marginal_best_n,
                    "marginal_abnormal_prompt": marginal_best_a,
                    "global_normal_prompt": best_n,
                    "global_abnormal_prompt": best_a,
                    "selection_differs": bool(
                        best_n != marginal_best_n or best_a != marginal_best_a
                    ),
                }
                if global_pair_record is not None:
                    audit.update(
                        {
                            "selected_pair_score": float(global_pair_record["pair_score"]),
                            "pair_matrix_size": int(global_pair_record["pair_matrix_size"]),
                            "selected_pair_terms": dict(global_pair_record["pair_terms"]),
                        }
                    )
                self.final_pair_selection_audit.append(audit)
                logger.info(
                    "COEVO_FINAL_PAIR_SELECT|cat=%s|mode=%s|rerank=%s|differs=%s|pair_score=%s|normal=%s|abnormal=%s",
                    name,
                    self.coevo_final_pair_select,
                    rerank_handling,
                    audit["selection_differs"],
                    audit.get("selected_pair_score"),
                    best_n,
                    best_a,
                )

            # QD: log archive summary (archive used for parent selection only;
            # final best respects rerank pipeline and coevo hard constraints)
            if qd_archive_normal is not None and qd_archive_normal.size > 0:
                logger.info(
                    "QD archive normal for '%s': %d/%d cells (%.0f%%)",
                    name, qd_archive_normal.size, qd_archive_normal.max_cells,
                    qd_archive_normal.coverage * 100,
                )
            if qd_archive_abnormal is not None and qd_archive_abnormal.size > 0:
                logger.info(
                    "QD archive abnormal for '%s': %d/%d cells (%.0f%%)",
                    name, qd_archive_abnormal.size, qd_archive_abnormal.max_cells,
                    qd_archive_abnormal.coverage * 100,
                )

            # Hard constraint: the final normal and abnormal selections must differ
            if best_n.strip() == best_a.strip():
                best_a = f"X abnormal {name}"
                logger.warning(
                    f"[CoEvo] normal==abnormal for '{name}', "
                    f"falling back to default abnormal prompt: '{best_a}'"
                )

            self.cache[cache_key_n] = best_n
            self.cache[cache_key_a] = best_a
            normal_optimized.append(best_n)
            abnormal_optimized.append(best_a)

        if self.llm_mutation_enabled and any(v > 0 for v in self._llm_stats.values()):
            logger.info(
                "LLM_MUTATE_STATS|rephrase=%d|creative=%d|fallback=%d|rule=%d",
                self._llm_stats["llm_rephrase"],
                self._llm_stats["llm_creative"],
                self._llm_stats["llm_fallback"],
                self._llm_stats["rule_ops"],
            )

        return normal_optimized, abnormal_optimized

    def save_optimized_rules(self, save_path: str) -> None:
        super().save_optimized_rules(save_path)
        try:
            with open(save_path, "r", encoding="utf-8") as f:
                rules = json.load(f)
            rules["coevo_final_pair_select"] = self.coevo_final_pair_select
            if self.final_pair_selection_audit:
                rules["final_pair_selection_audit"] = self.final_pair_selection_audit
            if self.record_population_trace:
                rules["population_trace_schema"] = "coevo_population_trace_v1"
                rules["population_trace"] = self.population_trace
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(rules, f, indent=2, ensure_ascii=False)

            if self.record_population_trace:
                trace_path = os.path.splitext(save_path)[0] + ".population_trace.jsonl"
                with open(trace_path, "w", encoding="utf-8") as f:
                    for entry in self.population_trace:
                        f.write(json.dumps(entry, sort_keys=True, ensure_ascii=False) + "\n")
                logger.info(
                    "CoEvo population trace saved: %s entries=%d",
                    trace_path,
                    len(self.population_trace),
                )
        except Exception as exc:
            logger.warning("CoEvo selector/trace save skipped: %s", exc)


def build_coevo_optimizer(**kwargs) -> CoEvoPromptOptimizer:
    """Build the co-evolutionary optimizer."""
    game_flag = kwargs.pop("game_metrics_enable", False)
    llm_enabled = kwargs.pop("llm_mutation_enabled", False)
    llm_model_id = kwargs.pop("llm_model_id", "")
    llm_max_tokens = kwargs.pop("llm_mutation_max_tokens", 32)
    return CoEvoPromptOptimizer(
        game_metrics_enable=game_flag,
        llm_mutation_enabled=llm_enabled,
        llm_model_id=llm_model_id,
        llm_mutation_max_tokens=llm_max_tokens,
        **kwargs,
    )
