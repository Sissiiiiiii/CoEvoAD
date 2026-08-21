from __future__ import annotations

import abc
import importlib
import inspect
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from models.evaluate import (
    aggregate_prompt_anomaly_map,
    aggregate_prompt_image_score,
    prompt_slot_anomaly_maps,
    prompt_slot_image_scores,
)


def _reduce_layer_maps_with_config(
    layer_maps: List[torch.Tensor],
    pixel_layer_weights: Optional[List[float]] = None,
    enable_layer_taa: bool = False,
    layer_taa_tau: float = 1.0,
) -> Optional[torch.Tensor]:
    if not layer_maps:
        return None

    if enable_layer_taa:
        tau = max(float(layer_taa_tau), 1e-6)
        per_layer_max = torch.stack(
            [layer_map.reshape(layer_map.shape[0], -1).max(dim=1).values for layer_map in layer_maps],
            dim=1,
        )
        weights = torch.softmax(per_layer_max / tau, dim=1)
        reduced = None
        for idx, layer_map in enumerate(layer_maps):
            view_shape = (layer_map.shape[0],) + (1,) * (layer_map.ndim - 1)
            weighted = weights[:, idx].view(view_shape) * layer_map
            reduced = weighted if reduced is None else reduced + weighted
        return reduced

    if pixel_layer_weights is not None and len(pixel_layer_weights) == len(layer_maps):
        weights = torch.tensor(
            pixel_layer_weights,
            dtype=layer_maps[0].dtype,
            device=layer_maps[0].device,
        ).clamp(min=0.0)
        if weights.sum() <= 0:
            weights = torch.ones(len(layer_maps), dtype=layer_maps[0].dtype, device=layer_maps[0].device)
        weights = weights / weights.sum()
        return sum(weights[i] * layer_maps[i] for i in range(len(layer_maps)))

    return torch.stack(layer_maps, dim=0).mean(dim=0)


def _validate_scoring_head_checkpoint(args, state_dict: Dict[str, torch.Tensor]) -> None:
    if not bool(getattr(args, "use_scoring_head", False)):
        return
    if any(k.startswith("scoring_head.") for k in state_dict.keys()):
        return
    raise ValueError(
        "--use_scoring_head requires checkpoint weights with scoring_head.* keys; "
        "refusing to evaluate a randomly initialized scoring head"
    )


class AnomalyScorer(abc.ABC):
    @abc.abstractmethod
    def score_candidate(self, prepared: Any, candidate: str, role: str, baseline: str) -> torch.Tensor:
        ...

    @abc.abstractmethod
    def infer(self, images: torch.Tensor, normal_prompt: str, abnormal_prompt: str, stage: int = 2) -> Dict[str, Any]:
        ...

    def prepare_images(self, images: torch.Tensor) -> Any:
        return images

    def get_text_embedding(self, prompt: str, role: str) -> Optional[torch.Tensor]:
        return None

    def evaluate_candidate(self, prepared: Any, candidate: str, role: str, baseline: str) -> Dict[str, Any]:
        image_scores = self.score_candidate(prepared, candidate, role, baseline)
        return {"image_scores": image_scores, "pixel_maps": None}

    def evaluate_candidates(
        self,
        prepared: Any,
        candidates: List[str],
        role: str,
        baseline: str,
    ) -> List[Dict[str, Any]]:
        return [
            self.evaluate_candidate(prepared=prepared, candidate=c, role=role, baseline=baseline)
            for c in candidates
        ]

    def evaluate_prompt_pair(
        self,
        prepared: Any,
        normal_prompt: str,
        abnormal_prompt: str,
        stage: int = 2,
    ) -> Dict[str, Any]:
        raise NotImplementedError("evaluate_prompt_pair is not implemented for this scorer")

    def prepare_prompt_pair(self, normal_prompt: str, abnormal_prompt: str, stage: int = 2) -> Any:
        return None

    def infer_prepared(self, images: torch.Tensor, prepared_prompt: Any, stage: int = 2) -> Dict[str, Any]:
        raise NotImplementedError("infer_prepared is not implemented for this scorer")


class PromptBankScorer(AnomalyScorer):
    def __init__(self, model, model_clip, text_encoder, tokenizer, args, device):
        self.model = model
        self.model_clip = model_clip
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.args = args
        self.device = device
        self.features_list = args.features_list
        self.pixel_layer_weights = getattr(args, "pixel_layer_weights", None)
        self.enable_layer_taa = bool(getattr(args, "enable_layer_taa", False))
        self.layer_taa_tau = float(getattr(args, "layer_taa_tau", 1.0))
        self.zero_shot_scoring = bool(getattr(args, "zero_shot_scoring", False))
        self.has_pfl_modules = all(
            hasattr(model, name) for name in ("PFL_context", "PFL_normal", "PFL_abnormal")
        )
        self.text_encoder_accepts_prompt_bias = self._detect_prompt_bias_support(text_encoder)
        self._baseline_emb_cache: Dict[str, torch.Tensor] = {}

        if self.zero_shot_scoring and not self.text_encoder_accepts_prompt_bias:
            import logging
            logging.getLogger(__name__).warning(
                "zero_shot_scoring=True but TextEncoder does not accept prompt_bias; "
                "flag has no effect in deterministic model"
            )

    def _resolve_stage(self) -> int:
        """Infer the inference stage from the checkpoint path.

        - stage1_final.pth → 1  (fuse/image_mapping not trained)
        - two_stage_final.pth → 2
        - epoch_post_*_1.pth → 1, epoch_post_*_2.pth → 2
        - otherwise -> 2 (default)
        """
        ckpt = getattr(self.args, "checkpoint_path", "")
        if not ckpt:
            return 2
        basename = os.path.basename(ckpt)
        if "stage1_final" in basename:
            return 1
        if "epoch_post" in basename:
            try:
                parsed = int(basename[:-4].split("_")[-1])
                if parsed in (1, 2):
                    return parsed
            except (ValueError, IndexError):
                pass
        return 2

    @staticmethod
    def _detect_prompt_bias_support(text_encoder) -> bool:
        try:
            sig = inspect.signature(text_encoder.forward)
            if "prompt_bias_list" in sig.parameters:
                return True
        except (TypeError, ValueError):
            pass

        forward = getattr(text_encoder, "forward", None)
        code = getattr(forward, "__code__", None)
        if code is not None:
            return code.co_argcount >= 5
        return False

    def _encode_text_branch(
        self,
        tokens,
        visual_feature: torch.Tensor,
        prompt_bias: Optional[torch.Tensor],
        prompt_bias_state: Optional[torch.Tensor],
        mode: str,
    ) -> torch.Tensor:
        if self.text_encoder_accepts_prompt_bias:
            if prompt_bias is None or prompt_bias_state is None:
                raise ValueError("Legacy TextEncoder requires prompt_bias and prompt_bias_state")
            return self.text_encoder(
                tokens,
                visual_feature,
                [prompt_bias, prompt_bias_state],
                mode=mode,
            )
        return self.text_encoder(tokens, visual_feature)

    def _compute_stage2_latents(
        self,
        image_features: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], str]:
        if not self.text_encoder_accepts_prompt_bias:
            return None, None, None, "test"

        if image_features is None:
            raise ValueError("Legacy TextEncoder compatibility requires image_features")

        if self.zero_shot_scoring or not self.has_pfl_modules:
            zeros = torch.zeros_like(image_features)
            return zeros, zeros, zeros, "zero_shot"

        # Use the train-style path here so legacy TextEncoder keeps the deterministic [B, prompt_num, D] shape.
        _, _, _, _, _, zk = self.model.PFL_context(image_features.clone(), mode="train")
        _, _, _, _, _, zk_n = self.model.PFL_normal(image_features.clone(), mode="train")
        _, _, _, _, _, zk_a = self.model.PFL_abnormal(image_features.clone(), mode="train")
        return zk, zk_n, zk_a, "train"

    def _reduce_layer_maps(self, layer_maps: List[torch.Tensor]) -> Optional[torch.Tensor]:
        return _reduce_layer_maps_with_config(
            layer_maps,
            pixel_layer_weights=self.pixel_layer_weights,
            enable_layer_taa=self.enable_layer_taa,
            layer_taa_tau=self.layer_taa_tau,
        )

    def prepare_images(self, images: torch.Tensor) -> Tuple[torch.Tensor, list]:
        images = images.to(self.device, non_blocking=True)
        with torch.no_grad():
            image_features, _, patch_tokens = self.model_clip.encode_image(images, self.features_list)
        return image_features, patch_tokens

    def _tokenize(self, prompts: List[str]):
        tokens = self.tokenizer(prompts)
        if hasattr(tokens, "to"):
            return tokens.to(self.device)
        if isinstance(tokens, dict):
            return {k: v.to(self.device) for k, v in tokens.items()}
        return torch.as_tensor(tokens, device=self.device)

    def _build_role_prompts(
        self,
        candidate: str,
        baseline: str,
        role: str,
        batch_size: int,
    ) -> Tuple[List[str], List[str]]:
        if role == "normal":
            return [candidate] * batch_size, [baseline] * batch_size
        if role == "abnormal":
            return [baseline] * batch_size, [candidate] * batch_size
        return [candidate] * batch_size, [candidate] * batch_size

    def clear_baseline_cache(self):
        """Clear the baseline embedding cache (called when switching category)."""
        self._baseline_emb_cache.clear()

    def _encode_fixed_clip_prompts(
        self,
        normal_prompts: List[str],
        abnormal_prompts: List[str],
    ) -> torch.Tensor:
        """Ablation: use fixed CLIP text encoding, bypassing learnable prompt bank."""
        with torch.no_grad():
            n_tokens = self._tokenize(normal_prompts)
            a_tokens = self._tokenize(abnormal_prompts)
            n_emb = self.model_clip.encode_text(n_tokens)
            a_emb = self.model_clip.encode_text(a_tokens)
        K = self.args.prompt_num
        n_emb = n_emb.unsqueeze(1).expand(-1, K, -1).contiguous()
        a_emb = a_emb.unsqueeze(1).expand(-1, K, -1).contiguous()
        text_embeddings = torch.cat([n_emb, a_emb], dim=1)
        return F.normalize(text_embeddings, dim=-1, eps=1e-12)

    def _encode_prompt_pair(
        self,
        normal_prompts: List[str],
        abnormal_prompts: List[str],
        image_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if getattr(self.args, "ablate_prompt_bank", False):
            return self._encode_fixed_clip_prompts(normal_prompts, abnormal_prompts)

        visual_normal = torch.cat([self.model.prompt_context, self.model.prompt_state_normal], dim=1)
        visual_abnormal = torch.cat([self.model.prompt_context, self.model.prompt_state_abnormal], dim=1)

        normal_tokens = self._tokenize(normal_prompts)
        abnormal_tokens = self._tokenize(abnormal_prompts)
        prompt_bias, prompt_bias_normal, prompt_bias_abnormal, encode_mode = self._compute_stage2_latents(
            image_features
            if image_features is not None
            else torch.zeros(len(normal_prompts), self.args.embed_dim, device=self.device)
        )
        normal_embeddings = self._encode_text_branch(
            normal_tokens,
            visual_normal,
            prompt_bias,
            prompt_bias_normal,
            mode=encode_mode,
        )
        abnormal_embeddings = self._encode_text_branch(
            abnormal_tokens,
            visual_abnormal,
            prompt_bias,
            prompt_bias_abnormal,
            mode=encode_mode,
        )
        text_embeddings = torch.cat([normal_embeddings, abnormal_embeddings], dim=1)
        return F.normalize(text_embeddings, dim=-1, eps=1e-12)

    def _infer_with_prepared_features(
        self,
        image_features: torch.Tensor,
        patch_tokens: List[torch.Tensor],
        text_embeddings: torch.Tensor,
        stage: int,
    ) -> Dict[str, Any]:
        pro_img, anomaly_maps_list = self.model(
            text_embeddings,
            image_features,
            patch_tokens,
            stage=stage,
            mode="test",
        )
        pro_img = pro_img.squeeze(2)

        _pagg = getattr(self.args, "prompt_agg", "mean")
        _stemp = float(getattr(self.args, "scorer_temperature", 1.0))
        _score_mode = getattr(self.args, "prompt_score_mode", "softmax_mean")
        image_scores = aggregate_prompt_image_score(pro_img, self.args.prompt_num, prompt_agg=_pagg,
                                                    scorer_temperature=_stemp, prompt_score_mode=_score_mode)
        prompt_layer_maps = [
            aggregate_prompt_anomaly_map([anomaly_map], self.args.prompt_num, layer_reduce="mean", prompt_agg=_pagg,
                                        scorer_temperature=_stemp)
            for anomaly_map in anomaly_maps_list
        ]
        pixel_maps = self._reduce_layer_maps(prompt_layer_maps)

        out = {
            "image_scores": image_scores,
            "pixel_maps": pixel_maps,
            "image_logits": pro_img.detach(),
        }
        if bool(getattr(self.args, "return_slot_outputs", False)):
            out["slot_image_scores"] = prompt_slot_image_scores(
                pro_img,
                self.args.prompt_num,
                scorer_temperature=_stemp,
            ).detach()
            slot_layer_maps = [
                prompt_slot_anomaly_maps(
                    anomaly_map,
                    self.args.prompt_num,
                    scorer_temperature=_stemp,
                )
                for anomaly_map in anomaly_maps_list
            ]
            if slot_layer_maps:
                out["slot_pixel_maps"] = torch.stack(slot_layer_maps, dim=0).mean(dim=0).detach()
        return out

    def score_candidate(self, prepared, candidate: str, role: str, baseline: str) -> torch.Tensor:
        return self.evaluate_candidate(prepared, candidate, role, baseline)["image_scores"]

    def _get_cached_baseline_emb(self, prompt: str, role_label: str) -> Optional[torch.Tensor]:
        """Return the cached baseline embedding, or None on a cache miss."""
        cache_key = f"{role_label}:{prompt}"
        return self._baseline_emb_cache.get(cache_key)

    def _put_baseline_emb(self, prompt: str, role_label: str, emb: torch.Tensor):
        cache_key = f"{role_label}:{prompt}"
        self._baseline_emb_cache[cache_key] = emb.detach()

    def evaluate_candidate(
        self,
        prepared: Tuple[torch.Tensor, list],
        candidate: str,
        role: str,
        baseline: str,
    ) -> Dict[str, Any]:
        image_features, patch_tokens = prepared
        batch_size = image_features.shape[0]
        prompts_n, prompts_a = self._build_role_prompts(candidate, baseline, role, batch_size)
        with torch.no_grad():
            # Try to speed things up with the baseline embedding cache
            baseline_role = "abnormal" if role == "normal" else "normal"
            cached = self._get_cached_baseline_emb(baseline, baseline_role)
            if (cached is not None
                    and not self.text_encoder_accepts_prompt_bias):
                # Cache hit: move to the right device and encode only the candidate side
                if cached.device != image_features.device:
                    cached = cached.to(image_features.device, non_blocking=True)
                if role == "normal":
                    visual_cand = torch.cat([self.model.prompt_context, self.model.prompt_state_normal], dim=1)
                else:
                    visual_cand = torch.cat([self.model.prompt_context, self.model.prompt_state_abnormal], dim=1)
                cand_tokens = self._tokenize([candidate] * batch_size)
                cand_emb = self._encode_text_branch(cand_tokens, visual_cand, None, None, mode="test")
                baseline_emb = cached.expand(batch_size, -1, -1).contiguous()
                if role == "normal":
                    text_embeddings = torch.cat([cand_emb, baseline_emb], dim=1)
                else:
                    text_embeddings = torch.cat([baseline_emb, cand_emb], dim=1)
                text_embeddings = F.normalize(text_embeddings, dim=-1, eps=1e-12)
            else:
                # No cache, or a legacy model: encode everything
                text_embeddings = self._encode_prompt_pair(prompts_n, prompts_a, image_features=image_features)
                # Cache the baseline-side embedding
                if not self.text_encoder_accepts_prompt_bias:
                    half = text_embeddings.size(1) // 2
                    if role == "normal":
                        self._put_baseline_emb(baseline, "abnormal", text_embeddings[:1, half:])
                    else:
                        self._put_baseline_emb(baseline, "normal", text_embeddings[:1, :half])
            return self._infer_with_prepared_features(image_features, patch_tokens, text_embeddings, stage=self._resolve_stage())

    def get_text_embedding(self, prompt: str, role: str) -> Optional[torch.Tensor]:
        with torch.no_grad():
            tokens = self._tokenize([prompt])
            if role == "normal":
                visual = torch.cat([self.model.prompt_context, self.model.prompt_state_normal], dim=1)
            else:
                visual = torch.cat([self.model.prompt_context, self.model.prompt_state_abnormal], dim=1)
            emb = self._encode_text_branch(
                tokens,
                visual,
                torch.zeros(1, self.args.embed_dim, device=self.device) if self.text_encoder_accepts_prompt_bias else None,
                torch.zeros(1, self.args.embed_dim, device=self.device) if self.text_encoder_accepts_prompt_bias else None,
                mode="zero_shot",
            )
            return emb.mean(dim=1).squeeze(0).detach().float().cpu()

    def prepare_prompt_pair(self, normal_prompt: str, abnormal_prompt: str, stage: int = 2) -> Dict[str, Any]:
        del stage
        with torch.no_grad():
            text_embeddings = self._encode_prompt_pair([normal_prompt], [abnormal_prompt]).detach()
        return {
            "normal_prompt": normal_prompt,
            "abnormal_prompt": abnormal_prompt,
            "text_embeddings": text_embeddings,
        }

    def infer_prepared(self, images: torch.Tensor, prepared_prompt: Dict[str, Any], stage: int = 2) -> Dict[str, Any]:
        if prepared_prompt is None or "text_embeddings" not in prepared_prompt:
            raise ValueError("prepared_prompt must contain text_embeddings")
        image_features, patch_tokens = self.prepare_images(images)
        text_embeddings = prepared_prompt["text_embeddings"].to(self.device)
        if text_embeddings.ndim == 2:
            text_embeddings = text_embeddings.unsqueeze(0)
        if text_embeddings.size(0) == 1 and image_features.size(0) > 1:
            text_embeddings = text_embeddings.expand(image_features.size(0), -1, -1).contiguous()
        return self._infer_with_prepared_features(image_features, patch_tokens, text_embeddings, stage=stage)

    def evaluate_prompt_pair(
        self,
        prepared: Tuple[torch.Tensor, list],
        normal_prompt: str,
        abnormal_prompt: str,
        stage: int = 2,
    ) -> Dict[str, Any]:
        image_features, patch_tokens = prepared
        batch_size = image_features.shape[0]
        with torch.no_grad():
            text_embeddings = self._encode_prompt_pair(
                [normal_prompt] * batch_size,
                [abnormal_prompt] * batch_size,
                image_features=image_features,
            )
        return self._infer_with_prepared_features(image_features, patch_tokens, text_embeddings, stage=stage)

    def infer(
        self,
        images: torch.Tensor,
        normal_prompt: str,
        abnormal_prompt: str,
        stage: int = 2,
    ) -> Dict[str, Any]:
        image_features, patch_tokens = self.prepare_images(images)
        batch_size = image_features.shape[0]
        with torch.no_grad():
            text_embeddings = self._encode_prompt_pair(
                [normal_prompt] * batch_size,
                [abnormal_prompt] * batch_size,
                image_features=image_features,
            )
            return self._infer_with_prepared_features(image_features, patch_tokens, text_embeddings, stage=stage)


class CLIPTransferScorer(AnomalyScorer):
    def __init__(self, model_clip, tokenizer, args, device):
        self.model_clip = model_clip
        self.tokenizer = tokenizer
        self.args = args
        self.device = device
        self.features_list = getattr(args, "features_list", [6, 12, 18, 24])
        self.image_size = int(getattr(args, "image_size", 518))
        self.pixel_layer_weights = getattr(args, "pixel_layer_weights", None)
        self.enable_layer_taa = bool(getattr(args, "enable_layer_taa", False))
        self.layer_taa_tau = float(getattr(args, "layer_taa_tau", 1.0))

    def _reduce_layer_maps(self, layer_maps: List[torch.Tensor]) -> Optional[torch.Tensor]:
        return _reduce_layer_maps_with_config(
            layer_maps,
            pixel_layer_weights=self.pixel_layer_weights,
            enable_layer_taa=self.enable_layer_taa,
            layer_taa_tau=self.layer_taa_tau,
        )

    def _tokenize(self, prompts: List[str]):
        tokens = self.tokenizer(prompts)
        if hasattr(tokens, "to"):
            return tokens.to(self.device)
        if isinstance(tokens, dict):
            return {k: v.to(self.device) for k, v in tokens.items()}
        return torch.as_tensor(tokens, device=self.device)

    def _encode_text(self, prompts: List[str]) -> torch.Tensor:
        with torch.no_grad():
            text_feat = self.model_clip.encode_text(self._tokenize(prompts))
        return F.normalize(text_feat.float(), dim=-1, eps=1e-12)

    def prepare_images(self, images: torch.Tensor) -> Tuple[torch.Tensor, list]:
        images = images.to(self.device, non_blocking=True)
        with torch.no_grad():
            image_features, _, patch_tokens = self.model_clip.encode_image(images, self.features_list)
            image_features = F.normalize(image_features.float(), dim=-1, eps=1e-12)
        return image_features, patch_tokens

    def _build_role_prompts(
        self,
        candidate: str,
        baseline: str,
        role: str,
        batch_size: int,
    ) -> Tuple[List[str], List[str]]:
        if role == "normal":
            return [candidate] * batch_size, [baseline] * batch_size
        if role == "abnormal":
            return [baseline] * batch_size, [candidate] * batch_size
        return [candidate] * batch_size, [candidate] * batch_size

    def score_candidate(self, prepared, candidate: str, role: str, baseline: str) -> torch.Tensor:
        return self.evaluate_candidate(prepared, candidate, role, baseline)["image_scores"]

    def evaluate_candidate(
        self,
        prepared: Tuple[torch.Tensor, list],
        candidate: str,
        role: str,
        baseline: str,
    ) -> Dict[str, Any]:
        image_features, patch_tokens = prepared
        batch_size = image_features.shape[0]
        prompts_n, prompts_a = self._build_role_prompts(candidate, baseline, role, batch_size)
        text_n = self._encode_text(prompts_n)
        text_a = self._encode_text(prompts_a)

        sim_n = (image_features * text_n).sum(dim=-1)
        sim_a = (image_features * text_a).sum(dim=-1)
        image_scores = torch.softmax(torch.stack([sim_n, sim_a], dim=1), dim=1)[:, 1]

        pixel_maps = None
        token_layers = patch_tokens if isinstance(patch_tokens, list) else [patch_tokens]
        layer_maps = []
        text_delta = F.normalize((text_a - text_n).float(), dim=-1, eps=1e-12)
        for token_tensor in token_layers:
            if token_tensor is None or token_tensor.ndim != 3:
                continue
            token_tensor = F.normalize(token_tensor.float(), dim=-1, eps=1e-12)
            token_scores = torch.einsum("bld,bd->bl", token_tensor, text_delta)
            side = int(np.sqrt(token_scores.shape[1]))
            usable = side * side
            if usable <= 0:
                continue
            token_maps = token_scores[:, :usable].view(batch_size, 1, side, side)
            token_maps = F.interpolate(token_maps, size=self.image_size, mode="bilinear", align_corners=False)
            layer_maps.append(torch.sigmoid(token_maps).squeeze(1))
        if layer_maps:
            pixel_maps = self._reduce_layer_maps(layer_maps)

        return {"image_scores": image_scores, "pixel_maps": pixel_maps}

    def get_text_embedding(self, prompt: str, role: str) -> Optional[torch.Tensor]:
        del role
        return self._encode_text([prompt]).squeeze(0).cpu()

    def prepare_prompt_pair(self, normal_prompt: str, abnormal_prompt: str, stage: int = 2) -> Dict[str, Any]:
        del stage
        return {"normal_prompt": normal_prompt, "abnormal_prompt": abnormal_prompt}

    def infer_prepared(self, images: torch.Tensor, prepared_prompt: Dict[str, Any], stage: int = 2) -> Dict[str, Any]:
        return self.infer(images, prepared_prompt["normal_prompt"], prepared_prompt["abnormal_prompt"], stage=stage)

    def evaluate_prompt_pair(
        self,
        prepared: Tuple[torch.Tensor, list],
        normal_prompt: str,
        abnormal_prompt: str,
        stage: int = 2,
    ) -> Dict[str, Any]:
        del stage
        image_features, patch_tokens = prepared
        batch_size = image_features.shape[0]
        text_n = self._encode_text([normal_prompt] * batch_size)
        text_a = self._encode_text([abnormal_prompt] * batch_size)

        sim_n = (image_features * text_n).sum(dim=-1)
        sim_a = (image_features * text_a).sum(dim=-1)
        image_scores = torch.softmax(torch.stack([sim_n, sim_a], dim=1), dim=1)[:, 1]

        pixel_maps = None
        token_layers = patch_tokens if isinstance(patch_tokens, list) else [patch_tokens]
        layer_maps = []
        text_delta = F.normalize((text_a - text_n).float(), dim=-1, eps=1e-12)
        for token_tensor in token_layers:
            if token_tensor is None or token_tensor.ndim != 3:
                continue
            token_tensor = F.normalize(token_tensor.float(), dim=-1, eps=1e-12)
            token_scores = torch.einsum("bld,bd->bl", token_tensor, text_delta)
            side = int(np.sqrt(token_scores.shape[1]))
            usable = side * side
            if usable <= 0:
                continue
            token_maps = token_scores[:, :usable].view(batch_size, 1, side, side)
            token_maps = F.interpolate(token_maps, size=self.image_size, mode="bilinear", align_corners=False)
            layer_maps.append(torch.sigmoid(token_maps).squeeze(1))
        if layer_maps:
            pixel_maps = self._reduce_layer_maps(layer_maps)

        return {"image_scores": image_scores, "pixel_maps": pixel_maps}

    def infer(
        self,
        images: torch.Tensor,
        normal_prompt: str,
        abnormal_prompt: str,
        stage: int = 2,
    ) -> Dict[str, Any]:
        del stage
        prepared = self.prepare_images(images)
        return self.evaluate_candidate(prepared, normal_prompt, "normal", abnormal_prompt)


def build_scorer(args, device: torch.device) -> AnomalyScorer:
    scorer_type = getattr(args, "scorer_type", "prompt_bank")
    if scorer_type == "external":
        return _build_external_scorer(args, device)
    if scorer_type == "clip_transfer":
        return _build_clip_transfer_scorer(args, device)
    if scorer_type == "prompt_bank":
        return _build_prompt_bank_scorer(args, device)
    if scorer_type == "custom":
        return _build_custom_scorer(args, device)
    raise ValueError(f"Unknown scorer_type: {scorer_type}")


def _load_model_config(args):
    with open(args.config_path, "r") as f:
        model_configs = json.load(f)
    args.vision_width = model_configs["vision_cfg"]["width"]
    args.text_width = model_configs["text_cfg"]["width"]
    args.embed_dim = model_configs["embed_dim"]


def _build_clip_transfer_scorer(args, device: torch.device) -> CLIPTransferScorer:
    from models.model_CLIP import Load_CLIP, tokenize

    model_clip, _, _ = Load_CLIP(args.image_size, args.pretrained_path, device=device)
    model_clip.to(device)
    model_clip.eval()
    for param in model_clip.parameters():
        param.requires_grad = False
    return CLIPTransferScorer(model_clip=model_clip, tokenizer=tokenize, args=args, device=device)


def _build_prompt_bank_scorer(args, device: torch.device) -> PromptBankScorer:
    from models.model_CLIP import Load_CLIP, tokenize
    from models.VPB import Context_Prompting, TextEncoder

    _load_model_config(args)
    model_clip, _, _ = Load_CLIP(args.image_size, args.pretrained_path, device=device)
    model_clip.to(device)
    model_clip.eval()

    text_encoder = TextEncoder(model_clip, args)

    checkpoint_path = getattr(args, "checkpoint_path", "")
    if not checkpoint_path:
        raise ValueError("prompt_bank scorer requires --checkpoint_path")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["MyModel"] if isinstance(checkpoint, dict) and "MyModel" in checkpoint else checkpoint
    _validate_scoring_head_checkpoint(args, state_dict)
    if any(k.startswith("slot_mappings.") for k in state_dict.keys()):
        if not getattr(args, "per_slot_mapping", False):
            print("[INFO] checkpoint has slot_mappings — auto-enabling per_slot_mapping")
            args.per_slot_mapping = True

    model = Context_Prompting(args=args).to(device)
    drop_prefixes = (
        "PFL_",
        "context_encoder",
        "context_decoder",
        "state_encoder",
        "state_decoder",
    )
    checkpoint_has_pfl = any(
        key.startswith(("PFL_context.", "PFL_normal.", "PFL_abnormal.")) for key in state_dict.keys()
    )
    model_has_pfl = all(hasattr(model, name) for name in ("PFL_context", "PFL_normal", "PFL_abnormal"))

    model_has_scoring_head = bool(getattr(args, "use_scoring_head", False))
    checkpoint_has_scoring_head = any(k.startswith("scoring_head.") for k in state_dict.keys())

    model_has_slot_mappings = getattr(args, "per_slot_mapping", False)
    checkpoint_has_slot_mappings = any(k.startswith("slot_mappings.") for k in state_dict.keys())
    checkpoint_has_class_mapping = any(k.startswith("class_mapping.") for k in state_dict.keys())

    strict = True
    load_state = state_dict
    if checkpoint_has_pfl and not model_has_pfl:
        load_state = {k: v for k, v in state_dict.items() if not k.startswith(drop_prefixes)}
        strict = False
    elif (not checkpoint_has_pfl) and model_has_pfl and bool(getattr(args, "zero_shot_scoring", False)):
        strict = False
    if (not model_has_scoring_head) and checkpoint_has_scoring_head:
        load_state = {k: v for k, v in load_state.items() if not k.startswith("scoring_head.")}
        strict = False

    if model_has_slot_mappings and not checkpoint_has_slot_mappings and checkpoint_has_class_mapping:
        load_state = {k: v for k, v in load_state.items() if not k.startswith("class_mapping.")}
        strict = False
    elif not model_has_slot_mappings and checkpoint_has_slot_mappings:
        load_state = {k: v for k, v in load_state.items() if not k.startswith("slot_mappings.")}
        strict = False

    incompat = model.load_state_dict(load_state, strict=strict)

    if model_has_slot_mappings and not checkpoint_has_slot_mappings and checkpoint_has_class_mapping:
        cm_weight = state_dict["class_mapping.weight"]
        cm_bias = state_dict.get("class_mapping.bias")
        for slot_linear in model.slot_mappings:
            slot_linear.weight.data.copy_(cm_weight)
            if cm_bias is not None and slot_linear.bias is not None:
                slot_linear.bias.data.copy_(cm_bias)
        print("[INFO] per_slot_mapping: initialized all slot_mappings from shared class_mapping weights")
    if not strict:
        missing = list(getattr(incompat, "missing_keys", []))
        unexpected = list(getattr(incompat, "unexpected_keys", []))
        if missing or unexpected:
            print(
                "[INFO] prompt_bank scorer loaded checkpoint with compatibility mode | "
                f"missing={len(missing)} unexpected={len(unexpected)}"
            )
    model.eval()

    for param in model.parameters():
        param.requires_grad = False
    for param in model_clip.parameters():
        param.requires_grad = False

    return PromptBankScorer(model, model_clip, text_encoder, tokenize, args, device)


def _build_custom_scorer(args, device: torch.device) -> AnomalyScorer:
    scorer_module = getattr(args, "scorer_module", "")
    scorer_class = getattr(args, "scorer_class", "")
    if not scorer_module or not scorer_class:
        raise ValueError("--scorer_type custom requires --scorer_module and --scorer_class")

    scorer_kwargs = {}
    scorer_config = getattr(args, "scorer_config", "")
    if scorer_config:
        if scorer_config.endswith((".yaml", ".yml")):
            import yaml
            with open(scorer_config, "r") as f:
                scorer_kwargs = yaml.safe_load(f) or {}
        else:
            with open(scorer_config, "r") as f:
                scorer_kwargs = json.load(f)

    scorer_kwargs_json = getattr(args, "scorer_kwargs_json", "")
    if scorer_kwargs_json and not scorer_kwargs:
        scorer_kwargs = json.loads(scorer_kwargs_json)

    mod = importlib.import_module(scorer_module)
    cls = getattr(mod, scorer_class)

    for trial_kwargs in [
        {"args": args, "device": device, **scorer_kwargs},
        {"device": device, **scorer_kwargs},
        {**scorer_kwargs},
    ]:
        try:
            scorer = cls(**trial_kwargs)
            break
        except TypeError:
            continue
    else:
        raise TypeError(
            f"Failed to instantiate {scorer_class}. Constructor must accept "
            "(args, device, **kwargs) or (device, **kwargs) or (**kwargs)."
        )

    required_methods = ("score_candidate", "infer")
    missing = [m for m in required_methods if not callable(getattr(scorer, m, None))]
    if missing:
        raise TypeError(f"Custom scorer missing required methods: {missing}")
    return scorer


def _build_external_scorer(args, device: torch.device) -> AnomalyScorer:
    from models.external_scorer import build_external_scorer
    return build_external_scorer(args, device)
