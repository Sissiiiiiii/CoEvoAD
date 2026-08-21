"""Universal External Model Adapter for CoEvo Stage 2.

Provides a generic adapter interface so ANY CLIP-based anomaly detection model
can serve as Stage 1 backbone, with CoEvo prompt evolution and test pipeline
running on top via the existing AnomalyScorer interface.

Architecture:
    ExternalModelAdapter (ABC)  -- 4 abstract methods per external model
        ↓
    ExternalScorer (AnomalyScorer)  -- bridge that implements full scorer API
        ↓
    optimize_universal.py / test_universal.py  -- unchanged
"""

from __future__ import annotations

import abc
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from models.scorer import AnomalyScorer


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract adapter interface
# ---------------------------------------------------------------------------

class ExternalModelAdapter(abc.ABC):
    """Interface that any CLIP-based AD model must implement to work with CoEvo."""

    @abc.abstractmethod
    def encode_image(self, images: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        """Pre-compute image features.

        Args:
            images: [B, 3, H, W] on correct device.

        Returns:
            image_features: [B, D] L2-normalised global features.
            patch_context:  opaque object (e.g. multi-layer patch tokens) used
                            by compute_image_scores / compute_pixel_maps.
        """

    @abc.abstractmethod
    def encode_text(self, prompts: List[str]) -> torch.Tensor:
        """Encode arbitrary free-form text through the model's text encoder.

        Args:
            prompts: list of N text strings.

        Returns:
            [N, D] L2-normalised text features.
        """

    @abc.abstractmethod
    def compute_image_scores(
        self,
        image_features: torch.Tensor,
        patch_context: Any,
        text_normal: torch.Tensor,
        text_abnormal: torch.Tensor,
    ) -> torch.Tensor:
        """Image-level anomaly scores.

        Returns:
            [B] float tensor, higher = more anomalous.
        """

    @abc.abstractmethod
    def compute_pixel_maps(
        self,
        image_features: torch.Tensor,
        patch_context: Any,
        text_normal: torch.Tensor,
        text_abnormal: torch.Tensor,
        output_size: int,
    ) -> Optional[torch.Tensor]:
        """Pixel-level anomaly maps.

        Returns:
            [B, H, W] float tensor, or None if not supported.
        """

    @property
    @abc.abstractmethod
    def embed_dim(self) -> int:
        """Text/image embedding dimension (e.g. 768 for ViT-L)."""

    @property
    @abc.abstractmethod
    def device(self) -> torch.device:
        ...

    @property
    @abc.abstractmethod
    def image_size(self) -> int:
        ...


# ---------------------------------------------------------------------------
# Bridge: ExternalModelAdapter -> AnomalyScorer
# ---------------------------------------------------------------------------

class ExternalScorer(AnomalyScorer):
    """Implements the full AnomalyScorer API by delegating to an adapter."""

    def __init__(self, adapter: ExternalModelAdapter, args, device: torch.device):
        self.adapter = adapter
        self.args = args
        self._device = device
        self._image_size = adapter.image_size
        self.features_list = getattr(args, "features_list", [6, 12, 18, 24])
        self._baseline_emb_cache: Dict[str, torch.Tensor] = {}
        self._logged_modes: set[str] = set()

        # Expose model_clip / tokenizer for test_universal.py compatibility
        # (semantic fallback, visual routing access these attributes directly).
        self.model_clip = getattr(adapter, "clip_model", None)
        self.tokenizer = getattr(adapter, "tokenizer", None)

    # -- helpers --

    @property
    def device(self) -> torch.device:
        return self._device

    @staticmethod
    def _build_role_prompts(candidate, baseline, role, batch_size=1):
        if role == "normal":
            return [candidate] * batch_size, [baseline] * batch_size
        return [baseline] * batch_size, [candidate] * batch_size

    def _encode_pair(self, normal_prompt: str, abnormal_prompt: str):
        feats = self.adapter.encode_text([normal_prompt, abnormal_prompt])
        return feats[0:1], feats[1:2]  # each [1, D]

    def _runtime_logger(self):
        for name in ("test_universal", "optimize_universal"):
            runtime_logger = logging.getLogger(name)
            if runtime_logger.handlers:
                return runtime_logger
        return logger

    def _log_mode_once(self, mode: str, message: str):
        if mode in self._logged_modes:
            return
        self._logged_modes.add(mode)
        self._runtime_logger().info(message)

    # -- AnomalyScorer mandatory methods --

    def prepare_images(self, images: torch.Tensor):
        return self.adapter.encode_image(images.to(self._device))

    def score_candidate(
        self,
        prepared,
        candidate: str,
        role: str,
        baseline: str,
    ) -> torch.Tensor:
        result = self.evaluate_candidate(prepared, candidate, role, baseline)
        return result["image_scores"]

    def evaluate_candidate(
        self,
        prepared,
        candidate: str,
        role: str,
        baseline: str,
    ) -> Dict[str, Any]:
        self._log_mode_once(
            "hybrid_coevo",
            "ExternalScorer path: hybrid_coevo (vanilla text encoder + generic scorer)",
        )
        image_features, patch_context = prepared
        batch_size = image_features.shape[0]

        if role == "normal":
            normal_prompt, abnormal_prompt = candidate, baseline
        else:
            normal_prompt, abnormal_prompt = baseline, candidate

        # Baseline caching (same pattern as PromptBankScorer)
        baseline_role = "abnormal" if role == "normal" else "normal"
        cache_key = f"{baseline_role}:{baseline}"
        cached = self._baseline_emb_cache.get(cache_key)

        with torch.no_grad():
            cand_feat = self.adapter.encode_text([candidate])  # [1, D]
            if cached is not None:
                baseline_feat = cached.to(self._device)
            else:
                baseline_feat = self.adapter.encode_text([baseline])  # [1, D]
                self._baseline_emb_cache[cache_key] = baseline_feat.detach()

            if role == "normal":
                text_n, text_a = cand_feat, baseline_feat
            else:
                text_n, text_a = baseline_feat, cand_feat

            scores = self.adapter.compute_image_scores(
                image_features, patch_context, text_n, text_a,
            )
            pixel_maps = self.adapter.compute_pixel_maps(
                image_features, patch_context, text_n, text_a, self._image_size,
            )

        return {"image_scores": scores, "pixel_maps": pixel_maps}

    def evaluate_prompt_pair(
        self,
        prepared,
        normal_prompt: str,
        abnormal_prompt: str,
        stage: int = 2,
    ) -> Dict[str, Any]:
        self._log_mode_once(
            "hybrid_coevo",
            "ExternalScorer path: hybrid_coevo (vanilla text encoder + generic scorer)",
        )
        image_features, patch_context = prepared
        with torch.no_grad():
            text_n, text_a = self._encode_pair(normal_prompt, abnormal_prompt)
            scores = self.adapter.compute_image_scores(
                image_features, patch_context, text_n, text_a,
            )
            pixel_maps = self.adapter.compute_pixel_maps(
                image_features, patch_context, text_n, text_a, self._image_size,
            )
        return {"image_scores": scores, "pixel_maps": pixel_maps}

    def infer(
        self,
        images: torch.Tensor,
        normal_prompt: str,
        abnormal_prompt: str,
        stage: int = 2,
    ) -> Dict[str, Any]:
        prepared = self.prepare_images(images)
        return self.evaluate_prompt_pair(prepared, normal_prompt, abnormal_prompt, stage)

    def prepare_prompt_pair(
        self,
        normal_prompt: str,
        abnormal_prompt: str,
        stage: int = 2,
    ) -> Dict[str, Any]:
        with torch.no_grad():
            text_n, text_a = self._encode_pair(normal_prompt, abnormal_prompt)
            # Concatenate as [1, 2, D] to match PromptBankScorer format
            # (safe-switch mixing expects "text_embeddings" with ndim==3)
            text_embeddings = torch.cat([text_n, text_a], dim=0).unsqueeze(0)  # [1, 2, D]
        return {
            "normal_prompt": normal_prompt,
            "abnormal_prompt": abnormal_prompt,
            "text_embeddings": text_embeddings.detach(),
        }

    def infer_prepared(
        self,
        images: torch.Tensor,
        prepared_prompt: Dict[str, Any],
        stage: int = 2,
    ) -> Dict[str, Any]:
        image_features, patch_context = self.prepare_images(images)
        # Support both concatenated (from prepare_prompt_pair / safe-switch mix)
        # and split (legacy) formats
        text_emb = prepared_prompt.get("text_embeddings")
        if text_emb is not None:
            text_emb = text_emb.to(self._device)
            if text_emb.ndim == 3:
                half = text_emb.size(1) // 2
                text_n = text_emb[:, :half]  # [1, half, D]
                text_a = text_emb[:, half:]  # [1, half, D]
                # Squeeze prompt dim if only 1 prompt per role
                text_n = text_n.reshape(-1, text_n.shape[-1])  # [half, D]
                text_a = text_a.reshape(-1, text_a.shape[-1])  # [half, D]
            else:
                half = text_emb.size(0) // 2
                text_n = text_emb[:half]
                text_a = text_emb[half:]
        else:
            text_n = prepared_prompt["text_features_normal"].to(self._device)
            text_a = prepared_prompt["text_features_abnormal"].to(self._device)
        with torch.no_grad():
            scores = self.adapter.compute_image_scores(
                image_features, patch_context, text_n, text_a,
            )
            pixel_maps = self.adapter.compute_pixel_maps(
                image_features, patch_context, text_n, text_a, self._image_size,
            )
        return {"image_scores": scores, "pixel_maps": pixel_maps}

    def get_text_embedding(self, prompt: str, role: str) -> Optional[torch.Tensor]:
        with torch.no_grad():
            feat = self.adapter.encode_text([prompt])  # [1, D]
        return feat.squeeze(0).cpu()

    def get_official_embeddings(self, class_name: str = "object") -> Dict[str, Any]:
        """Return text embeddings from the adapter's official inference path.

        For AnomalyCLIP, uses the trained prompt_learner (encode_text_with_learner).
        Falls back to encode_text with generic prompts for other adapters.
        """
        if hasattr(self.adapter, "encode_text_with_learner"):
            text_n, text_a = self.adapter.encode_text_with_learner()
        else:
            text_n, text_a = self._encode_pair(
                f"a photo of a normal {class_name}",
                f"a photo of a damaged {class_name}",
            )
        text_embeddings = torch.cat([text_n, text_a], dim=0).unsqueeze(0)  # [1, 2, D]
        return {
            "normal_prompt": "[official_normal]",
            "abnormal_prompt": "[official_abnormal]",
            "text_embeddings": text_embeddings.detach(),
        }

    def infer_official_baseline(
        self,
        images: torch.Tensor,
        class_name: str,
        stage: int = 2,
    ) -> Dict[str, Any]:
        self._log_mode_once(
            "official_baseline",
            "ExternalScorer path: official_baseline (prompt_learner + official similarity/map)",
        )
        infer_fn = getattr(self.adapter, "infer_official_baseline", None)
        if callable(infer_fn):
            return infer_fn(images, class_name=class_name, output_size=self._image_size)
        official_prep = self.get_official_embeddings(class_name)
        return self.infer_prepared(images, official_prep, stage=stage)

    def clear_baseline_cache(self):
        self._baseline_emb_cache.clear()


# ---------------------------------------------------------------------------
# AnomalyCLIP adapter
# ---------------------------------------------------------------------------

class AnomalyCLIPAdapter(ExternalModelAdapter):
    """Adapter for AnomalyCLIP (ICLR 2024).

    Provides three text encoding paths:
    1. encode_text_with_learner() — official prompt_learner + encode_text_learn
    2. encode_text_ensemble(class_name) — 7×35 normal + 5×35 abnormal templates
    3. encode_text(prompts) — vanilla CLIP for CoEvo arbitrary text
    """

    def __init__(
        self,
        clip_model_path: str,
        checkpoint_path: str,
        device: torch.device,
        image_size: int = 518,
        features_list: Optional[List[int]] = None,
        dpam_layer: int = 20,
    ):
        self._device = device
        self._image_size = image_size
        self._features_list = features_list or [6, 12, 18, 24]
        self._dpam_layer = dpam_layer
        self._official_text_features_cache: Optional[torch.Tensor] = None

        # Design details matching the default AnomalyCLIP config (9/12/4)
        design_details = {
            "Prompt_length": 12,
            "learnabel_text_embedding_depth": 9,
            "learnabel_text_embedding_length": 4,
        }

        # Load the CLIP model with AnomalyCLIP architecture
        from models.external.anomalyclip.model_load import load as ac_load
        clip_model, _ = ac_load(
            clip_model_path,
            device=device,
            design_details=design_details,
        )

        # Apply DPAM (Dual-Path Attention Modification) to vision encoder
        clip_model.visual.DAPM_replace(DPAM_layer=dpam_layer)
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False
        self.clip_model = clip_model

        # Use learned logit_scale instead of hardcoded temperature
        self._temperature = 1.0 / clip_model.logit_scale.exp().item()

        # Load prompt_learner with full state (ctx_pos, ctx_neg, compound_prompts)
        from models.external.anomalyclip.prompt_ensemble import (
            AnomalyCLIP_PromptLearner,
        )
        self.prompt_learner = AnomalyCLIP_PromptLearner(clip_model, design_details)
        if checkpoint_path and os.path.isfile(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            pl_state = self._extract_prompt_learner_state(ckpt)
            missing, unexpected = self.prompt_learner.load_state_dict(
                pl_state, strict=False,
            )
            critical_missing = [
                key for key in missing
                if key in {"ctx_pos", "ctx_neg"}
                or key.startswith("compound_prompts_text")
                or key.startswith("compound_prompt_projections")
            ]
            if critical_missing:
                raise ValueError(
                    "AnomalyCLIP prompt_learner missing critical keys: "
                    f"{critical_missing}"
                )
            logger.info(
                "Loaded AnomalyCLIP prompt_learner: ctx_pos=%s ctx_neg=%s compound=%d unexpected=%d",
                tuple(pl_state["ctx_pos"].shape),
                tuple(pl_state["ctx_neg"].shape),
                sum(1 for k in pl_state if k.startswith("compound_prompts_text")),
                len(unexpected),
            )
            if unexpected:
                logger.warning("AnomalyCLIP prompt_learner unexpected keys: %s", unexpected)
        else:
            raise ValueError(
                "AnomalyCLIP adapter requires a prompt_learner checkpoint. "
                f"checkpoint_path={checkpoint_path!r}"
            )
        self.prompt_learner.to(device).eval()
        for p in self.prompt_learner.parameters():
            p.requires_grad = False

        # Tokenizer
        from models.external.anomalyclip.prompt_ensemble import tokenize
        self.tokenizer = tokenize

        self._embed_dim = int(clip_model.text_projection.shape[1])

    # -- properties --

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def image_size(self) -> int:
        return self._image_size

    # -- core methods --

    @staticmethod
    def _is_state_dict_like(obj: Any) -> bool:
        return isinstance(obj, dict) and all(isinstance(k, str) for k in obj.keys())

    @staticmethod
    def _strip_prefix(state: Dict[str, Any], prefix: str) -> Dict[str, Any]:
        return {
            key[len(prefix):]: value
            for key, value in state.items()
            if key.startswith(prefix)
        }

    @staticmethod
    def _has_critical_prompt_keys(state: Dict[str, Any]) -> bool:
        if "ctx_pos" not in state or "ctx_neg" not in state:
            return False
        has_compound_text = any(
            key.startswith("compound_prompts_text") for key in state.keys()
        )
        has_compound_proj = any(
            key.startswith("compound_prompt_projections") for key in state.keys()
        )
        return has_compound_text and has_compound_proj

    def _extract_prompt_learner_state(self, ckpt: Dict[str, Any]) -> Dict[str, Any]:
        candidates: List[Tuple[str, Dict[str, Any]]] = []

        if self._is_state_dict_like(ckpt):
            nested = ckpt.get("prompt_learner")
            if self._is_state_dict_like(nested):
                candidates.append(("prompt_learner", nested))

            stripped = self._strip_prefix(ckpt, "prompt_learner.")
            if stripped:
                candidates.append(("prompt_learner.*", stripped))

            state_dict = ckpt.get("state_dict")
            if self._is_state_dict_like(state_dict):
                nested_state = state_dict.get("prompt_learner")
                if self._is_state_dict_like(nested_state):
                    candidates.append(("state_dict.prompt_learner", nested_state))

                stripped_state = self._strip_prefix(state_dict, "prompt_learner.")
                if stripped_state:
                    candidates.append(("state_dict prompt_learner.*", stripped_state))

                candidates.append(("state_dict", state_dict))

            candidates.append(("checkpoint", ckpt))

        for source_name, candidate in candidates:
            if self._has_critical_prompt_keys(candidate):
                logger.info("Resolved prompt_learner state from %s", source_name)
                return candidate

        raise ValueError(
            "Failed to resolve AnomalyCLIP prompt_learner state from checkpoint. "
            "Expected ctx_pos, ctx_neg, compound_prompts_text.*, and "
            "compound_prompt_projections.* keys."
        )

    def encode_image(self, images: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        with torch.no_grad():
            image_features, patch_features = self.clip_model.encode_image(
                images.to(self._device),
                feature_list=self._features_list,
                DPAM_layer=self._dpam_layer,
            )
            image_features = F.normalize(image_features.float(), dim=-1)
        return image_features, patch_features

    def encode_text(self, prompts: List[str]) -> torch.Tensor:
        """Encode arbitrary text through AnomalyCLIP text transformer.

        Injects compound prompts at intermediate layers (same as official path).
        AnomalyCLIP's text transformer uses ResidualAttentionBlock_learnable_token
        which requires list input [x, compound_prompts, counter]; calling the
        vanilla clip_model.encode_text() would crash.
        """
        with torch.no_grad():
            tokens = self.tokenizer(prompts).to(self._device)
            x = self.clip_model.token_embedding(tokens).type(self.clip_model.dtype)
            x = x + self.clip_model.positional_embedding.type(self.clip_model.dtype)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.clip_model.transformer(
                [x, self.prompt_learner.compound_prompts_text, 0]
            )
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.clip_model.ln_final(x).type(self.clip_model.dtype)
            x = x[torch.arange(x.shape[0]), tokens.argmax(dim=-1)]
            x = x @ self.clip_model.text_projection
            x = F.normalize(x.float(), dim=-1)
        return x

    def encode_text_with_learner(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Official AnomalyCLIP text encoding: prompt_learner → encode_text_learn.

        Returns:
            text_normal: [1, D] L2-normalised normal embedding
            text_abnormal: [1, D] L2-normalised abnormal embedding
        """
        with torch.no_grad():
            text_features = self._get_official_text_features()
            text_normal = text_features[:, 0, :].mean(dim=0, keepdim=True)
            text_abnormal = text_features[:, 1, :].mean(dim=0, keepdim=True)
        return text_normal, text_abnormal

    def _get_official_text_features(self) -> torch.Tensor:
        """Official prompt_learner features arranged as [N_cls, 2, D]."""
        if self._official_text_features_cache is not None:
            return self._official_text_features_cache

        with torch.no_grad():
            prompts, tokenized_prompts, compound_prompts = self.prompt_learner()
            text_features = self.clip_model.encode_text_learn(
                prompts, tokenized_prompts, compound_prompts,
            ).float()
            normal_num = int(getattr(self.prompt_learner, "normal_num", 1))
            abnormal_num = int(getattr(self.prompt_learner, "anormaly_num", 1))
            if normal_num <= 0 or abnormal_num <= 0:
                raise ValueError(
                    "AnomalyCLIP prompt_learner has invalid prompt counts: "
                    f"normal_num={normal_num}, anormaly_num={abnormal_num}"
                )

            prompts_per_class = normal_num + abnormal_num
            total_prompts = text_features.shape[0]
            if total_prompts % prompts_per_class != 0:
                raise ValueError(
                    "AnomalyCLIP prompt_learner produced incompatible text feature count: "
                    f"total={total_prompts}, per_class={prompts_per_class}"
                )

            n_cls = total_prompts // prompts_per_class
            text_features = text_features.reshape(n_cls, prompts_per_class, -1)
            normal_feats = text_features[:, :normal_num, :].mean(dim=1, keepdim=True)
            abnormal_feats = text_features[:, normal_num:, :].mean(dim=1, keepdim=True)
            text_features = torch.cat([normal_feats, abnormal_feats], dim=1)
            text_features = F.normalize(text_features, dim=-1)
            self._official_text_features_cache = text_features
        return self._official_text_features_cache

    def encode_text_ensemble(self, class_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Official AnomalyCLIP prompt ensemble (7 normal × 35 templates, etc.).

        Returns:
            text_normal: [1, D] L2-normalised
            text_abnormal: [1, D] L2-normalised
        """
        from models.external.anomalyclip.prompt_ensemble import (
            encode_text_with_prompt_ensemble,
        )
        with torch.no_grad():
            # Returns [2, D]: row 0 = normal_mean, row 1 = abnormal_mean
            text_feats = encode_text_with_prompt_ensemble(
                self.clip_model, [class_name], self._device,
            )
        return text_feats[0:1], text_feats[1:2]

    def infer_official_baseline(
        self,
        images: torch.Tensor,
        class_name: str,
        output_size: int,
    ) -> Dict[str, Any]:
        """Official-style AnomalyCLIP baseline inference path."""
        del class_name  # object-agnostic prompt_learner path
        from models.external.anomalyclip.model_load import (
            compute_similarity,
            get_similarity_map,
        )

        with torch.no_grad():
            # Average across M learnable prompts (official test.py pattern)
            text_features = self._get_official_text_features()  # [M, 2, D]
            text_features = text_features.mean(dim=0, keepdim=True)  # [1, 2, D]
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            image_features, patch_features = self.clip_model.encode_image(
                images.to(self._device),
                feature_list=self._features_list,
                DPAM_layer=self._dpam_layer,
            )
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            # [B, D] @ [1, D, 2] -> [1, B, 2]; take [0, :, 1] -> [B]
            text_probs = image_features @ text_features.permute(0, 2, 1)
            text_probs = (text_probs / 0.07).softmax(-1)
            text_probs = text_probs[0, :, 1]  # [B]

            anomaly_map_list = []
            for patch_feature in patch_features:
                patch_feature = patch_feature / patch_feature.norm(dim=-1, keepdim=True)
                similarity, _ = compute_similarity(patch_feature, text_features[0])  # [2, D]
                similarity_map = get_similarity_map(similarity[:, 1:, :], output_size)
                anomaly_map = (similarity_map[..., 1] + 1.0 - similarity_map[..., 0]) / 2.0
                anomaly_map_list.append(anomaly_map)

            if not anomaly_map_list:
                pixel_maps = None
            else:
                pixel_maps = torch.stack(anomaly_map_list).sum(dim=0)

        return {"image_scores": text_probs, "pixel_maps": pixel_maps}

    def compute_image_scores(
        self,
        image_features: torch.Tensor,
        patch_context: Any,
        text_normal: torch.Tensor,
        text_abnormal: torch.Tensor,
    ) -> torch.Tensor:
        # text_normal: [1, D], text_abnormal: [1, D]
        # image_features: [B, D]
        text_feats = torch.cat([text_normal, text_abnormal], dim=0)  # [2, D]
        logits = image_features @ text_feats.T / self._temperature  # [B, 2]
        probs = logits.softmax(dim=-1)
        return probs[:, 1]  # P(abnormal)

    def compute_pixel_maps(
        self,
        image_features: torch.Tensor,
        patch_context: Any,
        text_normal: torch.Tensor,
        text_abnormal: torch.Tensor,
        output_size: int,
    ) -> Optional[torch.Tensor]:
        patch_features_list = patch_context  # list of 4 × [B, num_patches+1, D]
        if not patch_features_list:
            return None

        text_feats = torch.cat([text_normal, text_abnormal], dim=0)  # [2, D]
        layer_maps = []

        for patch_feat in patch_features_list:
            # Normalize patch features, skip CLS token
            patches = F.normalize(patch_feat[:, 1:, :].float(), dim=-1)  # [B, P, D]
            # Compute similarity
            sim = patches @ text_feats.T / self._temperature  # [B, P, 2]
            sim = sim.softmax(dim=-1)
            # Anomaly map: (abnormal_prob + 1 - normal_prob) / 2
            anomaly = (sim[..., 1] + 1.0 - sim[..., 0]) / 2.0  # [B, P]
            # Reshape to spatial grid
            side = int(anomaly.shape[1] ** 0.5)
            amap = anomaly.reshape(-1, 1, side, side)
            amap = F.interpolate(
                amap, size=(output_size, output_size),
                mode="bilinear", align_corners=False,
            )
            layer_maps.append(amap.squeeze(1))  # [B, H, W]

        return torch.stack(layer_maps, dim=0).mean(dim=0)  # mean across layers


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_ADAPTER_REGISTRY = {
    "anomalyclip": AnomalyCLIPAdapter,
}


def build_external_scorer(args, device: torch.device) -> ExternalScorer:
    """Build an ExternalScorer from CLI args."""
    adapter_name = getattr(args, "external_adapter", "anomalyclip")
    adapter_cls = _ADAPTER_REGISTRY.get(adapter_name)
    if adapter_cls is None:
        raise ValueError(
            f"Unknown external adapter: {adapter_name}. "
            f"Available: {list(_ADAPTER_REGISTRY.keys())}"
        )

    if adapter_name == "anomalyclip":
        pretrained_path = getattr(args, "pretrained_path", "")
        checkpoint_path = getattr(args, "checkpoint_path", "")
        if not pretrained_path:
            raise ValueError("AnomalyCLIP adapter requires --pretrained_path (CLIP weights)")
        adapter = AnomalyCLIPAdapter(
            clip_model_path=pretrained_path,
            checkpoint_path=checkpoint_path,
            device=device,
            image_size=int(getattr(args, "image_size", 518)),
            features_list=getattr(args, "features_list", [6, 12, 18, 24]),
        )
    else:
        adapter = adapter_cls(args=args, device=device)

    return ExternalScorer(adapter=adapter, args=args, device=device)
