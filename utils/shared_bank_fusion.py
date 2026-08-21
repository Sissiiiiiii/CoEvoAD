from typing import Any, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def _to_float_score(value: Any) -> float:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return 0.0
    return float(arr[0])


def _to_pixel_map(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim >= 3:
        arr = arr[0]
    if arr.ndim != 2:
        return None
    return arr


def normalize_weights(weights: Optional[Sequence[float]], n_items: int) -> np.ndarray:
    if n_items <= 0:
        return np.zeros((0,), dtype=np.float32)
    if weights is None:
        return np.full((n_items,), 1.0 / n_items, dtype=np.float32)
    w = np.asarray(weights, dtype=np.float32).reshape(-1)
    if w.size != n_items:
        return np.full((n_items,), 1.0 / n_items, dtype=np.float32)
    w = np.clip(w, a_min=0.0, a_max=None)
    s = float(w.sum())
    if s <= 0:
        return np.full((n_items,), 1.0 / n_items, dtype=np.float32)
    return w / s


def fuse_cls_and_shared(
    image_score_cls: Any,
    pixel_map_cls: Any,
    shared_items: Iterable[Tuple[Any, Any]],
    alpha: float,
    weights: Optional[Sequence[float]] = None,
    confidence_weighted: bool = False,
) -> Tuple[float, Optional[np.ndarray]]:
    shared_list: List[Tuple[Any, Any]] = list(shared_items)
    if len(shared_list) == 0 or alpha <= 0.0:
        return _to_float_score(image_score_cls), _to_pixel_map(pixel_map_cls)

    w = normalize_weights(weights, len(shared_list))
    shared_scores = np.asarray([_to_float_score(s) for s, _ in shared_list], dtype=np.float32)
    shared_score = float(np.sum(w * shared_scores))

    base_score = _to_float_score(image_score_cls)
    final_score = float((1.0 - alpha) * base_score + alpha * shared_score)

    # Keep original indices aligned with routing weights `w`
    shared_maps_with_orig_idx = []
    for orig_i, (_, m) in enumerate(shared_list):
        pm = _to_pixel_map(m)
        if pm is not None:
            shared_maps_with_orig_idx.append((orig_i, pm))

    cls_map = _to_pixel_map(pixel_map_cls)
    if len(shared_maps_with_orig_idx) == 0:
        return final_score, cls_map

    base_shape = shared_maps_with_orig_idx[0][1].shape
    valid = [(oi, m) for oi, m in shared_maps_with_orig_idx if m.shape == base_shape]
    if not valid:
        return final_score, cls_map

    valid_orig_idx = [oi for oi, _ in valid]
    valid_maps = np.stack([m for _, m in valid], axis=0)
    valid_weights = normalize_weights([w[i] for i in valid_orig_idx], len(valid_orig_idx))
    shared_map = np.tensordot(valid_weights, valid_maps, axes=(0, 0)).astype(np.float32)

    if cls_map is None:
        return final_score, shared_map
    if cls_map.shape != shared_map.shape:
        return final_score, cls_map

    if confidence_weighted:
        cls_confidence = np.abs(cls_map - 0.5) * 2.0  # [0, 1]
        pixel_alpha = alpha * (1.0 - cls_confidence)
        final_map = ((1.0 - pixel_alpha) * cls_map + pixel_alpha * shared_map).astype(np.float32)
    else:
        final_map = ((1.0 - alpha) * cls_map + alpha * shared_map).astype(np.float32)
    return final_score, final_map


def resolve_alpha_from_source(
    source: str,
    alpha_exact: float,
    alpha_semantic: float,
    alpha_missing: float,
    alpha_icsr: float = 0.0,
) -> float:
    source_l = str(source or "").lower()
    if ("exact" in source_l or "alias" in source_l) and "semantic" not in source_l:
        return float(alpha_exact)
    if "icsr" in source_l:
        return float(alpha_icsr)
    if "semantic" in source_l or "visual" in source_l:
        return float(alpha_semantic)
    return float(alpha_missing)


def softmax_from_similarities(similarities: Sequence[float], temperature: float = 0.07) -> np.ndarray:
    sims = np.asarray(similarities, dtype=np.float32).reshape(-1)
    if sims.size == 0:
        return np.zeros((0,), dtype=np.float32)
    t = max(float(temperature), 1e-4)
    z = sims / t
    z = z - np.max(z)
    exp_z = np.exp(z)
    denom = float(np.sum(exp_z))
    if denom <= 0:
        return np.full_like(sims, 1.0 / sims.size, dtype=np.float32)
    return (exp_z / denom).astype(np.float32)
