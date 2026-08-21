"""Dump full maps and image scores for offline routing-ensemble replay."""
from __future__ import annotations

import json
import os
import shutil
from typing import Any, Dict

import numpy as np


SCHEMA_VERSION = 2
ARRAY_KEYS = ("anomaly_maps", "imgs_masks", "image_score", "gt_sp", "cls_name", "path")


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _to_hw_array(value: Any, *, name: str, dtype: np.dtype) -> np.ndarray:
    arr = np.squeeze(_to_numpy(value))
    if arr.ndim != 2:
        raise ValueError(f"{name} must resolve to a 2D HxW array; got shape {arr.shape}")
    return arr.astype(dtype, copy=False)


def _write_npz(path: str, payload: Dict[str, np.ndarray]) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent or ".", exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp_path, "wb") as handle:
            np.savez(
                handle,
                schema_version=np.asarray([SCHEMA_VERSION], dtype=np.int16),
                **payload,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _write_directory(path: str, payload: Dict[str, np.ndarray]) -> None:
    out_path = os.path.abspath(path)
    parent = os.path.dirname(out_path)
    os.makedirs(parent or ".", exist_ok=True)
    tmp_path = f"{out_path}.tmp.{os.getpid()}"

    if os.path.exists(tmp_path):
        shutil.rmtree(tmp_path)
    if os.path.exists(out_path):
        raise FileExistsError(
            f"ensemble dump directory already exists: {out_path}. "
            "Use a fresh run directory or remove the old diagnostic dump explicitly."
        )

    os.makedirs(tmp_path)
    try:
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "format": "ensemble_inputs_directory",
            "arrays": {
                key: {
                    "file": f"{key}.npy",
                    "dtype": str(payload[key].dtype),
                    "shape": list(payload[key].shape),
                }
                for key in ARRAY_KEYS
            },
        }
        for key in ARRAY_KEYS:
            np.save(os.path.join(tmp_path, f"{key}.npy"), payload[key], allow_pickle=False)
        with open(os.path.join(tmp_path, "manifest.json"), "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, out_path)
    finally:
        if os.path.exists(tmp_path):
            shutil.rmtree(tmp_path)


def dump_ensemble_inputs(path: str, results: Dict[str, Any]) -> None:
    """Write inputs for offline Sem/Tmpl routing ensemble replay.

    The live test loop stores the image-level score as ``results["pr_sp"]``.
    The dump exposes it as ``image_score`` to keep the replay schema explicit.
    Paths ending in ``.npz`` keep the legacy single-file format. Other paths
    write a directory of independent ``.npy`` arrays, which is safer for large
    full-map dumps because it avoids one giant zip finalization step.
    """
    if not path:
        return

    image_scores = results.get("image_score", results.get("pr_sp"))
    if image_scores is None:
        raise ValueError("results must contain 'pr_sp' or 'image_score'")

    n = len(results["cls_names"])
    lengths = {
        "anomaly_maps": len(results["anomaly_maps"]),
        "imgs_masks": len(results["imgs_masks"]),
        "image_score": len(image_scores),
        "gt_sp": len(results["gt_sp"]),
        "path": len(results["path"]),
    }
    bad = {key: value for key, value in lengths.items() if value != n}
    if bad:
        raise ValueError(
            f"results lists have mismatched lengths for ensemble dump: n={n}, bad={bad}"
        )

    anomaly_maps = []
    imgs_masks = []
    for i in range(n):
        anomaly_maps.append(
            _to_hw_array(results["anomaly_maps"][i], name=f"anomaly_maps[{i}]", dtype=np.float32)
        )
        mask = _to_hw_array(results["imgs_masks"][i], name=f"imgs_masks[{i}]", dtype=np.float32)
        imgs_masks.append((mask > 0.5).astype(np.uint8))

    anomaly_arr = np.stack(anomaly_maps, axis=0).astype(np.float32, copy=False)
    mask_arr = np.stack(imgs_masks, axis=0).astype(np.uint8, copy=False)
    image_score_arr = np.asarray([float(v) for v in image_scores], dtype=np.float32)
    gt_sp = np.asarray([int(v) for v in results["gt_sp"]], dtype=np.int8)
    cls_name = np.asarray([str(v) for v in results["cls_names"]])
    path_arr = np.asarray([str(v) for v in results["path"]])

    payload = {
        "anomaly_maps": anomaly_arr,
        "imgs_masks": mask_arr,
        "image_score": image_score_arr,
        "gt_sp": gt_sp,
        "cls_name": cls_name,
        "path": path_arr,
    }
    if path.lower().endswith(".npz"):
        _write_npz(path, payload)
    else:
        _write_directory(path, payload)
