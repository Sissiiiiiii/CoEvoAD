"""MAP-Elites Quality-Diversity archive for prompt optimization.

Maintains a grid of behavior cells, each storing the highest-quality
prompt found for that behavioral niche.  Behavior is characterized by
a low-dimensional descriptor (e.g., image_auroc × pixel_f1).
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ArchiveEntry:
    prompt: str
    quality: float
    descriptor: Tuple[float, ...]
    cell: Tuple[int, ...]
    metadata: Dict[str, Any] = field(default_factory=dict)


class QDArchive:
    """MAP-Elites archive with fixed-range grid discretization.

    Parameters
    ----------
    bd_names : sequence of str
        Names of each behavior descriptor dimension (e.g., ["image_auroc", "pixel_f1"]).
    bins_per_dim : int
        Number of bins per BD dimension.  Grid has bins_per_dim^len(bd_names) cells.
    bd_ranges : sequence of (float, float) or None
        Fixed (min, max) range per BD dimension.  If None, ranges are learned
        from the first ``range_warmup`` insertions and then locked.
    range_warmup : int
        Number of insertions before locking BD ranges (only used when bd_ranges is None).
    min_quality : float
        Minimum quality threshold for archive admission.
    """

    def __init__(
        self,
        bd_names: Sequence[str] = ("image_auroc", "pixel_f1"),
        bins_per_dim: int = 5,
        bd_ranges: Optional[Sequence[Tuple[float, float]]] = None,
        range_warmup: int = 20,
        min_quality: float = 0.0,
    ):
        self.bd_names = list(bd_names)
        self.dims = len(self.bd_names)
        self.bins_per_dim = bins_per_dim
        self.min_quality = min_quality
        self.range_warmup = range_warmup

        self._grid: Dict[Tuple[int, ...], ArchiveEntry] = {}
        self._insertion_count = 0

        if bd_ranges is not None:
            self._bd_min = np.array([lo for lo, _ in bd_ranges], dtype=np.float64)
            self._bd_max = np.array([hi for _, hi in bd_ranges], dtype=np.float64)
            self._ranges_locked = True
        else:
            self._bd_min = np.full(self.dims, np.inf, dtype=np.float64)
            self._bd_max = np.full(self.dims, -np.inf, dtype=np.float64)
            self._ranges_locked = False

        self._warmup_buffer: List[Tuple[str, float, Tuple[float, ...], Optional[Dict[str, Any]]]] = []

    @property
    def size(self) -> int:
        return len(self._grid)

    @property
    def max_cells(self) -> int:
        return self.bins_per_dim ** self.dims

    @property
    def coverage(self) -> float:
        return self.size / max(self.max_cells, 1)

    def _to_cell(self, descriptor: Tuple[float, ...]) -> Tuple[int, ...]:
        bd = np.array(descriptor, dtype=np.float64)
        span = self._bd_max - self._bd_min
        span = np.where(span < 1e-12, 1.0, span)
        normalized = (bd - self._bd_min) / span
        normalized = np.clip(normalized, 0.0, 1.0 - 1e-9)
        cell = tuple(int(v * self.bins_per_dim) for v in normalized)
        return cell

    def _update_ranges(self, descriptor: Tuple[float, ...]):
        bd = np.array(descriptor, dtype=np.float64)
        self._bd_min = np.minimum(self._bd_min, bd)
        self._bd_max = np.maximum(self._bd_max, bd)

    def try_add(
        self,
        prompt: str,
        quality: float,
        descriptor: Tuple[float, ...],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        if quality < self.min_quality:
            return False

        self._insertion_count += 1

        if not self._ranges_locked:
            self._update_ranges(descriptor)
            self._warmup_buffer.append((prompt, quality, descriptor, metadata))

            if self._insertion_count >= self.range_warmup:
                self._ranges_locked = True
                margin = 0.05 * (self._bd_max - self._bd_min)
                margin = np.maximum(margin, 1e-6)
                self._bd_min -= margin
                self._bd_max += margin
                logger.info(
                    "QD archive ranges locked after %d insertions: %s",
                    self.range_warmup,
                    {n: (float(lo), float(hi)) for n, lo, hi in
                     zip(self.bd_names, self._bd_min, self._bd_max)},
                )
                for p, q, d, m in self._warmup_buffer:
                    self._insert(p, q, d, m)
                self._warmup_buffer.clear()
                return True
            return True

        return self._insert(prompt, quality, descriptor, metadata)

    def _insert(
        self,
        prompt: str,
        quality: float,
        descriptor: Tuple[float, ...],
        metadata: Optional[Dict[str, Any]],
    ) -> bool:
        cell = self._to_cell(descriptor)
        existing = self._grid.get(cell)
        if existing is None or quality > existing.quality:
            self._grid[cell] = ArchiveEntry(
                prompt=prompt,
                quality=quality,
                descriptor=descriptor,
                cell=cell,
                metadata=metadata or {},
            )
            return True
        return False

    def sample_parents(self, k: int) -> List[str]:
        if not self._grid:
            return []
        entries = list(self._grid.values())
        k = min(k, len(entries))
        chosen = random.sample(entries, k)
        return [e.prompt for e in chosen]

    def get_elites(self, sort_by_quality: bool = True) -> List[ArchiveEntry]:
        entries = list(self._grid.values())
        if sort_by_quality:
            entries.sort(key=lambda e: e.quality, reverse=True)
        return entries

    def get_best(self) -> Optional[ArchiveEntry]:
        if not self._grid:
            return None
        return max(self._grid.values(), key=lambda e: e.quality)

    def summary(self) -> Dict[str, Any]:
        if not self._grid:
            return {
                "size": 0, "coverage": 0.0,
                "quality_mean": 0.0, "quality_max": 0.0,
                "quality_min": 0.0, "quality_std": 0.0,
                "bd_ranges": {
                    n: (float(lo), float(hi))
                    for n, lo, hi in zip(self.bd_names, self._bd_min, self._bd_max)
                },
            }

        entries = list(self._grid.values())
        qualities = [e.quality for e in entries]
        return {
            "size": len(entries),
            "coverage": self.coverage,
            "quality_mean": float(np.mean(qualities)),
            "quality_max": float(np.max(qualities)),
            "quality_min": float(np.min(qualities)),
            "quality_std": float(np.std(qualities)),
            "bd_ranges": {
                n: (float(lo), float(hi))
                for n, lo, hi in zip(self.bd_names, self._bd_min, self._bd_max)
            },
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bd_names": self.bd_names,
            "bins_per_dim": self.bins_per_dim,
            "bd_ranges": [
                (float(lo), float(hi))
                for lo, hi in zip(self._bd_min, self._bd_max)
            ],
            "min_quality": self.min_quality,
            "entries": [
                {
                    "prompt": e.prompt,
                    "quality": e.quality,
                    "descriptor": list(e.descriptor),
                    "cell": list(e.cell),
                    "metadata": e.metadata,
                }
                for e in self._grid.values()
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QDArchive":
        archive = cls(
            bd_names=data["bd_names"],
            bins_per_dim=data["bins_per_dim"],
            bd_ranges=[(lo, hi) for lo, hi in data["bd_ranges"]],
            min_quality=data.get("min_quality", 0.0),
        )
        for entry in data.get("entries", []):
            archive._insert(
                prompt=entry["prompt"],
                quality=entry["quality"],
                descriptor=tuple(entry["descriptor"]),
                metadata=entry.get("metadata", {}),
            )
        return archive

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info("QD archive saved: %d entries to %s", self.size, path)

    @classmethod
    def load(cls, path: str) -> "QDArchive":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        archive = cls.from_dict(data)
        logger.info("QD archive loaded: %d entries from %s", archive.size, path)
        return archive
