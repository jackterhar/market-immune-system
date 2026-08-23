"""
Parquet-backed cache.

The countywide parcel pull is ~2.4M rows and takes minutes. Caching turns
that into a once-a-week cost and makes the app usable offline afterwards.
Cache misses are never fatal — a corrupt or unreadable cache file is deleted
and treated as absent.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from cre import config


@dataclass(frozen=True)
class CacheEntry:
    path: Path
    meta_path: Path
    key: str

    @property
    def exists(self) -> bool:
        return self.path.exists() and self.meta_path.exists()

    def age_hours(self) -> float | None:
        if not self.exists:
            return None
        return (time.time() - self.path.stat().st_mtime) / 3600.0

    def meta(self) -> dict[str, Any]:
        try:
            return json.loads(self.meta_path.read_text())
        except (OSError, ValueError):
            return {}


def _key(namespace: str, params: dict[str, Any]) -> str:
    blob = json.dumps(params, sort_keys=True, default=str)
    digest = hashlib.sha256(blob.encode()).hexdigest()[:16]
    return f"{namespace}-{digest}"


def entry(namespace: str, **params: Any) -> CacheEntry:
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = _key(namespace, params)
    return CacheEntry(
        path=config.CACHE_DIR / f"{key}.parquet",
        meta_path=config.CACHE_DIR / f"{key}.json",
        key=key,
    )


def read(item: CacheEntry, ttl_hours: float | None = None) -> pd.DataFrame | None:
    """Return the cached frame, or None if absent, stale, or unreadable."""
    if not item.exists:
        return None
    ttl = config.CACHE_TTL_HOURS if ttl_hours is None else ttl_hours
    age = item.age_hours()
    if age is not None and age > ttl:
        return None
    try:
        return pd.read_parquet(item.path)
    except Exception:
        # A truncated write from an interrupted run should not wedge the app.
        for path in (item.path, item.meta_path):
            path.unlink(missing_ok=True)
        return None


def write(item: CacheEntry, frame: pd.DataFrame, **meta: Any) -> None:
    """Persist a frame. Failures are non-fatal; the app just re-fetches."""
    try:
        # Socrata returns everything as strings; object columns with mixed
        # types break the parquet writer. Coerce them to string up front.
        safe = frame.copy()
        for column in safe.columns:
            if safe[column].dtype == object:
                safe[column] = safe[column].astype("string")
        safe.to_parquet(item.path, index=False)
        item.meta_path.write_text(
            json.dumps({"rows": len(frame), "written_at": time.time(), **meta}, default=str)
        )
    except Exception:
        item.path.unlink(missing_ok=True)
        item.meta_path.unlink(missing_ok=True)


def clear() -> int:
    """Delete every cached file. Returns the number of files removed."""
    if not config.CACHE_DIR.exists():
        return 0
    removed = 0
    for path in config.CACHE_DIR.iterdir():
        if path.suffix in {".parquet", ".json"}:
            path.unlink(missing_ok=True)
            removed += 1
    return removed


def summary() -> list[dict[str, Any]]:
    """Describe what is currently cached, for the diagnostics tab."""
    if not config.CACHE_DIR.exists():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(config.CACHE_DIR.glob("*.parquet")):
        meta_path = path.with_suffix(".json")
        meta: dict[str, Any] = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
            except ValueError:
                meta = {}
        rows.append(
            {
                "key": path.stem,
                "rows": meta.get("rows"),
                "size_mb": round(path.stat().st_size / 1e6, 1),
                "age_hours": round((time.time() - path.stat().st_mtime) / 3600.0, 1),
                "source": meta.get("dataset_id"),
            }
        )
    return rows
