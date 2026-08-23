"""
City of LA zoning and transit overlays, via ArcGIS REST.

Two design notes:

* **Layer discovery.** ArcGIS layer indices move when a service is
  republished, so layers are located by name against the service catalog
  rather than pinned to ``/MapServer/3``.

* **Local spatial join.** Querying point-in-polygon over REST once per parcel
  would be hundreds of thousands of round trips. Instead the polygon layer is
  downloaded once, cached, and joined locally with a shapely STRtree. LA City
  zoning is on the order of 50k polygons, which indexes in a second or two.

City of LA only. Roughly 800k of the county's 2.4M parcels fall inside city
limits; the rest come back with a null zone, which the development scorer
handles by falling back to neighbor-built FAR.
"""

from __future__ import annotations

import json
from typing import Any, Callable

import numpy as np
import pandas as pd
import requests

from cre import cache, config
from cre.sources.socrata import SourceResult

ARCGIS_PAGE_SIZE = 2000
REQUEST_TIMEOUT = 120

try:  # pragma: no cover - exercised by environment, not tests
    from shapely.geometry import Point, shape
    from shapely.strtree import STRtree

    SHAPELY_AVAILABLE = True
except ImportError:  # pragma: no cover
    SHAPELY_AVAILABLE = False


ZONE_FIELD_CANDIDATES = ["ZONE_CMPLT", "ZONE_SMRY", "ZONE", "ZONING", "zone_cmplt"]
TOC_FIELD_CANDIDATES = ["TOC_TIER", "TIER", "toc_tier", "Tier"]


def discover_layer(service_url: str, name_contains: str) -> tuple[str | None, list[str]]:
    """Find a layer whose name contains ``name_contains``. Returns (url, log)."""
    log: list[str] = []
    try:
        response = requests.get(
            f"{service_url}?f=json", timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        catalog = response.json()
    except (requests.RequestException, ValueError) as exc:
        log.append(f"{service_url}: {type(exc).__name__} {exc}"[:200])
        return None, log

    needle = name_contains.lower()
    for layer in catalog.get("layers", []):
        if needle in str(layer.get("name", "")).lower():
            log.append(f"matched layer {layer.get('id')} ({layer.get('name')})")
            return f"{service_url}/{layer.get('id')}", log

    log.append(
        f"no layer matching {name_contains!r}; saw: "
        + ", ".join(str(l.get("name")) for l in catalog.get("layers", [])[:10])
    )
    return None, log


def fetch_polygons(
    layer_url: str, progress: Callable[[int], None] | None = None
) -> tuple[list[dict[str, Any]], list[str]]:
    """Page an entire ArcGIS polygon layer down as GeoJSON features."""
    features: list[dict[str, Any]] = []
    notes: list[str] = []
    offset = 0

    while True:
        params = {
            "where": "1=1",
            "outFields": "*",
            "returnGeometry": "true",
            "outSR": 4326,
            "f": "geojson",
            "resultOffset": offset,
            "resultRecordCount": ARCGIS_PAGE_SIZE,
        }
        try:
            response = requests.get(layer_url + "/query", params=params, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            notes.append(f"page at offset {offset} failed: {exc}"[:200])
            break

        page = payload.get("features", [])
        if not page:
            break
        features.extend(page)
        if progress is not None:
            progress(len(features))
        if len(page) < ARCGIS_PAGE_SIZE:
            break
        offset += len(page)

    notes.append(f"downloaded {len(features):,} polygons")
    return features, notes


def _pick_field(properties: dict[str, Any], candidates: list[str]) -> str | None:
    lowered = {k.lower(): k for k in properties}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def spatial_join(
    parcels: pd.DataFrame,
    features: list[dict[str, Any]],
    value_field_candidates: list[str],
    output_column: str,
) -> pd.Series:
    """
    Assign each parcel the attribute of the polygon containing its centroid.

    Returns a series aligned to ``parcels``. Parcels without coordinates, or
    outside every polygon, come back null.
    """
    result = pd.Series(pd.NA, index=parcels.index, dtype="object")

    if not SHAPELY_AVAILABLE or not features:
        return result
    if "lat" not in parcels.columns or "lon" not in parcels.columns:
        return result

    geometries = []
    values = []
    for feature in features:
        geometry = feature.get("geometry")
        properties = feature.get("properties") or {}
        if not geometry:
            continue
        field = _pick_field(properties, value_field_candidates)
        if field is None:
            continue
        try:
            geometries.append(shape(geometry))
        except (ValueError, TypeError):
            continue
        values.append(properties.get(field))

    if not geometries:
        return result

    tree = STRtree(geometries)

    coords = parcels[["lat", "lon"]].dropna()
    if coords.empty:
        return result

    points = [Point(lon, lat) for lat, lon in zip(coords["lat"], coords["lon"])]
    # query with predicate returns (input_index, tree_index) pairs
    hits = tree.query(points, predicate="within")

    assigned: dict[Any, Any] = {}
    for point_position, polygon_position in zip(hits[0], hits[1]):
        label = coords.index[point_position]
        if label not in assigned:
            assigned[label] = values[polygon_position]

    for label, value in assigned.items():
        result.at[label] = value

    return result


def enrich(
    parcels: pd.DataFrame,
    use_cache: bool = True,
    progress: Callable[[int], None] | None = None,
) -> tuple[pd.DataFrame, SourceResult]:
    """
    Attach ``zone`` and ``toc_tier`` columns where LA City data covers the parcel.

    Never raises: any failure leaves the columns null and is reported through
    the returned ``SourceResult`` so the diagnostics tab can explain why
    entitlement scoring is unavailable.
    """
    out = parcels.copy()
    out["zone"] = pd.NA
    out["toc_tier"] = np.nan

    result = SourceResult(frame=pd.DataFrame(), domain="maps.lacity.org")

    if not SHAPELY_AVAILABLE:
        result.errors.append(
            "shapely is not installed, so zoning cannot be joined. "
            "Install it with `pip install shapely` to enable entitlement scoring."
        )
        return out, result

    for label, candidates, column, field_candidates in (
        ("zoning", config.ARCGIS_SERVICES["zoning"], "zone", ZONE_FIELD_CANDIDATES),
        ("toc", config.ARCGIS_SERVICES["toc"], "toc_tier", TOC_FIELD_CANDIDATES),
    ):
        features: list[dict[str, Any]] = []
        entry = cache.entry("arcgis", layer=label)

        cached = cache.read(entry, ttl_hours=config.CACHE_TTL_HOURS * 4)
        if use_cache and cached is not None and "feature" in cached.columns:
            features = [json.loads(f) for f in cached["feature"].dropna()]
            result.notes.append(f"{label}: {len(features):,} polygons from cache")

        if not features:
            for service_url in candidates:
                layer_url, log = discover_layer(service_url, label if label != "toc" else "transit")
                result.notes.extend(f"{label}: {line}" for line in log)
                if layer_url is None:
                    continue
                features, fetch_notes = fetch_polygons(layer_url, progress)
                result.notes.extend(f"{label}: {line}" for line in fetch_notes)
                if features:
                    break

            if features and use_cache:
                cache.write(
                    entry,
                    pd.DataFrame({"feature": [json.dumps(f) for f in features]}),
                    layer=label,
                )

        if not features:
            result.errors.append(
                f"Could not load the {label} layer. Parcels will score without it."
            )
            continue

        joined = spatial_join(out, features, field_candidates, column)
        if column == "toc_tier":
            out[column] = pd.to_numeric(joined, errors="coerce")
        else:
            out[column] = joined.astype("string")

        matched = int(out[column].notna().sum())
        result.notes.append(f"{label}: matched {matched:,} of {len(out):,} parcels")

    return out, result
