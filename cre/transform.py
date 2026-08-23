"""
Normalization and derived metrics for the assessor parcel roll.

The assessor publishes a new roll each year and the column names are not
stable between them. Rather than hard-coding one year's schema, this module
resolves each logical field against a list of candidate source columns and
*reports what it matched*. An unmatched field degrades the features that need
it instead of silently producing nulls that look like real zeros.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from cre import config


def _canon(name: str) -> str:
    """Collapse a column name to letters and digits for tolerant matching."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


@dataclass
class SchemaReport:
    """What resolved, what did not, and what that costs downstream."""

    resolved: dict[str, str] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    unmapped_source_columns: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(f in self.resolved for f in config.REQUIRED_FIELDS)

    def degradations(self) -> list[str]:
        """Human-readable consequences of each missing field."""
        impact = {
            "lot_sqft": "FAR and underbuilt scoring fall back to peer comparison only.",
            "building_sqft": "Value-per-square-foot and FAR are unavailable.",
            "land_base_year": "Ownership tenure cannot be estimated (Prop 13 proxy).",
            "imp_base_year": "Ownership tenure falls back to the land base year alone.",
            "total_value": "Basis-gap scoring is unavailable.",
            "improvement_value": "Teardown / underimprovement signal is unavailable.",
            "lat": "Map view is disabled.",
            "lon": "Map view is disabled.",
            "year_built": "Building-age signal is unavailable.",
            "specific_use": "Peer groups fall back to general use type.",
            "situs_city": "Peer groups fall back to county-wide medians.",
        }
        return [impact[f] for f in self.missing if f in impact]


def resolve_columns(frame: pd.DataFrame) -> SchemaReport:
    """Map logical field names onto the columns actually present."""
    lookup = {_canon(c): c for c in frame.columns}
    report = SchemaReport()
    used: set[str] = set()

    for logical, candidates in config.COLUMN_ALIASES.items():
        for candidate in candidates:
            source = lookup.get(_canon(candidate))
            if source is not None:
                report.resolved[logical] = source
                used.add(source)
                break
        else:
            report.missing.append(logical)

    report.unmapped_source_columns = sorted(set(frame.columns) - used)
    return report


def _to_numeric(series: pd.Series) -> pd.Series:
    """Coerce a Socrata string column to float, blanking non-numeric junk."""
    return pd.to_numeric(series, errors="coerce")


def normalize(frame: pd.DataFrame) -> tuple[pd.DataFrame, SchemaReport]:
    """
    Rename to canonical field names and coerce types.

    Returns the normalized frame alongside the schema report so callers can
    surface resolution problems rather than discovering them as empty charts.
    """
    report = resolve_columns(frame)
    out = pd.DataFrame(index=frame.index)

    for logical, source in report.resolved.items():
        out[logical] = frame[source]

    numeric_fields = [
        "year_built",
        "effective_year",
        "building_sqft",
        "lot_sqft",
        "units",
        "land_value",
        "improvement_value",
        "total_value",
        "land_base_year",
        "imp_base_year",
        "lat",
        "lon",
        "homeowner_exemption",
        "roll_year",
    ]
    for column in numeric_fields:
        if column in out.columns:
            out[column] = _to_numeric(out[column])

    for column in ("parcel_id", "situs_address", "situs_city", "situs_zip",
                   "general_use", "specific_use", "use_code"):
        if column in out.columns:
            out[column] = out[column].astype("string").str.strip()

    # Zero is the assessor's "unknown" for these; it is not a real year or area.
    for column in ("year_built", "effective_year", "land_base_year", "imp_base_year"):
        if column in out.columns:
            out[column] = out[column].replace(0, np.nan)
    for column in ("building_sqft", "lot_sqft"):
        if column in out.columns:
            out[column] = out[column].where(out[column] > 0)

    # Latitude/longitude arrive as strings and occasionally as 0/0.
    if "lat" in out.columns and "lon" in out.columns:
        in_la = (
            out["lat"].between(32.5, 35.0) & out["lon"].between(-119.5, -117.0)
        )
        out.loc[~in_la, ["lat", "lon"]] = np.nan

    return out, report


def add_derived_metrics(
    frame: pd.DataFrame, roll_year: int | None = None
) -> pd.DataFrame:
    """
    Add the derived columns the scoring engine consumes.

    Every metric is null-safe: a missing input yields NaN rather than an
    exception or a misleading zero.
    """
    out = frame.copy()
    year = roll_year or config.CURRENT_ROLL_YEAR_FALLBACK

    has_building = "building_sqft" in out.columns
    has_total = "total_value" in out.columns

    # Assessed dollars per built square foot. Under Prop 13 this reflects when
    # the parcel last changed hands as much as what it is worth.
    if has_building and has_total:
        out["value_per_sqft"] = out["total_value"] / out["building_sqft"]
        out.loc[~np.isfinite(out["value_per_sqft"]), "value_per_sqft"] = np.nan
    else:
        out["value_per_sqft"] = np.nan

    # Land value per lot square foot is the cleaner development-side measure:
    # it is unaffected by whatever happens to be standing on the site.
    if "land_value" in out.columns and "lot_sqft" in out.columns:
        out["land_value_per_lot_sqft"] = out["land_value"] / out["lot_sqft"]
        out.loc[
            ~np.isfinite(out["land_value_per_lot_sqft"]), "land_value_per_lot_sqft"
        ] = np.nan
    else:
        out["land_value_per_lot_sqft"] = np.nan

    # Built floor area ratio.
    if has_building and "lot_sqft" in out.columns:
        out["built_far"] = out["building_sqft"] / out["lot_sqft"]
        out.loc[~np.isfinite(out["built_far"]), "built_far"] = np.nan
    else:
        out["built_far"] = np.nan

    # Prop 13 base year resets on transfer or new construction, so its distance
    # from today approximates how long the current owner has held the asset.
    base_years = [c for c in ("land_base_year", "imp_base_year") if c in out.columns]
    if base_years:
        out["base_year"] = out[base_years].max(axis=1)
        out["tenure_years"] = year - out["base_year"]
        out.loc[out["tenure_years"] < 0, "tenure_years"] = np.nan
    else:
        out["base_year"] = np.nan
        out["tenure_years"] = np.nan

    if "year_built" in out.columns:
        out["building_age"] = year - out["year_built"]
        out.loc[out["building_age"] < 0, "building_age"] = np.nan
    else:
        out["building_age"] = np.nan

    # Share of assessed value in the structure. A low ratio means the value is
    # in the dirt — the classic teardown / underimprovement signature.
    if "improvement_value" in out.columns and has_total:
        total = out["total_value"].where(out["total_value"] > 0)
        out["improvement_ratio"] = out["improvement_value"] / total
        out.loc[~np.isfinite(out["improvement_ratio"]), "improvement_ratio"] = np.nan
    else:
        out["improvement_ratio"] = np.nan

    return out


def peer_median(
    frame: pd.DataFrame,
    value_column: str,
    group_keys: list[str] | None = None,
    min_group_size: int | None = None,
) -> pd.Series:
    """
    Median of ``value_column`` within each parcel's peer group.

    Thin peer groups produce noisy medians, so any group below the size floor
    falls back to a broader grouping and finally to the global median. The
    returned series is aligned to ``frame``'s index.
    """
    keys = [k for k in (group_keys or config.PEER_GROUP_KEYS) if k in frame.columns]
    floor = min_group_size or config.MIN_PEER_GROUP_SIZE

    if value_column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)

    values = pd.to_numeric(frame[value_column], errors="coerce")
    global_median = values.median()

    if not keys:
        return pd.Series(global_median, index=frame.index, dtype=float)

    result = pd.Series(np.nan, index=frame.index, dtype=float)

    # Walk from the most specific grouping to the least, filling only the rows
    # whose group was too thin at the finer level.
    for depth in range(len(keys), 0, -1):
        subset = keys[:depth]
        grouped = values.groupby([frame[k] for k in subset], dropna=False)
        medians = grouped.transform("median")
        sizes = grouped.transform("count")
        eligible = (sizes >= floor) & medians.notna() & result.isna()
        result[eligible] = medians[eligible]

    result = result.fillna(global_median)
    return result
