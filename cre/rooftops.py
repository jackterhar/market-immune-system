"""
Rooftop growth: where new multifamily is landing.

Retail follows rooftops. A tired shopping center on a corridor that is adding
several hundred apartments has an incoming customer base the current owner is
not serving; the same center on a corridor adding nothing does not. This module
measures that at area level and joins the answer back onto commercial parcels
so the other lenses can use it.

Two sources feed it, with very different reliability:

* **Delivered** units come from the assessor roll — a residential parcel with
  five or more units and a recent year built. Countywide, no permit data
  needed, and about as solid as this gets.
* **Permitted** units come from City of LA building permits, which cover only
  the city and depend on permit fields this project cannot verify offline.
  Treated as an enrichment: when it resolves the tab shows a forward-looking
  column, and when it does not the delivered figures stand alone.

Delivered and permitted are deliberately never summed. They measure different
moments — one is built, the other may never break ground — and adding them
would imply a confidence neither supports.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from cre import config


@dataclass
class RooftopResult:
    """Area-level growth statistics plus provenance."""

    areas: pd.DataFrame
    area_key: str = "situs_zip"
    has_permits: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.areas.empty


def _area_series(frame: pd.DataFrame, area_key: str) -> pd.Series:
    """Normalized area labels, blanking empties so they do not form a group."""
    if area_key not in frame.columns:
        return pd.Series(pd.NA, index=frame.index, dtype="object")
    values = frame[area_key].astype("string").str.strip()
    return values.where(values.str.len() > 0)


def resolve_area_key(frame: pd.DataFrame) -> str:
    """Pick the finest area geography actually populated in this frame."""
    for key in config.AREA_KEYS:
        if key in frame.columns and _area_series(frame, key).notna().any():
            return key
    return config.AREA_KEYS[-1]


def is_new_multifamily(
    frame: pd.DataFrame,
    roll_year: int | None = None,
    lookback_years: int | None = None,
    min_units: int | None = None,
) -> pd.Series:
    """
    Flag recently delivered multifamily parcels.

    Uses year built rather than the effective year: a renovation advances the
    effective year without adding a single rooftop.
    """
    year = roll_year or config.CURRENT_ROLL_YEAR_FALLBACK
    lookback = lookback_years or config.NEW_MF_LOOKBACK_YEARS
    units_floor = min_units or config.MIN_MF_UNITS

    mask = pd.Series(True, index=frame.index)

    if "units" in frame.columns:
        mask &= pd.to_numeric(frame["units"], errors="coerce") >= units_floor
    else:
        return pd.Series(False, index=frame.index)

    if "year_built" in frame.columns:
        built = pd.to_numeric(frame["year_built"], errors="coerce")
        mask &= built.between(year - lookback, year)
    else:
        return pd.Series(False, index=frame.index)

    if "general_use" in frame.columns:
        mask &= (
            frame["general_use"].astype("string").str.contains(
                "residential", case=False, na=False
            )
        )

    return mask.fillna(False)


def summarize_areas(
    multifamily: pd.DataFrame,
    area_key: str | None = None,
    roll_year: int | None = None,
    lookback_years: int | None = None,
    permits_by_area: pd.DataFrame | None = None,
) -> RooftopResult:
    """
    Aggregate delivered (and optionally permitted) units by area.

    ``multifamily`` should already be narrowed to new multifamily parcels.
    Returns one row per area, ranked by units delivered.
    """
    lookback = lookback_years or config.NEW_MF_LOOKBACK_YEARS
    notes: list[str] = []

    if multifamily.empty:
        return RooftopResult(
            areas=pd.DataFrame(),
            notes=[
                "No recently built multifamily found. Either the parcel set "
                "excludes residential, or the unit and year columns did not "
                "resolve — check the Diagnostics tab."
            ],
        )

    key = area_key or resolve_area_key(multifamily)
    working = multifamily.copy()
    working["_area"] = _area_series(working, key)
    working = working[working["_area"].notna()]

    if working.empty:
        return RooftopResult(
            areas=pd.DataFrame(),
            area_key=key,
            notes=[f"No parcels carry a usable {key} value."],
        )

    units = pd.to_numeric(working.get("units"), errors="coerce")
    working["_units"] = units

    grouped = working.groupby("_area", dropna=True)
    areas = pd.DataFrame(
        {
            "units_delivered": grouped["_units"].sum(min_count=1),
            "projects": grouped.size(),
            "median_year_built": grouped["year_built"].median()
            if "year_built" in working.columns
            else np.nan,
            "largest_project": grouped["_units"].max(),
        }
    )

    if "situs_city" in working.columns:
        # Modal city for the area, so ZIP rows stay labelled.
        areas["city"] = grouped["situs_city"].agg(
            lambda values: values.mode().iloc[0] if not values.mode().empty else pd.NA
        )

    for coordinate in ("lat", "lon"):
        if coordinate in working.columns:
            areas[coordinate] = grouped[coordinate].median()

    areas["units_per_year"] = areas["units_delivered"] / lookback

    has_permits = False
    if permits_by_area is not None and not permits_by_area.empty:
        permit_frame = permits_by_area.copy()
        permit_frame.index = permit_frame.index.astype("string").str.strip()
        areas = areas.join(permit_frame, how="left")
        # Deliberately not added to units_delivered: permitted is not built.
        for column in ("units_permitted", "permit_projects"):
            if column in areas.columns:
                areas[column] = areas[column].fillna(0)
                has_permits = True
        if has_permits:
            notes.append(
                "Permitted units are shown alongside delivered ones, never "
                "added to them — a permit is not a building."
            )

    areas = areas.sort_values("units_delivered", ascending=False)
    areas["growth_rank"] = range(1, len(areas) + 1)
    areas.index.name = key

    notes.append(
        f"{int(areas['projects'].sum()):,} multifamily projects delivered across "
        f"{len(areas):,} areas in the last {lookback} years."
    )

    return RooftopResult(
        areas=areas, area_key=key, has_permits=has_permits, notes=notes
    )


def attach(parcels: pd.DataFrame, rooftops: RooftopResult) -> pd.DataFrame:
    """
    Add area growth columns to a parcel frame.

    Every parcel receives the delivered-unit total and percentile of the area
    it sits in, so any lens can filter or sort on incoming rooftops.
    """
    out = parcels.copy()
    out["area_new_units"] = np.nan
    out["area_units_permitted"] = np.nan
    out["area_growth_percentile"] = np.nan

    if not rooftops.ok or rooftops.area_key not in out.columns:
        return out

    areas = rooftops.areas
    keys = _area_series(out, rooftops.area_key)

    delivered = areas["units_delivered"]
    out["area_new_units"] = keys.map(delivered).astype(float).fillna(0.0)

    if "units_permitted" in areas.columns:
        out["area_units_permitted"] = (
            keys.map(areas["units_permitted"]).astype(float).fillna(0.0)
        )

    # Percentile across areas, not across parcels: a ZIP with one shopping
    # center and a ZIP with fifty should rank on their growth, not their
    # parcel count.
    area_percentile = delivered.rank(pct=True)
    # An area absent from the summary genuinely has no new multifamily — the
    # underlying pull is countywide — so it belongs at the bottom rather than
    # as a null. This branch is only reached when the pull succeeded; a failed
    # one returns all-null above, which is a different thing and must stay
    # distinguishable from "no growth here".
    out["area_growth_percentile"] = (
        keys.map(area_percentile).astype(float).fillna(0.0)
    )

    return out
