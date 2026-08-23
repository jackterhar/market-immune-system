"""
City of LA building permits.

Used for one signal: how long a parcel has gone without permit activity.
Dormancy alongside age is a deferred-capex tell; it also separates parcels a
sophisticated owner is actively working from ones nobody has touched.

City of LA only. County parcels outside city limits get a null, which the
scorer treats as a missing component rather than as "no activity".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cre import cache, config
from cre.sources import socrata

# LA permit records carry the assessor parcel number split across three
# columns; the AIN is their zero-padded concatenation (4 + 3 + 3 digits).
_AIN_PARTS = [
    (["assessor_book", "assessorbook", "asr_book"], 4),
    (["assessor_page", "assessorpage", "asr_page"], 3),
    (["assessor_parcel", "assessorparcel", "asr_parcel"], 3),
]

_DATE_CANDIDATES = [
    "issue_date",
    "status_date",
    "permit_issue_date",
    "issued_date",
    "latest_status_date",
]


def _find(columns: set[str], candidates: list[str]) -> str | None:
    lowered = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return None


def build_ain(frame: pd.DataFrame) -> pd.Series:
    """Reassemble the 10-digit AIN from the split book/page/parcel columns."""
    columns = set(frame.columns)
    pieces: list[pd.Series] = []
    for candidates, width in _AIN_PARTS:
        column = _find(columns, candidates)
        if column is None:
            return pd.Series(pd.NA, index=frame.index, dtype="string")
        pieces.append(
            frame[column]
            .astype("string")
            .str.replace(r"\D", "", regex=True)
            .str.zfill(width)
        )
    return pieces[0] + pieces[1] + pieces[2]


def fetch_permit_recency(
    max_rows: int | None = 400_000, use_cache: bool = True
) -> tuple[pd.DataFrame, socrata.SourceResult]:
    """
    Return one row per parcel: AIN, most recent permit date, permit count.

    The full permit history is large and only the recency matters here, so the
    frame is collapsed before caching.
    """
    entry = cache.entry("permit_recency", max_rows=max_rows)
    if use_cache:
        cached = cache.read(entry)
        if cached is not None:
            return cached, socrata.SourceResult(
                frame=cached,
                row_count=len(cached),
                notes=[f"Served from cache ({entry.age_hours():.0f}h old)."],
            )

    result = socrata.fetch(config.LA_CITY_PERMITS, max_rows=max_rows)
    if result.frame.empty:
        return pd.DataFrame(columns=["parcel_id", "last_permit_date", "permit_count"]), result

    frame = result.frame
    ain = build_ain(frame)
    date_column = _find(set(frame.columns), _DATE_CANDIDATES)

    if ain.isna().all() or date_column is None:
        missing = "AIN book/page/parcel columns" if ain.isna().all() else "a permit date column"
        result.errors.append(
            f"Permit data loaded but {missing} could not be found; "
            "permit dormancy scoring is disabled."
        )
        return pd.DataFrame(columns=["parcel_id", "last_permit_date", "permit_count"]), result

    working = pd.DataFrame(
        {
            "parcel_id": ain,
            "permit_date": pd.to_datetime(frame[date_column], errors="coerce", utc=True),
        }
    ).dropna(subset=["parcel_id"])
    working = working[working["parcel_id"].str.len() == 10]

    collapsed = (
        working.groupby("parcel_id", dropna=True)
        .agg(last_permit_date=("permit_date", "max"), permit_count=("permit_date", "size"))
        .reset_index()
    )
    collapsed["last_permit_date"] = collapsed["last_permit_date"].dt.tz_localize(None)

    if use_cache:
        cache.write(entry, collapsed, dataset_id=result.dataset_id)

    return collapsed, result


def attach(parcels: pd.DataFrame, recency: pd.DataFrame) -> pd.DataFrame:
    """Join permit recency onto parcels and derive years since last permit."""
    out = parcels.copy()
    if recency.empty or "parcel_id" not in out.columns:
        out["years_since_permit"] = np.nan
        return out

    keyed = out.copy()
    keyed["_join_key"] = (
        keyed["parcel_id"].astype("string").str.replace(r"\D", "", regex=True).str.zfill(10)
    )
    merged = keyed.merge(
        recency.rename(columns={"parcel_id": "_join_key"}), on="_join_key", how="left"
    )
    merged.index = out.index

    now = pd.Timestamp.now()
    merged["years_since_permit"] = (
        now - pd.to_datetime(merged["last_permit_date"], errors="coerce")
    ).dt.days / 365.25

    return merged.drop(columns=["_join_key"])


# Permit records describe new residential construction across several columns
# whose names vary. Rather than pin one schema, candidates are searched in
# order and the first populated match wins.
_PERMIT_TYPE_CANDIDATES = ["permit_type", "permittype", "type"]
_PERMIT_SUBTYPE_CANDIDATES = [
    "permit_sub_type",
    "permitsubtype",
    "sub_type",
    "use_desc",
    "usedesc",
]
_WORK_TEXT_CANDIDATES = [
    "work_description",
    "workdescription",
    "description",
    "use_desc",
    "project_description",
]
_UNIT_CANDIDATES = [
    "du_changed",
    "dwelling_units",
    "dwellingunits",
    "units",
    "no_of_residential_dwelling_units",
]
_ZIP_CANDIDATES = ["zip_code", "zipcode", "zip", "site_zip"]

# A permit counts as new multifamily when it reads as new construction AND the
# use reads as multi-unit residential. Requiring both keeps out tenant
# improvements to existing apartment buildings, which add no rooftops.
_NEW_CONSTRUCTION_TERMS = ["new", "bldg-new", "new building"]
_MULTIFAMILY_TERMS = [
    "apartment",
    "multi-family",
    "multifamily",
    "condominium",
    "condo",
    "dwelling units",
    "5 or more",
]


def _matches_any(series: pd.Series, terms: list[str]) -> pd.Series:
    lowered = series.astype("string").str.lower().fillna("")
    hit = pd.Series(False, index=series.index)
    for term in terms:
        hit |= lowered.str.contains(term, regex=False, na=False)
    return hit


def fetch_multifamily_permits(
    lookback_years: int | None = None,
    max_rows: int | None = 400_000,
    use_cache: bool = True,
) -> tuple[pd.DataFrame, socrata.SourceResult]:
    """
    Permitted multifamily units by ZIP. City of LA only.

    Forward-looking and correspondingly softer than delivered units: a permit
    may lapse, get amended, or never break ground. Returned separately from
    delivered counts so the two are never conflated.

    Returns an empty frame with explanatory notes whenever the permit schema
    does not expose what this needs — which is a normal outcome, not a failure.
    """
    lookback = lookback_years or config.NEW_MF_LOOKBACK_YEARS
    entry = cache.entry("mf_permits", lookback=lookback, max_rows=max_rows)

    if use_cache:
        cached = cache.read(entry)
        if cached is not None:
            frame = cached.set_index(cached.columns[0])
            return frame, socrata.SourceResult(
                frame=cached,
                row_count=len(cached),
                notes=[f"Served from cache ({entry.age_hours():.0f}h old)."],
            )

    result = socrata.fetch(config.LA_CITY_PERMITS, max_rows=max_rows)
    if result.frame.empty:
        return pd.DataFrame(), result

    frame = result.frame
    columns = set(frame.columns)

    zip_column = _find(columns, _ZIP_CANDIDATES)
    if zip_column is None:
        result.notes.append(
            "Permit records carry no ZIP column, so permitted units cannot be "
            "placed on the map. Delivered units are unaffected."
        )
        return pd.DataFrame(), result

    type_column = _find(columns, _PERMIT_TYPE_CANDIDATES)
    subtype_column = _find(columns, _PERMIT_SUBTYPE_CANDIDATES)
    work_column = _find(columns, _WORK_TEXT_CANDIDATES)

    if type_column is None and work_column is None:
        result.notes.append(
            "No permit type or description column matched, so new multifamily "
            "permits cannot be isolated. Delivered units are unaffected."
        )
        return pd.DataFrame(), result

    is_new = (
        _matches_any(frame[type_column], _NEW_CONSTRUCTION_TERMS)
        if type_column
        else pd.Series(True, index=frame.index)
    )
    multifamily = pd.Series(False, index=frame.index)
    for column in (subtype_column, work_column):
        if column:
            multifamily |= _matches_any(frame[column], _MULTIFAMILY_TERMS)

    date_column = _find(columns, _DATE_CANDIDATES)
    recent = pd.Series(True, index=frame.index)
    if date_column:
        issued = pd.to_datetime(frame[date_column], errors="coerce", utc=True)
        cutoff = pd.Timestamp.now(tz="UTC") - pd.DateOffset(years=lookback)
        recent = issued >= cutoff

    selected = frame[is_new & multifamily & recent]
    if selected.empty:
        result.notes.append(
            "No permits matched new multifamily construction in the window."
        )
        return pd.DataFrame(), result

    unit_column = _find(columns, _UNIT_CANDIDATES)
    working = pd.DataFrame(
        {"zip": selected[zip_column].astype("string").str.strip().str[:5]}
    )
    if unit_column:
        working["units"] = pd.to_numeric(selected[unit_column], errors="coerce")
    else:
        # Without a unit count the project count is still meaningful; unit
        # totals are left null rather than guessed at.
        working["units"] = np.nan
        result.notes.append(
            "Permit records expose no dwelling-unit count, so only project "
            "counts are reported for permitted supply."
        )

    working = working[working["zip"].str.len() == 5]
    summary = working.groupby("zip").agg(
        units_permitted=("units", "sum"), permit_projects=("units", "size")
    )
    if not unit_column:
        summary["units_permitted"] = np.nan

    result.notes.append(
        f"{len(selected):,} new multifamily permits across {len(summary):,} ZIPs."
    )

    if use_cache:
        cache.write(entry, summary.reset_index(), dataset_id=result.dataset_id)

    return summary, result
