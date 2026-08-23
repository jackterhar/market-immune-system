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
