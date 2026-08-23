"""
LA County assessor secured property roll.

The roll is the spine of this app: it is the only free source that covers all
~2.4M parcels in all 88 cities with use type, assessed values, Prop 13 base
years, building area and coordinates.

Filtering happens server-side via SoQL so a countywide commercial pull moves
~200k rows instead of 2.4M. Because the column carrying use type is not named
consistently across roll years, the filter column is resolved from a live
schema probe before the query is built.
"""

from __future__ import annotations

from typing import Any, Callable

import pandas as pd

from cre import cache, config, transform
from cre.sources import socrata


def _quote_list(values: list[str]) -> str:
    escaped = [v.replace("'", "''") for v in values]
    return ", ".join(f"'{v}'" for v in escaped)


def build_where(
    use_column: str,
    use_types: list[str],
    city_column: str | None = None,
    cities: list[str] | None = None,
) -> str:
    """Compose the SoQL filter. Values are quoted; column names come from the probe."""
    clauses = [f"{use_column} in ({_quote_list(use_types)})"]
    if city_column and cities:
        clauses.append(f"{city_column} in ({_quote_list(cities)})")
    return " AND ".join(clauses)


def resolve_probe_columns(columns: list[str]) -> dict[str, str]:
    """Map logical fields onto real column names using a probed column list."""
    frame = pd.DataFrame(columns=columns)
    return transform.resolve_columns(frame).resolved


def fetch_parcels(
    use_types: list[str] | None = None,
    cities: list[str] | None = None,
    max_rows: int | None = None,
    use_cache: bool = True,
    progress: Callable[[int, str], None] | None = None,
) -> tuple[pd.DataFrame, transform.SchemaReport | None, socrata.SourceResult]:
    """
    Pull, normalize and enrich the parcel roll.

    Returns (parcels, schema_report, raw_result). The raw result carries the
    diagnostics — resolved dataset ID, discovery log, errors — that the
    dashboard shows when something comes back empty.
    """
    use_types = use_types or config.CRE_GENERAL_USE_TYPES
    entry = cache.entry(
        "assessor",
        use_types=sorted(use_types),
        cities=sorted(cities) if cities else None,
        max_rows=max_rows,
    )

    if use_cache:
        cached = cache.read(entry)
        if cached is not None:
            report = transform.resolve_columns(cached) if not cached.empty else None
            result = socrata.SourceResult(
                frame=cached,
                row_count=len(cached),
                dataset_id=entry.meta().get("dataset_id"),
                domain=config.LA_COUNTY_DOMAIN,
                notes=[f"Served from cache ({entry.age_hours():.0f}h old)."],
            )
            enriched = transform.add_derived_metrics(cached)
            return enriched, report, result

    dataset_id, discovery_log = socrata.discover_dataset_id(config.ASSESSOR_PARCELS)
    if dataset_id is None:
        failed = socrata.SourceResult(
            frame=pd.DataFrame(),
            domain=config.LA_COUNTY_DOMAIN,
            notes=discovery_log,
            errors=[
                "Could not locate the assessor parcel dataset on "
                f"{config.LA_COUNTY_DOMAIN}. Check connectivity, then set an "
                "explicit dataset ID in cre/config.py:ASSESSOR_PARCELS."
            ],
        )
        return pd.DataFrame(), None, failed

    try:
        columns = socrata.probe_schema(config.LA_COUNTY_DOMAIN, dataset_id)
    except socrata.SocrataError as exc:
        return (
            pd.DataFrame(),
            None,
            socrata.SourceResult(
                frame=pd.DataFrame(),
                dataset_id=dataset_id,
                domain=config.LA_COUNTY_DOMAIN,
                notes=discovery_log,
                errors=[f"Schema probe failed: {exc}"],
            ),
        )

    resolved = resolve_probe_columns(columns)
    use_column = resolved.get("general_use")
    if use_column is None:
        return (
            pd.DataFrame(),
            None,
            socrata.SourceResult(
                frame=pd.DataFrame(),
                dataset_id=dataset_id,
                domain=config.LA_COUNTY_DOMAIN,
                notes=discovery_log + [f"Columns seen: {', '.join(columns[:40])}"],
                errors=[
                    "No use-type column matched. Add the real column name to "
                    "COLUMN_ALIASES['general_use'] in cre/config.py."
                ],
            ),
        )

    where = build_where(use_column, use_types, resolved.get("situs_city"), cities)
    result = socrata.fetch(
        config.ASSESSOR_PARCELS,
        where=where,
        max_rows=max_rows,
        dataset_id=dataset_id,
        progress=progress,
    )
    result.notes = discovery_log + result.notes

    if result.frame.empty:
        return pd.DataFrame(), None, result

    normalized, report = transform.normalize(result.frame)
    roll_year = None
    if "roll_year" in normalized.columns:
        years = normalized["roll_year"].dropna()
        if not years.empty:
            roll_year = int(years.mode().iloc[0])

    enriched = transform.add_derived_metrics(normalized, roll_year=roll_year)

    if use_cache:
        cache.write(entry, normalized, dataset_id=dataset_id, roll_year=roll_year)

    return enriched, report, result


def _numeric_where(
    resolved: dict[str, str],
    use_column: str,
    roll_year: int,
    lookback_years: int,
    min_units: int,
) -> str | None:
    """SoQL filtering multifamily server-side, or None if columns are missing."""
    units_column = resolved.get("units")
    year_column = resolved.get("year_built")
    if not units_column or not year_column:
        return None
    return (
        f"{use_column} = 'Residential' "
        f"AND {units_column} >= {min_units} "
        f"AND {year_column} >= {roll_year - lookback_years}"
    )


def fetch_new_multifamily(
    roll_year: int | None = None,
    lookback_years: int | None = None,
    min_units: int | None = None,
    max_rows: int | None = None,
    use_cache: bool = True,
    progress: Callable[[int, str], None] | None = None,
) -> tuple[pd.DataFrame, socrata.SourceResult]:
    """
    Pull recently built multifamily parcels, countywide.

    Deliberately a separate, narrow query rather than part of the main load.
    Residential is ~1.8M of the county's 2.4M parcels; filtering to five-plus
    units built in the last few years cuts that to tens of thousands, which
    moves in seconds instead of minutes.

    Socrata columns are sometimes typed as text, in which case a numeric
    comparison is rejected. When that happens the query falls back to a
    use-type filter with a row cap and the numeric narrowing happens locally.
    Which path ran is reported in the result notes.
    """
    from cre import rooftops

    year = roll_year or config.CURRENT_ROLL_YEAR_FALLBACK
    lookback = lookback_years or config.NEW_MF_LOOKBACK_YEARS
    units_floor = min_units or config.MIN_MF_UNITS
    cap = max_rows or config.NEW_MF_MAX_ROWS

    entry = cache.entry(
        "new_multifamily", year=year, lookback=lookback, units=units_floor, cap=cap
    )
    if use_cache:
        cached = cache.read(entry)
        if cached is not None:
            enriched = transform.add_derived_metrics(cached, roll_year=year)
            return enriched, socrata.SourceResult(
                frame=cached,
                row_count=len(cached),
                dataset_id=entry.meta().get("dataset_id"),
                domain=config.LA_COUNTY_DOMAIN,
                notes=[f"Served from cache ({entry.age_hours():.0f}h old)."],
            )

    dataset_id, discovery_log = socrata.discover_dataset_id(config.ASSESSOR_PARCELS)
    if dataset_id is None:
        return pd.DataFrame(), socrata.SourceResult(
            frame=pd.DataFrame(),
            domain=config.LA_COUNTY_DOMAIN,
            notes=discovery_log,
            errors=["Could not resolve the assessor dataset for multifamily."],
        )

    try:
        columns = socrata.probe_schema(config.LA_COUNTY_DOMAIN, dataset_id)
    except socrata.SocrataError as exc:
        return pd.DataFrame(), socrata.SourceResult(
            frame=pd.DataFrame(),
            dataset_id=dataset_id,
            domain=config.LA_COUNTY_DOMAIN,
            errors=[f"Schema probe failed: {exc}"],
        )

    resolved = resolve_probe_columns(columns)
    use_column = resolved.get("general_use")
    if use_column is None:
        return pd.DataFrame(), socrata.SourceResult(
            frame=pd.DataFrame(),
            dataset_id=dataset_id,
            domain=config.LA_COUNTY_DOMAIN,
            errors=["No use-type column matched; multifamily cannot be isolated."],
        )

    result: socrata.SourceResult | None = None
    where = _numeric_where(resolved, use_column, year, lookback, units_floor)

    if where is not None:
        result = socrata.fetch(
            config.ASSESSOR_PARCELS,
            where=where,
            max_rows=cap,
            dataset_id=dataset_id,
            progress=progress,
        )
        if result.errors and result.frame.empty:
            result.notes.append(
                "Server-side numeric filter was rejected — the unit or year "
                "column is stored as text. Retrying with a broader query."
            )
            where = None

    if where is None:
        result = socrata.fetch(
            config.ASSESSOR_PARCELS,
            where=f"{use_column} = 'Residential'",
            max_rows=cap,
            dataset_id=dataset_id,
            progress=progress,
        )
        result.notes.append(
            f"Filtered locally to {units_floor}+ units built since "
            f"{year - lookback}. A row cap was applied, so coverage may be "
            "partial in the largest cities."
        )

    assert result is not None
    result.notes = discovery_log + result.notes
    if result.frame.empty:
        return pd.DataFrame(), result

    normalized, _ = transform.normalize(result.frame)
    enriched = transform.add_derived_metrics(normalized, roll_year=year)

    mask = rooftops.is_new_multifamily(
        enriched, roll_year=year, lookback_years=lookback, min_units=units_floor
    )
    enriched = enriched[mask]
    result.row_count = len(enriched)
    result.notes.append(f"{len(enriched):,} new multifamily parcels retained.")

    if use_cache and not enriched.empty:
        cache.write(entry, normalized[mask], dataset_id=dataset_id, roll_year=year)

    return enriched, result
