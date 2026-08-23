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
