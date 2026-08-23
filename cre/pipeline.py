"""
Orchestration: raw sources in, scored parcels out.

Kept separate from the Streamlit layer so the whole pipeline can be run from a
script or a notebook, and so the UI never has to know how sources fail.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

from cre import config, scoring, transform
from cre.sources import assessor, mls, permits, zoning
from cre.sources.socrata import SourceResult


@dataclass
class PipelineResult:
    """Scored parcels plus per-source health for the diagnostics tab."""

    parcels: pd.DataFrame
    schema: transform.SchemaReport | None = None
    sources: dict[str, SourceResult] = field(default_factory=dict)
    acquisition: scoring.ScoreResult | None = None
    development: scoring.ScoreResult | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.parcels.empty

    def source_table(self) -> pd.DataFrame:
        rows = []
        for name, result in self.sources.items():
            rows.append(
                {
                    "source": name,
                    "status": result.status,
                    "rows": result.row_count,
                    "dataset": result.dataset_id or "—",
                    "detail": "; ".join(result.errors or result.notes)[:300] or "—",
                }
            )
        return pd.DataFrame(rows)


def build_use_types(
    include_commercial: bool = True,
    include_vacant: bool = True,
    include_income_residential: bool = False,
) -> list[str]:
    """Assemble the assessor general-use filter from UI toggles."""
    use_types: list[str] = []
    if include_commercial:
        use_types.extend(config.CRE_GENERAL_USE_TYPES)
    if include_vacant:
        use_types.extend(config.VACANT_USE_TYPES)
    if include_income_residential:
        use_types.extend(config.INCOME_RESIDENTIAL_USE_TYPES)
    # Preserve order while removing duplicates.
    return list(dict.fromkeys(use_types))


def run(
    use_types: list[str] | None = None,
    cities: list[str] | None = None,
    max_rows: int | None = None,
    with_permits: bool = True,
    with_zoning: bool = True,
    with_mls: bool = False,
    use_cache: bool = True,
    progress: Callable[[str, int], None] | None = None,
) -> PipelineResult:
    """
    Run the full screen.

    Every enrichment is optional and every one of them fails soft: a dead
    permit feed costs the dormancy component, not the run. Only the assessor
    roll is load-bearing.
    """

    def report(stage: str, count: int = 0) -> None:
        if progress is not None:
            progress(stage, count)

    report("Loading assessor parcel roll")
    parcels, schema, assessor_result = assessor.fetch_parcels(
        use_types=use_types,
        cities=cities,
        max_rows=max_rows,
        use_cache=use_cache,
        progress=lambda n, _ds: report("Loading assessor parcel roll", n),
    )
    result = PipelineResult(parcels=parcels, schema=schema)
    result.sources["assessor"] = assessor_result

    if parcels.empty:
        result.notes.append(
            "No parcels loaded — every downstream feature is unavailable. "
            "See the source table for why."
        )
        return result

    if with_permits:
        report("Loading building permits")
        recency, permit_result = permits.fetch_permit_recency(use_cache=use_cache)
        result.sources["permits"] = permit_result
        if not recency.empty:
            result.parcels = permits.attach(result.parcels, recency)
        else:
            result.parcels["years_since_permit"] = pd.NA

    if with_zoning:
        report("Joining zoning and transit overlays")
        result.parcels, zoning_result = zoning.enrich(
            result.parcels,
            use_cache=use_cache,
            progress=lambda n: report("Joining zoning and transit overlays", n),
        )
        result.sources["zoning"] = zoning_result

    if with_mls:
        report("Fetching MLS listings")
        listings, mls_result = mls.fetch_listings()
        result.sources["mls"] = mls_result
        if not listings.empty:
            result.parcels = mls.attach(result.parcels, listings)

    report("Scoring")
    result.acquisition = scoring.acquisition_score(result.parcels)
    result.development = scoring.development_score(result.parcels)
    result.parcels["acquisition_score"] = result.acquisition.scores
    result.parcels["development_score"] = result.development.scores

    gap, max_far, _ = scoring.far_gap(result.parcels)
    result.parcels["far_headroom"] = gap
    result.parcels["assumed_max_far"] = max_far
    if "lot_sqft" in result.parcels.columns:
        result.parcels["unused_buildable_sqft"] = (
            (max_far - result.parcels["built_far"].fillna(0))
            * result.parcels["lot_sqft"]
        ).clip(lower=0)

    result.notes.extend(result.acquisition.notes)
    result.notes.extend(result.development.notes)
    return result


def rehab_candidates(
    parcels: pd.DataFrame,
    min_lot_sqft: float | None = None,
    min_building_sqft: float | None = None,
    max_building_sqft: float | None = None,
) -> tuple[pd.DataFrame, scoring.ScoreResult]:
    """
    Filter to shopping centers and score them for repositioning.

    Scoring happens *after* filtering on purpose. The components are percentile
    ranks, so they only mean something relative to a comparable set: a 90 here
    should mean "in the top decile of retail centers", not "of all parcels in
    LA County". Scoring first and filtering second would produce the latter.
    """
    candidates = scoring.shopping_center_candidates(
        parcels,
        min_lot_sqft=min_lot_sqft,
        min_building_sqft=min_building_sqft,
        max_building_sqft=max_building_sqft,
    ).copy()

    if candidates.empty:
        return candidates, scoring.ScoreResult(
            scores=pd.Series(dtype=float),
            components=pd.DataFrame(),
            notes=["No parcels matched the shopping center filters."],
        )

    result = scoring.rehab_score(candidates)
    candidates["rehab_score"] = result.scores

    flags = scoring.rehab_flags(candidates)
    for column in flags.columns:
        candidates[column] = flags[column]

    return candidates, result


REHAB_DISPLAY_COLUMNS = [
    "parcel_id",
    "situs_address",
    "situs_city",
    "specific_use",
    "rehab_score",
    "building_sqft",
    "lot_sqft",
    "built_far",
    "year_built",
    "building_age",
    "renovation_gap",
    "years_since_improvement",
    "improvement_per_sqft",
    "tenure_years",
    "total_value",
    "never_renovated",
    "excess_parking",
    "below_peer_condition",
    "years_since_permit",
    "mls_listed",
]


def rehab_display_frame(candidates: pd.DataFrame) -> pd.DataFrame:
    columns = [c for c in REHAB_DISPLAY_COLUMNS if c in candidates.columns]
    return candidates[columns]


DISPLAY_COLUMNS = [
    "parcel_id",
    "situs_address",
    "situs_city",
    "specific_use",
    "acquisition_score",
    "development_score",
    "tenure_years",
    "building_age",
    "lot_sqft",
    "building_sqft",
    "built_far",
    "assumed_max_far",
    "far_headroom",
    "unused_buildable_sqft",
    "total_value",
    "value_per_sqft",
    "improvement_ratio",
    "zone",
    "toc_tier",
    "years_since_permit",
    "mls_listed",
]


def display_frame(parcels: pd.DataFrame) -> pd.DataFrame:
    """Project to the columns worth showing, in a sensible order."""
    columns = [c for c in DISPLAY_COLUMNS if c in parcels.columns]
    return parcels[columns]
