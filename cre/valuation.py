"""
Market value estimation from recent transfers.

Assessed value is not market value. Proposition 13 freezes a parcel's
assessment at its base year and caps growth at 2% a year, so a center held
since 1975 carries an assessment far below what it would trade for. Filtering a
purchase-price budget against assessed value would therefore select almost
entirely on how long someone has owned a property, not on what it costs.

Prop 13 also supplies the remedy. A change of ownership resets the assessment
to the purchase price, so **parcels with a recent base year are a record of
recent sale prices**. Pooling those by submarket and use type gives a price per
square foot that can be applied to parcels that have not traded.

One refinement matters. A change of ownership reassesses the whole parcel;
new construction reassesses only the improvement. So a recent *land* base year
indicates a sale, while a recent improvement base year on top of an old land
base year usually indicates construction or renovation. The comp set is drawn
on the land base year for that reason.

What this does not do: adjust for market movement since the comp traded. A
2021 sale is carried at its 2021 price plus the Prop 13 inflation factor, not
marked to today. Comp vintage and count travel with every estimate so the
uncertainty stays visible.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from cre import config

# How far back to draw comparable transfers. Too short and there are no comps;
# too long and prices go stale.
TRANSFER_LOOKBACK_YEARS = 5

# Minimum transfers behind a benchmark before it is trusted at that grouping.
MIN_COMPS = 5

# Guards against nonsense benchmarks from data errors.
MIN_PLAUSIBLE_PSF = 5.0
MAX_PLAUSIBLE_PSF = 3_000.0


@dataclass
class ValuationResult:
    """Estimates plus the evidence behind them."""

    estimate: pd.Series
    comp_count: pd.Series
    basis: pd.Series
    notes: list[str] = field(default_factory=list)

    @property
    def coverage(self) -> float:
        """Share of parcels that received an estimate."""
        if len(self.estimate) == 0:
            return 0.0
        return float(self.estimate.notna().mean())


def recent_transfer_mask(
    frame: pd.DataFrame,
    roll_year: int | None = None,
    lookback_years: int = TRANSFER_LOOKBACK_YEARS,
) -> pd.Series:
    """
    Flag parcels whose assessment reflects a recent arm's-length sale.

    Uses the land base year: a sale reassesses the land, construction does not.
    """
    year = roll_year or config.CURRENT_ROLL_YEAR_FALLBACK
    if "land_base_year" in frame.columns:
        base = pd.to_numeric(frame["land_base_year"], errors="coerce")
    elif "base_year" in frame.columns:
        base = pd.to_numeric(frame["base_year"], errors="coerce")
    else:
        return pd.Series(False, index=frame.index)
    return (base >= year - lookback_years) & (base <= year)


def _comp_median(
    frame: pd.DataFrame,
    values: pd.Series,
    mask: pd.Series,
    keys: list[str],
    min_comps: int,
) -> tuple[pd.Series, pd.Series]:
    """
    Median of ``values`` over comps only, broadcast to every row in the group.

    Walks key prefixes from most specific to least, filling each row at the
    first level with enough comps behind it. Returns (median, comp count).
    Keys must be ordered broadest-first so the fallback actually widens — see
    the note on config.LOCATION_PEER_KEYS.
    """
    comp_values = values.where(mask)
    median = pd.Series(np.nan, index=frame.index, dtype=float)
    counts = pd.Series(0, index=frame.index, dtype=int)

    for depth in range(len(keys), 0, -1):
        subset = keys[:depth]
        grouped = comp_values.groupby([frame[k] for k in subset], dropna=False)
        level_median = grouped.transform("median")
        level_count = grouped.transform("count")
        eligible = (level_count >= min_comps) & level_median.notna() & median.isna()
        median[eligible] = level_median[eligible]
        counts[eligible] = level_count[eligible].astype(int)

    # Final fallback: the whole comp pool.
    global_median = comp_values.median()
    global_count = int(comp_values.notna().sum())
    if pd.notna(global_median) and global_count >= min_comps:
        unfilled = median.isna()
        median[unfilled] = global_median
        counts[unfilled] = global_count

    return median, counts


def estimate_market_value(
    frame: pd.DataFrame,
    roll_year: int | None = None,
    lookback_years: int = TRANSFER_LOOKBACK_YEARS,
    group_keys: list[str] | None = None,
    min_comps: int = MIN_COMPS,
) -> ValuationResult:
    """
    Estimate what each parcel would sell for, from recent nearby transfers.

    Improved property is valued on price per building square foot, which is how
    commercial buildings actually trade. Parcels with no building — or so
    little of one that the land dominates — are valued on price per lot square
    foot instead. The chosen basis is reported per parcel.
    """
    index = frame.index
    notes: list[str] = []
    keys = [k for k in (group_keys or config.PEER_GROUP_KEYS) if k in frame.columns]

    empty = pd.Series(np.nan, index=index, dtype=float)
    if "total_value" not in frame.columns:
        return ValuationResult(
            estimate=empty,
            comp_count=pd.Series(0, index=index, dtype=int),
            basis=pd.Series(pd.NA, index=index, dtype="object"),
            notes=["No assessed value column, so market value cannot be estimated."],
        )

    mask = recent_transfer_mask(frame, roll_year, lookback_years)
    comp_total = int(mask.sum())
    if comp_total < min_comps:
        return ValuationResult(
            estimate=empty,
            comp_count=pd.Series(0, index=index, dtype=int),
            basis=pd.Series(pd.NA, index=index, dtype="object"),
            notes=[
                f"Only {comp_total} parcels transferred in the last "
                f"{lookback_years} years — too few to build price benchmarks. "
                "Widen the loaded parcel set or the lookback window."
            ],
        )

    value = pd.to_numeric(frame["total_value"], errors="coerce")
    building = pd.to_numeric(
        frame.get("building_sqft", pd.Series(np.nan, index=index)), errors="coerce"
    )
    lot = pd.to_numeric(
        frame.get("lot_sqft", pd.Series(np.nan, index=index)), errors="coerce"
    )

    building_psf = (value / building).replace([np.inf, -np.inf], np.nan)
    lot_psf = (value / lot).replace([np.inf, -np.inf], np.nan)

    # Discard implausible per-foot figures before they poison a median.
    building_psf = building_psf.where(
        building_psf.between(MIN_PLAUSIBLE_PSF, MAX_PLAUSIBLE_PSF)
    )
    lot_psf = lot_psf.where(lot_psf.between(MIN_PLAUSIBLE_PSF, MAX_PLAUSIBLE_PSF))

    building_benchmark, building_comps = _comp_median(
        frame, building_psf, mask, keys, min_comps
    )
    lot_benchmark, lot_comps = _comp_median(frame, lot_psf, mask, keys, min_comps)

    from_building = building * building_benchmark
    from_lot = lot * lot_benchmark

    # Improved property trades on building area; land-dominant parcels do not.
    use_building = building.notna() & from_building.notna()
    if "built_far" in frame.columns:
        coverage = pd.to_numeric(frame["built_far"], errors="coerce")
        use_building &= coverage.isna() | (coverage >= 0.05)

    estimate = pd.Series(np.nan, index=index, dtype=float)
    basis = pd.Series(pd.NA, index=index, dtype="object")
    comp_count = pd.Series(0, index=index, dtype=int)

    estimate[use_building] = from_building[use_building]
    basis[use_building] = "building sqft"
    comp_count[use_building] = building_comps[use_building]

    use_lot = estimate.isna() & from_lot.notna()
    estimate[use_lot] = from_lot[use_lot]
    basis[use_lot] = "lot sqft"
    comp_count[use_lot] = lot_comps[use_lot]

    notes.append(
        f"Benchmarks built from {comp_total:,} parcels that changed hands in the "
        f"last {lookback_years} years, grouped by {' then '.join(keys) or 'the whole set'}."
    )
    notes.append(
        "Comps are carried at their transfer-year price and are not marked to "
        "today's market."
    )

    return ValuationResult(
        estimate=estimate, comp_count=comp_count, basis=basis, notes=notes
    )


def attach(
    frame: pd.DataFrame,
    roll_year: int | None = None,
    lookback_years: int = TRANSFER_LOOKBACK_YEARS,
) -> tuple[pd.DataFrame, ValuationResult]:
    """Add estimated_value, value_comp_count and value_basis columns."""
    result = estimate_market_value(
        frame, roll_year=roll_year, lookback_years=lookback_years
    )
    out = frame.copy()
    out["estimated_value"] = result.estimate
    out["value_comp_count"] = result.comp_count
    out["value_basis"] = result.basis
    if "total_value" in out.columns:
        ratio = out["estimated_value"] / pd.to_numeric(
            out["total_value"], errors="coerce"
        ).replace(0, np.nan)
        out["assessed_to_estimate_ratio"] = ratio.replace([np.inf, -np.inf], np.nan)
    return out, result
