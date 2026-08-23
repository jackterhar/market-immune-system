"""
Opportunity scoring.

Two independent lenses over the same parcel spine:

* **Acquisition** — finds long-tenured ownership. See the note on
  ``config.AcquisitionWeights``: a low Prop 13 assessment is evidence of a
  distant base year, not of a discount. What it reliably indicates is an owner
  who has held for decades, which correlates with below-market in-place rents
  and with generational transfer events.

* **Development** — finds underbuilt land. Where LA City zoning resolves, the
  gap is measured against permitted FAR; everywhere else it is measured
  against what comparable neighbors actually built, which needs no zoning data
  and works across all 88 cities in the county.

Both scores are 0-100 percentile-relative *within the loaded parcel set*. They
rank candidates against each other; they are not absolute quality grades, and
a 90 in a thin filter is not a 90 countywide.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from cre import config, transform

NEUTRAL = 0.5


def percentile_rank(series: pd.Series, invert: bool = False) -> pd.Series:
    """
    Rank to [0, 1]. NaN stays NaN so callers decide how to treat it.

    ``invert`` flips the direction, for metrics where lower is more
    interesting (a low improvement ratio, a low basis relative to peers).
    """
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().sum() == 0:
        return pd.Series(np.nan, index=series.index, dtype=float)
    ranked = values.rank(pct=True, na_option="keep")
    return (1.0 - ranked) if invert else ranked


def parse_height_district(zone_string: str | None) -> str | None:
    """
    Pull the height district out of an LA zone string.

    Zone strings look like ``[Q]C2-1VL-CDO`` or ``(T)(Q)C4-2D-SN``: prefixes in
    brackets, zone class, height district, then overlay suffixes. The height
    district is the token after the first hyphen. A trailing ``D`` marks a
    development limitation on top of the base district and is stripped.
    """
    if not zone_string or not isinstance(zone_string, str):
        return None
    cleaned = re.sub(r"[\[\(][^\]\)]*[\]\)]", "", zone_string).strip()
    parts = [p for p in cleaned.split("-") if p]
    if len(parts) < 2:
        return None

    token = parts[1].upper()
    if token in config.HEIGHT_DISTRICT_FAR:
        return token
    if token.endswith("D") and token[:-1] in config.HEIGHT_DISTRICT_FAR:
        return token[:-1]
    match = re.match(r"^(\d)", token)
    if match and match.group(1) in config.HEIGHT_DISTRICT_FAR:
        return match.group(1)
    return None


def max_far_for_zone(zone_string: str | None) -> float | None:
    """Permitted FAR implied by a zone string, or None if it cannot be read."""
    district = parse_height_district(zone_string)
    if district is None:
        return None
    return config.HEIGHT_DISTRICT_FAR.get(district)


def is_commercial_zone(zone_string: str | None) -> bool:
    """
    True for commercial and manufacturing zone classes.

    These are the parcels reachable by the state and city programs that allow
    housing on commercially zoned land (AB 2011, ED1, the CHIP corridor
    rezoning). Indicative only — actual eligibility carries exclusions this
    cannot see.
    """
    if not zone_string or not isinstance(zone_string, str):
        return False
    cleaned = re.sub(r"[\[\(][^\]\)]*[\]\)]", "", str(zone_string)).strip().upper()
    zone_class = cleaned.split("-")[0]
    return bool(re.match(r"^(C\d|CM|CR|CW|M\d|MR\d)", zone_class))


@dataclass
class ScoreResult:
    """Scores plus an explanation of which components actually contributed."""

    scores: pd.Series
    components: pd.DataFrame
    used_weights: dict[str, float] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _combine(
    components: dict[str, pd.Series], weights: dict[str, float], index: pd.Index
) -> ScoreResult:
    """
    Weighted blend of [0,1] components into a 0-100 score.

    A component with no data anywhere is dropped and the remaining weights are
    renormalized, so missing sources shift the *basis* of the score rather than
    quietly dragging every parcel toward zero. Individual gaps inside an
    otherwise-populated component are treated as neutral.
    """
    usable: dict[str, pd.Series] = {}
    skipped: list[str] = []

    for name, series in components.items():
        if series is None:
            skipped.append(name)
            continue
        values = pd.to_numeric(series, errors="coerce")
        if values.notna().sum() == 0:
            skipped.append(name)
            continue
        usable[name] = values.fillna(NEUTRAL)

    # A component identical across every row cannot rank anything: it would
    # hold weight while conveying nothing, muting the components that do
    # discriminate. This happens in practice when a peer group falls below the
    # size floor and every parcel collapses to the same fallback median.
    #
    # Two guards matter here. Variance is measured *after* the neutral fill,
    # because that is the series actually scored — [1.0, NaN] becomes
    # [1.0, 0.5], which discriminates fine. And constants are only dropped when
    # something else can still rank: if every component is flat, they are all
    # kept, so the frame scores evenly rather than returning nulls.
    if len(index) > 1:
        varying = {
            name: values
            for name, values in usable.items()
            if values.nunique(dropna=True) > 1
        }
        if varying:
            skipped.extend(name for name in usable if name not in varying)
            usable = varying

    if not usable:
        return ScoreResult(
            scores=pd.Series(np.nan, index=index, dtype=float),
            components=pd.DataFrame(index=index),
            skipped=skipped,
            notes=["No scoring inputs were available for this parcel set."],
        )

    total_weight = sum(weights[name] for name in usable)
    normalized = {name: weights[name] / total_weight for name in usable}

    score = pd.Series(0.0, index=index, dtype=float)
    for name, series in usable.items():
        score = score + series.clip(0, 1) * normalized[name]

    notes: list[str] = []
    if skipped:
        notes.append(
            "Scored without "
            + ", ".join(sorted(skipped))
            + " (no data, or no variation across this set); remaining weights "
            "were renormalized."
        )

    return ScoreResult(
        scores=(score * 100).round(1),
        components=pd.DataFrame(usable, index=index),
        used_weights=normalized,
        skipped=skipped,
        notes=notes,
    )


def acquisition_score(
    frame: pd.DataFrame, weights: config.AcquisitionWeights | None = None
) -> ScoreResult:
    """Rank parcels by off-market sourcing signal strength."""
    weights = weights or config.ACQUISITION_WEIGHTS
    index = frame.index

    # Basis gap: assessed value per square foot against same-use, same-city
    # peers. Below-peer parcels rank higher.
    if "value_per_sqft" in frame.columns:
        peer = transform.peer_median(frame, "value_per_sqft")
        ratio = pd.to_numeric(frame["value_per_sqft"], errors="coerce") / peer.replace(
            0, np.nan
        )
        basis = percentile_rank(ratio, invert=True)
    else:
        basis = pd.Series(np.nan, index=index, dtype=float)

    dormancy = (
        percentile_rank(frame["years_since_permit"])
        if "years_since_permit" in frame.columns
        else pd.Series(np.nan, index=index, dtype=float)
    )

    components = {
        "tenure": percentile_rank(frame.get("tenure_years", pd.Series(dtype=float))),
        "basis_gap": basis,
        "building_age": percentile_rank(
            frame.get("building_age", pd.Series(dtype=float))
        ),
        "permit_dormancy": dormancy,
        "improvement_ratio": percentile_rank(
            frame.get("improvement_ratio", pd.Series(dtype=float)), invert=True
        ),
    }
    weight_map = {
        "tenure": weights.tenure,
        "basis_gap": weights.basis_gap,
        "building_age": weights.building_age,
        "permit_dormancy": weights.permit_dormancy,
        "improvement_ratio": weights.improvement_ratio,
    }
    return _combine(components, weight_map, index)


def far_gap(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series, list[str]]:
    """
    Unused floor area as a share of what the site could plausibly hold.

    Returns (gap in [0,1], the max FAR assumed, notes). Where a zone string
    resolves, the ceiling comes from the height district. Otherwise it comes
    from the 75th percentile of what comparable neighbors actually built —
    a zoning-free measure that works countywide.
    """
    notes: list[str] = []
    index = frame.index
    built = pd.to_numeric(frame.get("built_far", pd.Series(dtype=float)), errors="coerce")
    built = built.reindex(index)

    max_far = pd.Series(np.nan, index=index, dtype=float)

    if "zone" in frame.columns:
        zoned = frame["zone"].map(max_far_for_zone)
        max_far = pd.to_numeric(zoned, errors="coerce")
        resolved = int(max_far.notna().sum())
        if resolved:
            notes.append(
                f"{resolved:,} parcels scored against zoned FAR (LA City height districts)."
            )

    # Peer-relative ceiling for everything zoning could not answer.
    unresolved = max_far.isna()
    if unresolved.any() and built.notna().any():
        keys = [k for k in config.PEER_GROUP_KEYS if k in frame.columns]
        if keys:
            peer_p75 = built.groupby(
                [frame[k] for k in keys], dropna=False
            ).transform(lambda s: s.quantile(0.75))
        else:
            peer_p75 = pd.Series(built.quantile(0.75), index=index)
        peer_p75 = peer_p75.replace(0, np.nan).fillna(built.quantile(0.75))
        max_far[unresolved] = peer_p75[unresolved]
        notes.append(
            f"{int(unresolved.sum()):,} parcels scored against neighbor-built FAR "
            "(no zoning data outside the City of LA)."
        )

    # LA measures FAR against buildable area, not gross lot area. Haircut the
    # ceiling so the headroom estimate stays conservative.
    max_far = max_far * config.BUILDABLE_AREA_FACTOR

    gap = (max_far - built) / max_far.replace(0, np.nan)
    gap = gap.clip(lower=0, upper=1)
    gap[~np.isfinite(gap)] = np.nan

    # A vacant parcel has no building; its headroom is total, not unknown.
    if "general_use" in frame.columns:
        vacant = frame["general_use"].astype("string").str.contains(
            "vacant", case=False, na=False
        )
        gap[vacant & built.isna()] = 1.0

    return gap, max_far, notes


def development_score(
    frame: pd.DataFrame, weights: config.DevelopmentWeights | None = None
) -> ScoreResult:
    """Rank parcels by redevelopment headroom."""
    weights = weights or config.DEVELOPMENT_WEIGHTS
    index = frame.index

    gap, _, gap_notes = far_gap(frame)

    if "toc_tier" in frame.columns:
        transit = pd.to_numeric(frame["toc_tier"], errors="coerce").fillna(0) / 4.0
        transit = transit.clip(0, 1)
    else:
        transit = pd.Series(np.nan, index=index, dtype=float)

    if "zone" in frame.columns:
        corridor = frame["zone"].map(is_commercial_zone).astype(float)
    else:
        corridor = pd.Series(np.nan, index=index, dtype=float)

    components = {
        "far_gap": gap,
        "lot_size": percentile_rank(frame.get("lot_sqft", pd.Series(dtype=float))),
        "improvement_ratio": percentile_rank(
            frame.get("improvement_ratio", pd.Series(dtype=float)), invert=True
        ),
        "transit": transit,
        "corridor_upzone": corridor,
    }
    weight_map = {
        "far_gap": weights.far_gap,
        "lot_size": weights.lot_size,
        "improvement_ratio": weights.improvement_ratio,
        "transit": weights.transit,
        "corridor_upzone": weights.corridor_upzone,
    }
    result = _combine(components, weight_map, index)
    result.notes.extend(gap_notes)
    return result


def explain(frame: pd.DataFrame, row_index: Any, components: pd.DataFrame) -> list[str]:
    """Plain-language reasons a specific parcel scored where it did."""
    if row_index not in frame.index:
        return []
    row = frame.loc[row_index]
    reasons: list[str] = []

    tenure = row.get("tenure_years")
    if pd.notna(tenure):
        base = row.get("base_year")
        base_text = f" (base year {int(base)})" if pd.notna(base) else ""
        reasons.append(f"Held ~{int(tenure)} years since last reassessment{base_text}.")

    age = row.get("building_age")
    if pd.notna(age):
        reasons.append(f"Structure is about {int(age)} years old.")

    ratio = row.get("improvement_ratio")
    if pd.notna(ratio):
        reasons.append(
            f"{ratio:.0%} of assessed value is in the building — "
            f"{'value is mostly in the land' if ratio < 0.35 else 'improvements carry the value'}."
        )

    built = row.get("built_far")
    if pd.notna(built):
        reasons.append(f"Built FAR is {built:.2f}.")

    zone = row.get("zone")
    if isinstance(zone, str) and zone:
        far = max_far_for_zone(zone)
        far_text = f", implying roughly {far:.1f}:1 FAR" if far else ""
        reasons.append(f"Zoned {zone}{far_text}.")

    return reasons


def band_score(
    series: pd.Series, low: float, sweet_low: float, sweet_high: float, high: float
) -> pd.Series:
    """
    Score a value by where it falls in a preferred band, in [0, 1].

    Flat at 1.0 across the sweet spot, ramping in from ``low`` and tapering out
    to ``high``, zero beyond either end. Percentile ranking cannot express this:
    for rehab scale, bigger is not monotonically better — a 400,000 sqft
    regional mall is a worse fit than a 60,000 sqft neighborhood center, and a
    ranking would put the mall on top.
    """
    values = pd.to_numeric(series, errors="coerce")
    score = pd.Series(np.nan, index=series.index, dtype=float)

    below = values < low
    ramp_up = (values >= low) & (values < sweet_low)
    plateau = (values >= sweet_low) & (values <= sweet_high)
    ramp_down = (values > sweet_high) & (values <= high)
    above = values > high

    score[below] = 0.0
    score[above] = 0.0
    score[plateau] = 1.0
    if sweet_low > low:
        score[ramp_up] = (values[ramp_up] - low) / (sweet_low - low)
    else:
        score[ramp_up] = 1.0
    if high > sweet_high:
        score[ramp_down] = (high - values[ramp_down]) / (high - sweet_high)
    else:
        score[ramp_down] = 1.0

    return score


def is_retail_use(use_description: Any) -> bool:
    """True when an assessor use description reads as retail."""
    if not isinstance(use_description, str) or not use_description.strip():
        return False
    lowered = use_description.lower()
    return any(keyword in lowered for keyword in config.RETAIL_USE_KEYWORDS)


def shopping_center_candidates(
    frame: pd.DataFrame,
    min_lot_sqft: float | None = None,
    min_building_sqft: float | None = None,
    max_building_sqft: float | None = None,
) -> pd.DataFrame:
    """
    Narrow a parcel set to plausible shopping centers.

    Retail use plus size gates. The gates do most of the work: the assessor
    codes a corner liquor store and a 90,000 sqft neighborhood center under the
    same "Store" description, and only the second is a repositioning candidate.
    """
    min_lot = config.REHAB_MIN_LOT_SQFT if min_lot_sqft is None else min_lot_sqft
    min_building = (
        config.REHAB_MIN_BUILDING_SQFT if min_building_sqft is None else min_building_sqft
    )
    max_building = (
        config.REHAB_MAX_BUILDING_SQFT if max_building_sqft is None else max_building_sqft
    )

    mask = pd.Series(True, index=frame.index)

    if "specific_use" in frame.columns:
        retail = frame["specific_use"].map(is_retail_use)
        # Fall back to the general use type when the specific one is unavailable.
        if not retail.any() and "general_use" in frame.columns:
            retail = frame["general_use"].map(is_retail_use)
        mask &= retail.fillna(False)

    if "lot_sqft" in frame.columns:
        mask &= pd.to_numeric(frame["lot_sqft"], errors="coerce") >= min_lot
    if "building_sqft" in frame.columns:
        building = pd.to_numeric(frame["building_sqft"], errors="coerce")
        mask &= building.between(min_building, max_building)

    return frame[mask.fillna(False)]


def rehab_score(
    frame: pd.DataFrame, weights: config.RehabWeights | None = None
) -> ScoreResult:
    """
    Rank retail centers as renovate-and-sell candidates.

    Deliberately inverted from ``acquisition_score`` on the value dimension.
    That score rewards a cheap parcel; this one rewards a *worn-out building on
    expensive land*. Cheap dirt under a tired center is not a repositioning
    opportunity — it means the corridor will not carry the better tenants the
    whole thesis depends on.
    """
    weights = weights or config.REHAB_WEIGHTS
    index = frame.index

    # Physical obsolescence: how old the structure is.
    obsolescence = percentile_rank(frame.get("building_age", pd.Series(dtype=float)))

    # How long since anything substantial was done to it. The assessor's
    # effective year built already folds renovations in, so a long gap here
    # means dated finishes, systems and layout — the "outdated facilities" case.
    dormancy_parts: list[pd.Series] = []
    if "years_since_improvement" in frame.columns:
        dormancy_parts.append(percentile_rank(frame["years_since_improvement"]))
    if "years_since_permit" in frame.columns:
        permit_gap = percentile_rank(frame["years_since_permit"])
        if permit_gap.notna().any():
            dormancy_parts.append(permit_gap)
    if dormancy_parts:
        dormancy = pd.concat(dormancy_parts, axis=1).mean(axis=1)
    else:
        dormancy = pd.Series(np.nan, index=index, dtype=float)

    # Location quality from submarket land values, not the subject's own frozen
    # assessment. See transform.location_quality_index for why that matters.
    location = percentile_rank(transform.location_quality_index(frame))

    under_management = percentile_rank(frame.get("tenure_years", pd.Series(dtype=float)))

    scale = (
        band_score(
            frame["building_sqft"],
            config.REHAB_MIN_BUILDING_SQFT,
            config.REHAB_SWEET_SPOT_SQFT[0],
            config.REHAB_SWEET_SPOT_SQFT[1],
            config.REHAB_MAX_BUILDING_SQFT,
        )
        if "building_sqft" in frame.columns
        else pd.Series(np.nan, index=index, dtype=float)
    )

    components = {
        "obsolescence": obsolescence,
        "renovation_dormancy": dormancy,
        "location_quality": location,
        "under_management": under_management,
        "rehab_scale": scale,
    }
    weight_map = {
        "obsolescence": weights.obsolescence,
        "renovation_dormancy": weights.renovation_dormancy,
        "location_quality": weights.location_quality,
        "under_management": weights.under_management,
        "rehab_scale": weights.rehab_scale,
    }
    result = _combine(components, weight_map, index)
    result.notes.append(
        "Location is scored from submarket land values rather than each "
        "parcel's own assessment, which Prop 13 freezes at its base year."
    )
    return result


def rehab_flags(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Plain-language condition flags for a retail center.

    These are observations, not scores — the things you would want to know
    before deciding whether a site is worth a drive-by.
    """
    out = pd.DataFrame(index=frame.index)

    if "renovation_gap" in frame.columns:
        out["never_renovated"] = frame["renovation_gap"].fillna(0).eq(0)
    if "built_far" in frame.columns:
        out["excess_parking"] = pd.to_numeric(
            frame["built_far"], errors="coerce"
        ) < config.REHAB_LOW_COVERAGE_THRESHOLD
    if "improvement_per_sqft" in frame.columns:
        peer = transform.peer_median(frame, "improvement_per_sqft")
        out["below_peer_condition"] = (
            pd.to_numeric(frame["improvement_per_sqft"], errors="coerce") < peer * 0.6
        )
    if "mls_listed" in frame.columns:
        out["already_listed"] = frame["mls_listed"].fillna(False).astype(bool)

    return out.fillna(False)
