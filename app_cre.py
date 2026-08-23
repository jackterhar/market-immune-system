"""
LA Commercial Real Estate Opportunity Screener
==============================================

Screens LA County parcels for two kinds of opportunity:

  • Off-market acquisitions — long-tenured ownership, below-peer assessed
    basis, aging structures, no recent permit activity.
  • Development sites — unused floor area against zoned or neighbor-built
    ceilings, low improvement ratios, transit proximity.

Built on the LA County assessor roll (all 88 cities), City of LA permits and
zoning, and — optionally — your MLS feed.

Usage:
  pip install -r requirements.txt
  streamlit run app_cre.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cre import cache, config, demo, pipeline, rooftops, scoring, valuation  # noqa: E402
from cre.sources import mls  # noqa: E402

st.set_page_config(
    layout="wide",
    page_title="LA CRE Opportunity Screener",
    page_icon="🏢",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .stApp { background-color: #0e1117; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .metric-note { color: #8b93a7; font-size: 0.8rem; }
    div[data-testid="stMetricValue"] { font-size: 1.6rem; }
</style>
""",
    unsafe_allow_html=True,
)

SCORE_HELP = {
    "acquisition": (
        "Ranks parcels by **ownership tenure signal**, not by price. A low "
        "Prop 13 assessment reflects a distant base year, not a discount. "
        "Long tenure correlates with below-market in-place rents and with "
        "generational transfer events — but it also means a large embedded "
        "capital gain, which cuts against a sale. Treat it as a sourcing "
        "list, not a valuation."
    ),
    "development": (
        "Ranks parcels by **unused floor area**. Inside the City of LA the "
        "ceiling comes from the zone's height district; elsewhere it comes "
        "from what comparable neighbors actually built. Every FAR figure is "
        "a screening heuristic — LA measures FAR against buildable area, and "
        "overlays, specific plans and Q/D conditions are invisible here. "
        "Verify on ZIMAS before acting."
    ),
}


# ── Formatting helpers ───────────────────────────────────────────────────


def money(value) -> str:
    if pd.isna(value):
        return "—"
    value = float(value)
    if abs(value) >= 1e6:
        return f"${value / 1e6:,.2f}M"
    if abs(value) >= 1e3:
        return f"${value / 1e3:,.0f}K"
    return f"${value:,.0f}"


def number(value, suffix: str = "") -> str:
    if pd.isna(value):
        return "—"
    return f"{float(value):,.0f}{suffix}"


def score_color(score: float) -> str:
    """Map a 0-100 score onto a red-to-green hex ramp for the map."""
    if pd.isna(score):
        return "#4a5568"
    fraction = max(0.0, min(1.0, float(score) / 100.0))
    red = int(220 - 150 * fraction)
    green = int(70 + 140 * fraction)
    return f"#{red:02x}{green:02x}6b"


# ── Sidebar: data loading ────────────────────────────────────────────────


def sidebar_load_controls() -> dict:
    st.sidebar.header("Data")

    demo_mode = st.sidebar.toggle(
        "Demo data (no network)",
        value=True,
        help=(
            "Synthetic parcels for evaluating the interface. Turn this off to "
            "pull the real LA County assessor roll."
        ),
    )

    settings: dict = {"demo": demo_mode}

    if demo_mode:
        settings["demo_rows"] = st.sidebar.slider(
            "Synthetic parcels", 500, 20_000, 4_000, step=500
        )
        return settings

    st.sidebar.caption(
        "The countywide roll is ~2.4M parcels. Filtering to commercial uses "
        "cuts that sharply, but the first pull still takes several minutes. "
        "Results are cached for a week."
    )

    settings["include_commercial"] = st.sidebar.checkbox(
        "Commercial / industrial", value=True
    )
    settings["include_vacant"] = st.sidebar.checkbox("Vacant land", value=True)
    settings["include_income_residential"] = st.sidebar.checkbox(
        "Residential (incl. multifamily)",
        value=False,
        help="Large — adds most of the county's 2.4M parcels.",
    )

    settings["max_rows"] = st.sidebar.select_slider(
        "Row cap",
        options=[10_000, 50_000, 100_000, 250_000, 500_000, None],
        value=50_000,
        format_func=lambda v: "No cap" if v is None else f"{v:,}",
    )

    st.sidebar.subheader("Enrichments")
    settings["with_permits"] = st.sidebar.checkbox(
        "Building permits", value=True, help="City of LA only."
    )
    settings["with_zoning"] = st.sidebar.checkbox(
        "Zoning + transit overlays", value=True, help="City of LA only."
    )

    creds = mls.credentials_from_env()
    settings["with_mls"] = st.sidebar.checkbox(
        "MLS listings",
        value=False,
        disabled=not creds.configured,
        help=(
            "Configured via environment variables."
            if creds.configured
            else "Not configured. Set MLS_BASE_URL and credentials to enable."
        ),
    )
    settings["use_cache"] = st.sidebar.checkbox("Use cache", value=True)

    return settings


def load_data(settings: dict) -> pipeline.PipelineResult:
    """Run the pipeline, streaming progress into the sidebar."""
    if settings["demo"]:
        parcels = demo.generate(settings["demo_rows"])
        multifamily = demo.generate_multifamily(
            max(120, settings["demo_rows"] // 7)
        )
        rooftop_result = rooftops.summarize_areas(multifamily)
        parcels = rooftops.attach(parcels, rooftop_result)
        parcels, valuation_result = valuation.attach(parcels)
        result = pipeline.PipelineResult(parcels=parcels)
        result.rooftops = rooftop_result
        result.multifamily = multifamily
        result.acquisition = scoring.acquisition_score(parcels)
        result.development = scoring.development_score(parcels)
        parcels["acquisition_score"] = result.acquisition.scores
        parcels["development_score"] = result.development.scores
        gap, max_far, notes = scoring.far_gap(parcels)
        parcels["far_headroom"] = gap
        parcels["assumed_max_far"] = max_far
        parcels["unused_buildable_sqft"] = (
            (max_far - parcels["built_far"].fillna(0)) * parcels["lot_sqft"]
        ).clip(lower=0)
        result.notes = (
            ["Demo data — synthetic parcels, not real records."]
            + valuation_result.notes
            + rooftop_result.notes
            + notes
        )
        return result

    use_types = pipeline.build_use_types(
        settings["include_commercial"],
        settings["include_vacant"],
        settings["include_income_residential"],
    )
    if not use_types:
        result = pipeline.PipelineResult(parcels=pd.DataFrame())
        result.notes.append("Select at least one property type.")
        return result

    status = st.sidebar.status("Loading…", expanded=True)

    def report(stage: str, count: int) -> None:
        status.update(label=f"{stage}… {count:,} rows" if count else f"{stage}…")

    try:
        result = pipeline.run(
            use_types=use_types,
            max_rows=settings["max_rows"],
            with_permits=settings["with_permits"],
            with_zoning=settings["with_zoning"],
            with_mls=settings["with_mls"],
            use_cache=settings["use_cache"],
            progress=report,
        )
    except Exception as exc:  # surface, never swallow
        status.update(label="Load failed", state="error")
        failed = pipeline.PipelineResult(parcels=pd.DataFrame())
        failed.notes.append(f"{type(exc).__name__}: {exc}")
        return failed

    state = "complete" if result.ok else "error"
    status.update(label=f"Loaded {len(result.parcels):,} parcels", state=state)
    return result


# ── Sidebar: filters ─────────────────────────────────────────────────────


DEFAULT_PRICE_RANGE = (1_000_000.0, 4_000_000.0)


def price_filter(parcels: pd.DataFrame) -> pd.Series:
    """
    Budget filter on *estimated* purchase price, not assessed value.

    Assessed value would be the wrong basis: Prop 13 freezes it at the base
    year, so filtering on it selects on how long someone has owned a property
    rather than on what it costs. The estimate is built from parcels that
    changed hands recently, whose assessments are their actual sale prices.
    """
    if "estimated_value" not in parcels.columns:
        return pd.Series(True, index=parcels.index)

    values = pd.to_numeric(parcels["estimated_value"], errors="coerce")
    if values.notna().sum() == 0:
        st.sidebar.caption(
            "No market value estimates available — too few recent transfers in "
            "this parcel set to build price benchmarks."
        )
        return pd.Series(True, index=parcels.index)

    ceiling = float(min(values.quantile(0.995), 50_000_000))
    ceiling = max(ceiling, DEFAULT_PRICE_RANGE[1])
    default = (
        min(DEFAULT_PRICE_RANGE[0], ceiling),
        min(DEFAULT_PRICE_RANGE[1], ceiling),
    )

    low, high = st.sidebar.slider(
        "Purchase price ($M)",
        0.0,
        ceiling / 1e6,
        (default[0] / 1e6, default[1] / 1e6),
        step=0.25,
        format="$%.2fM",
        help=(
            "Estimated from parcels that changed hands in the last five years, "
            "whose assessments equal their sale price under Prop 13. This is "
            "NOT assessed value — long-held parcels are assessed at a fraction "
            "of what they would trade for."
        ),
    )
    in_range = values.between(low * 1e6, high * 1e6)

    keep_unpriced = st.sidebar.checkbox(
        "Include parcels without an estimate",
        value=False,
        help="Parcels with too few nearby transfers to price. They cannot be budget-checked.",
    )
    if keep_unpriced:
        in_range |= values.isna()

    priced = int(in_range.sum())
    st.sidebar.caption(f"{priced:,} parcels in budget.")
    return in_range


def sidebar_filters(parcels: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filters")
    mask = pd.Series(True, index=parcels.index)

    mask &= price_filter(parcels)

    if "situs_city" in parcels.columns:
        cities = sorted(parcels["situs_city"].dropna().unique().tolist())
        chosen = st.sidebar.multiselect("City", cities, default=[])
        if chosen:
            mask &= parcels["situs_city"].isin(chosen)

    if "specific_use" in parcels.columns:
        uses = sorted(parcels["specific_use"].dropna().unique().tolist())
        chosen_uses = st.sidebar.multiselect("Use type", uses, default=[])
        if chosen_uses:
            mask &= parcels["specific_use"].isin(chosen_uses)

    def range_filter(column: str, label: str, step: float = 1.0, fmt: str = "%.0f"):
        nonlocal mask
        if column not in parcels.columns:
            return
        series = pd.to_numeric(parcels[column], errors="coerce")
        if series.notna().sum() == 0:
            return
        low, high = float(series.min()), float(series.max())
        if low == high:
            return
        selected = st.sidebar.slider(
            label, low, high, (low, high), step=step, format=fmt
        )
        if selected != (low, high):
            mask &= series.between(*selected) | series.isna()

    range_filter("lot_sqft", "Lot size (sqft)", step=500.0)
    range_filter("building_sqft", "Building size (sqft)", step=500.0)
    range_filter("total_value", "Assessed value ($)", step=50_000.0)
    range_filter("tenure_years", "Ownership tenure (years)")
    range_filter("building_age", "Building age (years)")

    if "mls_listed" in parcels.columns and parcels["mls_listed"].any():
        if st.sidebar.checkbox(
            "Hide parcels already listed",
            value=True,
            help="Off-market sourcing means excluding what is already marketed.",
        ):
            mask &= ~parcels["mls_listed"].fillna(False).astype(bool)

    return parcels[mask]


# ── Views ────────────────────────────────────────────────────────────────


def render_map(frame: pd.DataFrame, score_column: str) -> None:
    if "lat" not in frame.columns or frame["lat"].notna().sum() == 0:
        st.info("No coordinates available for the current selection.")
        return

    plot = frame.dropna(subset=["lat", "lon"]).copy()
    if plot.empty:
        st.info("No mappable parcels in the current selection.")
        return

    plot = plot.nlargest(min(len(plot), 5_000), score_column, keep="all")
    plot["color"] = plot[score_column].map(score_color)
    if "lot_sqft" in plot.columns:
        radius = pd.to_numeric(plot["lot_sqft"], errors="coerce").fillna(5_000)
        plot["radius"] = (np.sqrt(radius) / 2).clip(15, 120)
    else:
        plot["radius"] = 40

    st.map(plot, latitude="lat", longitude="lon", color="color", size="radius")
    st.caption(
        f"Showing the top {len(plot):,} parcels by {score_column.replace('_', ' ')}. "
        "Color runs red (low) to green (high); circle size tracks lot area."
    )


def score_table(
    frame: pd.DataFrame, sort_column: str, key: str, limit: int = 500
) -> None:
    display = pipeline.display_frame(frame).sort_values(
        sort_column, ascending=False, na_position="last"
    )
    st.dataframe(
        display.head(limit),
        width="stretch",
        hide_index=True,
        column_config={
            "parcel_id": st.column_config.TextColumn("APN", width="small"),
            "situs_address": st.column_config.TextColumn("Address", width="medium"),
            "situs_city": st.column_config.TextColumn("City", width="small"),
            "specific_use": st.column_config.TextColumn("Use", width="small"),
            "acquisition_score": st.column_config.ProgressColumn(
                "Acquisition", min_value=0, max_value=100, format="%.0f"
            ),
            "development_score": st.column_config.ProgressColumn(
                "Development", min_value=0, max_value=100, format="%.0f"
            ),
            "tenure_years": st.column_config.NumberColumn("Tenure (yr)", format="%.0f"),
            "building_age": st.column_config.NumberColumn("Age (yr)", format="%.0f"),
            "lot_sqft": st.column_config.NumberColumn("Lot sqft", format="%,d"),
            "building_sqft": st.column_config.NumberColumn("Bldg sqft", format="%,d"),
            "built_far": st.column_config.NumberColumn("Built FAR", format="%.2f"),
            "assumed_max_far": st.column_config.NumberColumn("Max FAR*", format="%.2f"),
            "far_headroom": st.column_config.NumberColumn("Headroom", format="%.0%%"),
            "unused_buildable_sqft": st.column_config.NumberColumn(
                "Unused sqft", format="%,d"
            ),
            "estimated_value": st.column_config.NumberColumn(
                "Est. price",
                format="$%,d",
                help="Estimated market value from recent nearby transfers. Not assessed value.",
            ),
            "value_comp_count": st.column_config.NumberColumn(
                "Comps", format="%d", help="Transfers behind the estimate. Fewer means less reliable."
            ),
            "total_value": st.column_config.NumberColumn("Assessed", format="$%,d"),
            "value_per_sqft": st.column_config.NumberColumn("$/sqft", format="$%.0f"),
            "improvement_ratio": st.column_config.NumberColumn("Imp %", format="%.0%%"),
            "years_since_permit": st.column_config.NumberColumn(
                "Permit gap (yr)", format="%.1f"
            ),
            "area_new_units": st.column_config.NumberColumn(
                "New units nearby", format="%,d"
            ),
            "area_growth_percentile": st.column_config.NumberColumn(
                "Growth pct", format="%.0%%"
            ),
        },
    )
    st.download_button(
        "Download this list as CSV",
        display.to_csv(index=False).encode(),
        file_name=f"la_cre_{sort_column}.csv",
        mime="text/csv",
        key=f"download_{key}",
    )


def render_screener(frame: pd.DataFrame, result: pipeline.PipelineResult) -> None:
    columns = st.columns(5)
    columns[0].metric("Parcels", f"{len(frame):,}")
    if "acquisition_score" in frame.columns:
        columns[1].metric(
            "Median acquisition", f"{frame['acquisition_score'].median():.0f}"
        )
    if "development_score" in frame.columns:
        columns[2].metric(
            "Median development", f"{frame['development_score'].median():.0f}"
        )
    if "tenure_years" in frame.columns:
        columns[3].metric("Median tenure", f"{frame['tenure_years'].median():.0f} yr")
    if "unused_buildable_sqft" in frame.columns:
        total = frame["unused_buildable_sqft"].sum()
        columns[4].metric("Unused buildable", f"{total / 1e6:,.1f}M sqft")

    lens = st.radio(
        "Rank by",
        ["acquisition_score", "development_score"],
        format_func=lambda c: c.replace("_score", "").title(),
        horizontal=True,
        key="screener_lens",
    )
    render_map(frame, lens)
    score_table(frame, lens, key="screener")


def render_lens(
    frame: pd.DataFrame,
    score_column: str,
    result: pipeline.PipelineResult,
    help_key: str,
) -> None:
    st.info(SCORE_HELP[help_key])

    score_result = (
        result.acquisition if help_key == "acquisition" else result.development
    )
    if score_result is not None:
        if score_result.used_weights:
            weights = ", ".join(
                f"{name.replace('_', ' ')} {weight:.0%}"
                for name, weight in sorted(
                    score_result.used_weights.items(), key=lambda kv: -kv[1]
                )
            )
            st.caption(f"Active weights — {weights}")
        for note in score_result.notes:
            st.caption(f"· {note}")

    if score_column not in frame.columns or frame.empty:
        st.warning("No scored parcels in the current selection.")
        return

    threshold = st.slider("Minimum score", 0, 100, 70, key=f"thr_{score_column}")
    subset = frame[frame[score_column] >= threshold]
    st.caption(f"{len(subset):,} of {len(frame):,} parcels at or above {threshold}.")
    render_map(subset, score_column)
    score_table(subset, score_column, key=f"lens_{help_key}")


REHAB_HELP = (
    "Finds **tired retail centers on corridors that can carry better tenants** — "
    "the renovate, re-tenant and sell play. Ranks on building age, years since "
    "any substantial improvement, submarket land values, ownership tenure, and "
    "whether the center is a workable size.\n\n"
    "Note the inversion: unlike the off-market score, this one wants *expensive "
    "dirt under a worn-out building*. Cheap land beneath a tired center is not a "
    "repositioning — it means the corridor will not support the tenants the "
    "whole thesis depends on. Location is measured from submarket land values "
    "rather than each parcel's own assessment, which Prop 13 freezes at its "
    "base year."
)


def rehab_table(frame: pd.DataFrame) -> None:
    display = pipeline.rehab_display_frame(frame).sort_values(
        "rehab_score", ascending=False, na_position="last"
    )
    st.dataframe(
        display.head(500),
        width="stretch",
        hide_index=True,
        column_config={
            "parcel_id": st.column_config.TextColumn("APN", width="small"),
            "situs_address": st.column_config.TextColumn("Address", width="medium"),
            "situs_city": st.column_config.TextColumn("City", width="small"),
            "specific_use": st.column_config.TextColumn("Use", width="small"),
            "rehab_score": st.column_config.ProgressColumn(
                "Rehab", min_value=0, max_value=100, format="%.0f"
            ),
            "building_sqft": st.column_config.NumberColumn("Bldg sqft", format="%,d"),
            "lot_sqft": st.column_config.NumberColumn("Lot sqft", format="%,d"),
            "built_far": st.column_config.NumberColumn("Coverage", format="%.2f"),
            "year_built": st.column_config.NumberColumn("Built", format="%d"),
            "building_age": st.column_config.NumberColumn("Age (yr)", format="%.0f"),
            "renovation_gap": st.column_config.NumberColumn(
                "Reno (yr)",
                format="%.0f",
                help="Years the assessor added to the effective year built. 0 = never substantially improved.",
            ),
            "years_since_improvement": st.column_config.NumberColumn(
                "Since reno", format="%.0f"
            ),
            "improvement_per_sqft": st.column_config.NumberColumn(
                "Imp $/sqft",
                format="$%.0f",
                help="Assessed value of the structure per square foot. Low means a worn-out building.",
            ),
            "tenure_years": st.column_config.NumberColumn("Tenure (yr)", format="%.0f"),
            "estimated_value": st.column_config.NumberColumn(
                "Est. price",
                format="$%,d",
                help="Estimated market value from recent nearby transfers. Not assessed value.",
            ),
            "value_comp_count": st.column_config.NumberColumn(
                "Comps", format="%d", help="Transfers behind the estimate. Fewer means less reliable."
            ),
            "total_value": st.column_config.NumberColumn("Assessed", format="$%,d"),
            "area_new_units": st.column_config.NumberColumn(
                "New units nearby",
                format="%,d",
                help="Multifamily units delivered in this ZIP in recent years.",
            ),
            "area_units_permitted": st.column_config.NumberColumn(
                "Units permitted", format="%,d", help="City of LA only. Not yet built."
            ),
            "never_renovated": st.column_config.CheckboxColumn("Never reno"),
            "excess_parking": st.column_config.CheckboxColumn("Excess parking"),
            "below_peer_condition": st.column_config.CheckboxColumn("Below peer"),
            "years_since_permit": st.column_config.NumberColumn(
                "Permit gap", format="%.1f"
            ),
            "mls_listed": st.column_config.CheckboxColumn("Listed"),
        },
    )
    st.download_button(
        "Download this list as CSV",
        display.to_csv(index=False).encode(),
        file_name="la_cre_shopping_center_rehab.csv",
        mime="text/csv",
        key="download_rehab",
    )


def render_rehab(frame: pd.DataFrame) -> None:
    st.info(REHAB_HELP)

    with st.expander("What counts as a shopping center", expanded=False):
        st.caption(
            "The assessor codes a corner liquor store and a 90,000 sqft "
            "neighborhood center under the same description, so size gates do "
            "most of the filtering. Widen them to cast a broader net."
        )
        gate_columns = st.columns(3)
        min_lot = gate_columns[0].number_input(
            "Min lot sqft", 1_000, 500_000, config.REHAB_MIN_LOT_SQFT, step=5_000
        )
        min_building = gate_columns[1].number_input(
            "Min building sqft", 1_000, 200_000, config.REHAB_MIN_BUILDING_SQFT, step=1_000
        )
        max_building = gate_columns[2].number_input(
            "Max building sqft", 10_000, 1_000_000, config.REHAB_MAX_BUILDING_SQFT, step=10_000
        )

    candidates, score_result = pipeline.rehab_candidates(
        frame,
        min_lot_sqft=min_lot,
        min_building_sqft=min_building,
        max_building_sqft=max_building,
    )

    if candidates.empty:
        st.warning(
            "No shopping centers matched. Try widening the size gates above, or "
            "clearing the city and use-type filters in the sidebar."
        )
        return

    if score_result.used_weights:
        weights = ", ".join(
            f"{name.replace('_', ' ')} {weight:.0%}"
            for name, weight in sorted(
                score_result.used_weights.items(), key=lambda kv: -kv[1]
            )
        )
        st.caption(f"Active weights — {weights}")
    for note in score_result.notes:
        st.caption(f"· {note}")

    flag_columns = st.columns(4)
    only_never_renovated = flag_columns[0].checkbox(
        "Never renovated", value=False, help="Effective year built still equals the original."
    )
    only_excess_parking = flag_columns[1].checkbox(
        "Excess parking",
        value=False,
        help=f"Building covers under {config.REHAB_LOW_COVERAGE_THRESHOLD:.0%} of the lot — auto-era layout, possible pad site.",
    )
    only_below_peer = flag_columns[2].checkbox(
        "Below-peer condition",
        value=False,
        help="Structure assessed well under comparable centers per square foot.",
    )
    hide_listed = flag_columns[3].checkbox(
        "Hide listed", value=True, help="Exclude anything already being marketed."
    )

    subset = candidates
    if only_never_renovated and "never_renovated" in subset.columns:
        subset = subset[subset["never_renovated"]]
    if only_excess_parking and "excess_parking" in subset.columns:
        subset = subset[subset["excess_parking"]]
    if only_below_peer and "below_peer_condition" in subset.columns:
        subset = subset[subset["below_peer_condition"]]
    if hide_listed and "already_listed" in subset.columns:
        subset = subset[~subset["already_listed"]]

    threshold = st.slider("Minimum rehab score", 0, 100, 65, key="thr_rehab")
    subset = subset[subset["rehab_score"] >= threshold]

    # The rooftop link: a repositioning needs customers to reposition toward.
    if "area_new_units" in subset.columns and subset["area_new_units"].notna().any():
        max_units = int(pd.to_numeric(subset["area_new_units"], errors="coerce").max())
        if max_units > 0:
            min_rooftops = st.slider(
                "Minimum new apartment units in the area",
                0,
                max_units,
                0,
                step=max(1, max_units // 50),
                key="thr_rooftops",
                help=(
                    "Multifamily delivered nearby in recent years. Retail follows "
                    "rooftops — see the Rooftop growth tab."
                ),
            )
            if min_rooftops > 0:
                subset = subset[
                    pd.to_numeric(subset["area_new_units"], errors="coerce").fillna(0)
                    >= min_rooftops
                ]

    metric_columns = st.columns(5)
    metric_columns[0].metric("Centers", f"{len(subset):,}")
    if "never_renovated" in subset.columns:
        metric_columns[1].metric("Never renovated", f"{int(subset['never_renovated'].sum()):,}")
    if "building_age" in subset.columns and subset["building_age"].notna().any():
        metric_columns[2].metric("Median age", f"{subset['building_age'].median():.0f} yr")
    if "building_sqft" in subset.columns and subset["building_sqft"].notna().any():
        metric_columns[3].metric("Median size", f"{subset['building_sqft'].median():,.0f} sqft")
    if "tenure_years" in subset.columns and subset["tenure_years"].notna().any():
        metric_columns[4].metric("Median tenure", f"{subset['tenure_years'].median():.0f} yr")

    if subset.empty:
        st.warning("Nothing left after those filters. Lower the score threshold.")
        return

    st.caption(
        f"{len(subset):,} of {len(candidates):,} shopping centers at or above {threshold}."
    )
    render_map(subset, "rehab_score")
    rehab_table(subset)

    st.subheader("Renovation scope")
    scope_columns = st.columns([1, 2])
    cost_per_sqft = scope_columns[0].number_input(
        "Renovation cost $/sqft", 10, 800, 120, step=10,
        help="Your number. Facade, common areas, parking, signage, systems.",
    )
    total_sqft = float(pd.to_numeric(subset["building_sqft"], errors="coerce").sum())
    with scope_columns[1]:
        st.metric(
            f"Construction budget across {len(subset):,} centers",
            money(total_sqft * cost_per_sqft),
        )
        st.caption(
            f"{total_sqft:,.0f} sqft × ${cost_per_sqft}/sqft. Arithmetic on your "
            "input, nothing more."
        )
        if "estimated_value" in subset.columns:
            median_price = pd.to_numeric(
                subset["estimated_value"], errors="coerce"
            ).median()
            median_building = pd.to_numeric(
                subset["building_sqft"], errors="coerce"
            ).median()
            if pd.notna(median_price) and pd.notna(median_building):
                st.caption(
                    f"Median center: {money(median_price)} estimated price plus "
                    f"{money(median_building * cost_per_sqft)} of work."
                )

    st.warning(
        "**No returns are projected here, deliberately.** Doing that needs in-place "
        "rents, a rent roll, lease expiries and market pricing — none of which is in "
        "public assessor data. Assessed value is also *not* market value: Prop 13 "
        "freezes it at the base year, which is precisely why the long-held centers "
        "at the top of this list carry assessments far below what they would trade "
        "for. Any basis or cap-rate math built on these figures would be wrong.",
        icon="⚠️",
    )


ROOFTOP_HELP = (
    "**Retail follows rooftops.** A tired center on a corridor absorbing several "
    "hundred new apartments has an incoming customer base its current owner is "
    "not serving. The same center where nothing is being built does not. This "
    "tab measures where multifamily is actually landing, by ZIP.\n\n"
    "*Delivered* units come from the assessor roll — five or more units with a "
    "recent year built — and cover the whole county. *Permitted* units come "
    "from City of LA permits and are softer: a permit may lapse, get amended, "
    "or never break ground. The two are shown side by side and never summed."
)


def render_rooftops(
    frame: pd.DataFrame, result: pipeline.PipelineResult
) -> None:
    st.info(ROOFTOP_HELP)

    rooftop_result = result.rooftops
    if rooftop_result is None or not rooftop_result.ok:
        st.warning(
            "No rooftop growth data loaded. New multifamily is fetched with a "
            "separate query, since residential is excluded from the main "
            "commercial pull."
        )
        if rooftop_result is not None:
            for note in rooftop_result.notes:
                st.caption(f"· {note}")
        return

    areas = rooftop_result.areas
    multifamily = result.multifamily

    metric_columns = st.columns(4)
    metric_columns[0].metric(
        "Units delivered", f"{int(areas['units_delivered'].sum()):,}"
    )
    metric_columns[1].metric("Projects", f"{int(areas['projects'].sum()):,}")
    metric_columns[2].metric("Areas with growth", f"{len(areas):,}")
    if not areas.empty:
        top = areas.index[0]
        metric_columns[3].metric(
            "Busiest area",
            str(top),
            help=f"{int(areas.iloc[0]['units_delivered']):,} units delivered.",
        )
    if rooftop_result.has_permits and "units_permitted" in areas.columns:
        permitted = areas["units_permitted"].sum()
        if pd.notna(permitted) and permitted > 0:
            st.caption(
                f"Plus {int(permitted):,} units permitted but not yet counted as "
                "delivered — City of LA only."
            )

    for note in rooftop_result.notes:
        st.caption(f"· {note}")

    top_n = st.slider("Areas to show", 5, min(100, max(5, len(areas))), min(25, len(areas)))
    shown = areas.head(top_n).reset_index()

    display_columns = [
        rooftop_result.area_key,
        "city",
        "units_delivered",
        "projects",
        "largest_project",
        "units_per_year",
        "median_year_built",
    ]
    if "units_permitted" in shown.columns:
        display_columns.insert(4, "units_permitted")
    if "permit_projects" in shown.columns:
        display_columns.insert(5, "permit_projects")
    display_columns = [c for c in display_columns if c in shown.columns]

    st.dataframe(
        shown[display_columns],
        width="stretch",
        hide_index=True,
        column_config={
            rooftop_result.area_key: st.column_config.TextColumn("ZIP", width="small"),
            "city": st.column_config.TextColumn("City", width="small"),
            "units_delivered": st.column_config.NumberColumn(
                "Units built", format="%,d"
            ),
            "units_permitted": st.column_config.NumberColumn(
                "Units permitted",
                format="%,d",
                help="City of LA only. Not added to units built — a permit is not a building.",
            ),
            "permit_projects": st.column_config.NumberColumn("Permits", format="%d"),
            "projects": st.column_config.NumberColumn("Projects", format="%d"),
            "largest_project": st.column_config.NumberColumn(
                "Largest", format="%,d", help="Units in the biggest single project."
            ),
            "units_per_year": st.column_config.NumberColumn(
                "Units/yr", format="%.0f"
            ),
            "median_year_built": st.column_config.NumberColumn(
                "Median year", format="%d"
            ),
        },
    )
    st.download_button(
        "Download areas as CSV",
        shown[display_columns].to_csv(index=False).encode(),
        file_name="la_rooftop_growth_by_area.csv",
        mime="text/csv",
        key="download_rooftops",
    )

    if {"lat", "lon"}.issubset(areas.columns):
        mappable = areas.dropna(subset=["lat", "lon"]).head(top_n).copy()
        if not mappable.empty:
            units = mappable["units_delivered"].astype(float)
            span = max(float(units.max()), 1.0)
            mappable["radius"] = (units / span).pow(0.5) * 900 + 120
            mappable["color"] = "#4fd1c5"
            st.map(
                mappable, latitude="lat", longitude="lon",
                color="color", size="radius",
            )
            st.caption("Circle size tracks units delivered in each area.")

    if not multifamily.empty and "year_built" in multifamily.columns:
        st.subheader("Delivery by year")
        by_year = (
            multifamily.assign(
                year=pd.to_numeric(multifamily["year_built"], errors="coerce")
            )
            .dropna(subset=["year"])
            .groupby("year")["units"]
            .sum()
            .reset_index()
        )
        if not by_year.empty:
            # Plotly rather than st.bar_chart: the built-in chart inferred a
            # y-domain running into the negatives here and drew the bars
            # floating above the axis. An explicit figure is not worth arguing
            # with a heuristic over.
            by_year["year"] = by_year["year"].astype(int)
            figure = go.Figure(
                go.Bar(
                    x=by_year["year"],
                    y=by_year["units"],
                    marker_color="#4fd1c5",
                    hovertemplate="%{x}: %{y:,.0f} units<extra></extra>",
                )
            )
            figure.update_layout(
                height=300,
                margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e6e9ef",
                xaxis=dict(title="Year built", dtick=1, showgrid=False),
                yaxis=dict(
                    title="Units delivered",
                    rangemode="tozero",
                    gridcolor="#2a3040",
                ),
            )
            st.plotly_chart(figure, width="stretch")

    st.subheader("Commercial parcels in the busiest areas")
    if "area_new_units" not in frame.columns:
        st.caption("Growth has not been joined onto the commercial parcel set.")
        return

    top_areas = set(areas.head(top_n).index.astype(str))
    keys = frame.get(rooftop_result.area_key)
    if keys is None:
        st.caption(f"Commercial parcels carry no {rooftop_result.area_key} column.")
        return

    in_growth = frame[keys.astype("string").str.strip().isin(top_areas)]
    st.caption(
        f"{len(in_growth):,} of the {len(frame):,} filtered commercial parcels sit "
        f"in the top {top_n} growth areas."
    )
    if in_growth.empty:
        return

    sort_column = (
        "rehab_score" if "rehab_score" in in_growth.columns else "acquisition_score"
    )
    if sort_column in in_growth.columns:
        score_table(in_growth, sort_column, key="rooftop_crossref", limit=200)


def render_detail(frame: pd.DataFrame, result: pipeline.PipelineResult) -> None:
    if frame.empty:
        st.warning("No parcels in the current selection.")
        return

    ranked = frame.sort_values("acquisition_score", ascending=False, na_position="last")
    labels = {
        label: (
            f"{row.get('situs_address', '—')} · {row.get('situs_city', '—')} "
            f"(APN {row.get('parcel_id', '—')})"
        )
        for label, row in ranked.head(300).iterrows()
    }
    if not labels:
        st.warning("No parcels available.")
        return

    chosen = st.selectbox(
        "Parcel", list(labels), format_func=lambda label: labels[label]
    )
    row = frame.loc[chosen]

    columns = st.columns(4)
    columns[0].metric("Acquisition", f"{row.get('acquisition_score', float('nan')):.0f}")
    columns[1].metric("Development", f"{row.get('development_score', float('nan')):.0f}")
    columns[2].metric(
        "Est. price",
        money(row.get("estimated_value")),
        help=f"From {int(row.get('value_comp_count') or 0)} recent nearby transfers.",
    )
    columns[3].metric("Assessed", money(row.get("total_value")))

    left, right = st.columns(2)

    with left:
        st.subheader("Why it scored")
        for reason in scoring.explain(frame, chosen, pd.DataFrame()):
            st.markdown(f"- {reason}")
        headroom = row.get("unused_buildable_sqft")
        if pd.notna(headroom) and headroom > 0:
            st.markdown(
                f"- Roughly **{number(headroom)} sqft** of unused buildable area "
                "against the assumed ceiling."
            )
        if row.get("mls_listed"):
            st.markdown(
                f"- **Currently listed on MLS** at {money(row.get('mls_list_price'))} "
                f"(#{row.get('mls_listing_id')})."
            )

    with right:
        st.subheader("Peer comparison")
        peers = frame
        if "specific_use" in frame.columns and pd.notna(row.get("specific_use")):
            peers = peers[peers["specific_use"] == row["specific_use"]]
        if "situs_city" in frame.columns and pd.notna(row.get("situs_city")):
            in_city = peers[peers["situs_city"] == row["situs_city"]]
            if len(in_city) >= config.MIN_PEER_GROUP_SIZE:
                peers = in_city

        comparison = []
        for column, label, formatter in (
            ("value_per_sqft", "Assessed $/sqft", money),
            ("built_far", "Built FAR", lambda v: "—" if pd.isna(v) else f"{v:.2f}"),
            ("tenure_years", "Tenure (yr)", number),
            ("building_age", "Age (yr)", number),
        ):
            if column not in frame.columns:
                continue
            comparison.append(
                {
                    "Metric": label,
                    "This parcel": formatter(row.get(column)),
                    "Peer median": formatter(
                        pd.to_numeric(peers[column], errors="coerce").median()
                    ),
                }
            )
        st.dataframe(
            pd.DataFrame(comparison), hide_index=True, width="stretch"
        )
        st.caption(f"Peer set: {len(peers):,} parcels.")

    apn = str(row.get("parcel_id", "")).strip()
    if apn:
        st.markdown(
            "**Verify before acting** — "
            f"[ZIMAS](https://zimas.lacity.org/) · "
            f"[LA County Assessor portal](https://portal.assessor.lacounty.gov/parceldetail/{apn})"
        )

    with st.expander("All fields"):
        st.dataframe(
            row.rename("value").to_frame().astype(str),
            width="stretch",
        )


def render_diagnostics(result: pipeline.PipelineResult) -> None:
    st.subheader("Source health")
    table = result.source_table()
    if table.empty:
        st.caption("No live sources were queried (demo mode).")
    else:
        st.dataframe(table, hide_index=True, width="stretch")

    if result.schema is not None:
        st.subheader("Schema resolution")
        left, right = st.columns(2)
        with left:
            st.caption(f"Resolved {len(result.schema.resolved)} fields")
            st.dataframe(
                pd.DataFrame(
                    sorted(result.schema.resolved.items()),
                    columns=["field", "source column"],
                ),
                hide_index=True,
                width="stretch",
            )
        with right:
            if result.schema.missing:
                st.caption(f"Unresolved: {', '.join(result.schema.missing)}")
                for consequence in result.schema.degradations():
                    st.markdown(f"- {consequence}")
            else:
                st.success("Every expected field resolved.")
            if result.schema.unmapped_source_columns:
                with st.expander("Columns present but unused"):
                    st.write(", ".join(result.schema.unmapped_source_columns))

    st.subheader("Cache")
    summary = cache.summary()
    if summary:
        st.dataframe(pd.DataFrame(summary), hide_index=True, width="stretch")
    else:
        st.caption("Cache is empty.")
    if st.button("Clear cache"):
        removed = cache.clear()
        st.success(f"Removed {removed} cached files.")

    if result.notes:
        st.subheader("Run notes")
        for note in result.notes:
            st.markdown(f"- {note}")


# ── Main ─────────────────────────────────────────────────────────────────


def main() -> None:
    st.title("🏢 LA Commercial Real Estate Opportunity Screener")

    settings = sidebar_load_controls()
    if st.sidebar.button("Load data", type="primary", width="stretch"):
        st.session_state["result"] = load_data(settings)

    if "result" not in st.session_state:
        st.session_state["result"] = load_data(settings)

    result: pipeline.PipelineResult = st.session_state["result"]

    if settings["demo"] and result.ok:
        st.warning(
            "**Demo data.** These parcels are synthetic, generated for interface "
            "evaluation. Switch off *Demo data* in the sidebar to screen the real "
            "LA County assessor roll.",
            icon="⚠️",
        )

    if not result.ok:
        st.error("No parcels loaded.")
        for note in result.notes:
            st.markdown(f"- {note}")
        render_diagnostics(result)
        return

    filtered = sidebar_filters(result.parcels)
    st.caption(
        f"{len(filtered):,} parcels after filters, out of {len(result.parcels):,} loaded."
    )

    tabs = st.tabs(
        [
            "Screener",
            "Off-market",
            "Development sites",
            "Shopping center rehab",
            "Rooftop growth",
            "Parcel detail",
            "Diagnostics",
        ]
    )
    with tabs[0]:
        render_screener(filtered, result)
    with tabs[1]:
        render_lens(filtered, "acquisition_score", result, "acquisition")
    with tabs[2]:
        render_lens(filtered, "development_score", result, "development")
    with tabs[3]:
        render_rehab(filtered)
    with tabs[4]:
        render_rooftops(filtered, result)
    with tabs[5]:
        render_detail(filtered, result)
    with tabs[6]:
        render_diagnostics(result)


if __name__ == "__main__":
    main()
