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
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cre import cache, config, demo, pipeline, scoring  # noqa: E402
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
        result = pipeline.PipelineResult(parcels=parcels)
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
        result.notes = ["Demo data — synthetic parcels, not real records."] + notes
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


def sidebar_filters(parcels: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filters")
    mask = pd.Series(True, index=parcels.index)

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
        "Colour runs red (low) to green (high); circle size tracks lot area."
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
            "total_value": st.column_config.NumberColumn("Assessed", format="$%,d"),
            "value_per_sqft": st.column_config.NumberColumn("$/sqft", format="$%.0f"),
            "improvement_ratio": st.column_config.NumberColumn("Imp %", format="%.0%%"),
            "years_since_permit": st.column_config.NumberColumn(
                "Permit gap (yr)", format="%.1f"
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
    columns[2].metric("Assessed value", money(row.get("total_value")))
    columns[3].metric("Lot", number(row.get("lot_sqft"), " sqft"))

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
        ["Screener", "Off-market", "Development sites", "Parcel detail", "Diagnostics"]
    )
    with tabs[0]:
        render_screener(filtered, result)
    with tabs[1]:
        render_lens(filtered, "acquisition_score", result, "acquisition")
    with tabs[2]:
        render_lens(filtered, "development_score", result, "development")
    with tabs[3]:
        render_detail(filtered, result)
    with tabs[4]:
        render_diagnostics(result)


if __name__ == "__main__":
    main()
