"""
Tests for the pure logic of the CRE screener.

These deliberately avoid the network: every LA data endpoint is exercised
through structural code paths, so the parts that can be verified offline
(schema resolution, derived metrics, scoring math, join keys) actually are.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cre import config, pipeline, scoring, transform  # noqa: E402
from cre.sources import assessor, mls, permits  # noqa: E402


# ── Schema resolution ────────────────────────────────────────────────────


def test_resolve_columns_is_case_and_separator_insensitive():
    frame = pd.DataFrame(columns=["AIN", "SQFTmain", "Roll_LandValue", "CENTER_LAT"])
    report = transform.resolve_columns(frame)
    assert report.resolved["parcel_id"] == "AIN"
    assert report.resolved["building_sqft"] == "SQFTmain"
    assert report.resolved["land_value"] == "Roll_LandValue"
    assert report.resolved["lat"] == "CENTER_LAT"


def test_resolve_columns_reports_missing_and_unmapped():
    frame = pd.DataFrame(columns=["AIN", "GeneralUseType", "SomethingElse"])
    report = transform.resolve_columns(frame)
    assert "lot_sqft" in report.missing
    assert "SomethingElse" in report.unmapped_source_columns
    assert report.ok  # both required fields present


def test_schema_report_not_ok_without_required_fields():
    report = transform.resolve_columns(pd.DataFrame(columns=["Irrelevant"]))
    assert not report.ok
    assert report.degradations()  # explains the consequences


def test_first_matching_alias_wins():
    # 'situsaddress' precedes 'propertylocation' in the alias list.
    frame = pd.DataFrame(columns=["PropertyLocation", "SitusAddress"])
    report = transform.resolve_columns(frame)
    assert report.resolved["situs_address"] == "SitusAddress"


# ── Normalization ────────────────────────────────────────────────────────


def test_normalize_blanks_sentinel_zeros():
    frame = pd.DataFrame(
        {
            "AIN": ["1", "2"],
            "GeneralUseType": ["Commercial", "Commercial"],
            "YearBuilt": ["0", "1962"],
            "SQFTmain": ["0", "4200"],
        }
    )
    out, _ = transform.normalize(frame)
    assert pd.isna(out.loc[0, "year_built"])
    assert out.loc[1, "year_built"] == 1962
    assert pd.isna(out.loc[0, "building_sqft"])


def test_normalize_rejects_coordinates_outside_la():
    frame = pd.DataFrame(
        {
            "AIN": ["1", "2", "3"],
            "GeneralUseType": ["Commercial"] * 3,
            "CENTER_LAT": ["34.05", "0", "40.71"],
            "CENTER_LON": ["-118.24", "0", "-74.00"],
        }
    )
    out, _ = transform.normalize(frame)
    assert out.loc[0, "lat"] == pytest.approx(34.05)
    assert pd.isna(out.loc[1, "lat"])  # 0/0 null island
    assert pd.isna(out.loc[2, "lat"])  # New York


# ── Derived metrics ──────────────────────────────────────────────────────


def _base_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "parcel_id": ["a", "b", "c"],
            "general_use": ["Commercial", "Commercial", "Vacant"],
            "building_sqft": [10_000.0, 5_000.0, np.nan],
            "lot_sqft": [20_000.0, 0.0, 15_000.0],
            "total_value": [1_000_000.0, 500_000.0, 300_000.0],
            "land_value": [600_000.0, 400_000.0, 300_000.0],
            "improvement_value": [400_000.0, 100_000.0, 0.0],
            "land_base_year": [1978.0, 2019.0, np.nan],
            "imp_base_year": [1985.0, 2019.0, np.nan],
            "year_built": [1955.0, 2005.0, np.nan],
        }
    )


def test_derived_metrics_never_produce_infinities():
    out = transform.add_derived_metrics(_base_frame(), roll_year=2025)
    for column in ("value_per_sqft", "built_far", "improvement_ratio",
                   "land_value_per_lot_sqft"):
        assert not np.isinf(out[column].astype(float)).any(), column


def test_zero_lot_size_yields_null_far_not_infinity():
    out = transform.add_derived_metrics(_base_frame(), roll_year=2025)
    assert pd.isna(out.loc[1, "built_far"])  # lot_sqft was 0
    assert out.loc[0, "built_far"] == pytest.approx(0.5)


def test_tenure_uses_latest_base_year():
    out = transform.add_derived_metrics(_base_frame(), roll_year=2025)
    # Improvement base year 1985 is later than land's 1978, so tenure is 40.
    assert out.loc[0, "tenure_years"] == pytest.approx(40.0)
    assert out.loc[1, "tenure_years"] == pytest.approx(6.0)
    assert pd.isna(out.loc[2, "tenure_years"])


def test_negative_tenure_and_age_are_discarded():
    frame = _base_frame()
    frame.loc[0, "land_base_year"] = 2099.0
    frame.loc[0, "imp_base_year"] = 2099.0
    frame.loc[0, "year_built"] = 2099.0
    out = transform.add_derived_metrics(frame, roll_year=2025)
    assert pd.isna(out.loc[0, "tenure_years"])
    assert pd.isna(out.loc[0, "building_age"])


def test_missing_columns_degrade_to_null_not_exception():
    minimal = pd.DataFrame({"parcel_id": ["a"], "general_use": ["Commercial"]})
    out = transform.add_derived_metrics(minimal, roll_year=2025)
    assert pd.isna(out.loc[0, "built_far"])
    assert pd.isna(out.loc[0, "tenure_years"])


# ── Peer medians ─────────────────────────────────────────────────────────


def test_peer_median_falls_back_when_group_too_thin():
    frame = pd.DataFrame(
        {
            "situs_city": ["LA"] * 10 + ["POMONA"],
            "specific_use": ["Store"] * 10 + ["Store"],
            "value_per_sqft": [100.0] * 10 + [999.0],
        }
    )
    medians = transform.peer_median(frame, "value_per_sqft", min_group_size=8)
    # The 10-parcel LA group is its own peer set.
    assert medians.iloc[0] == pytest.approx(100.0)
    # Pomona has one parcel, below the floor, so it falls back to a broader set.
    assert medians.iloc[10] != pytest.approx(999.0)


def test_peer_median_without_group_keys_uses_global():
    frame = pd.DataFrame({"value_per_sqft": [10.0, 20.0, 30.0]})
    medians = transform.peer_median(frame, "value_per_sqft", group_keys=[])
    assert medians.unique().tolist() == [pytest.approx(20.0)]


def test_peer_median_missing_column_returns_nulls():
    frame = pd.DataFrame({"situs_city": ["LA"]})
    medians = transform.peer_median(frame, "nonexistent")
    assert medians.isna().all()


# ── Percentile ranking ───────────────────────────────────────────────────


def test_percentile_rank_inverts():
    series = pd.Series([1.0, 2.0, 3.0, 4.0])
    normal = scoring.percentile_rank(series)
    inverted = scoring.percentile_rank(series, invert=True)
    assert normal.iloc[0] < normal.iloc[3]
    assert inverted.iloc[0] > inverted.iloc[3]
    assert (normal + inverted).round(6).eq(1.0).all()


def test_percentile_rank_all_null_returns_null():
    assert scoring.percentile_rank(pd.Series([np.nan, np.nan])).isna().all()


def test_percentile_rank_preserves_nulls():
    ranked = scoring.percentile_rank(pd.Series([1.0, np.nan, 3.0]))
    assert pd.isna(ranked.iloc[1])


# ── Zone parsing ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "zone,expected",
    [
        ("[Q]C2-1VL-CDO", "1VL"),
        ("(T)(Q)C4-2D-SN", "2"),
        ("C2-1", "1"),
        ("R3-1XL", "1XL"),
        ("M2-3", "3"),
        ("C1.5-1VL", "1VL"),
        ("bogus", None),
        ("", None),
        (None, None),
        (123, None),
    ],
)
def test_parse_height_district(zone, expected):
    assert scoring.parse_height_district(zone) == expected


def test_max_far_matches_height_district_table():
    assert scoring.max_far_for_zone("C2-2") == config.HEIGHT_DISTRICT_FAR["2"]
    assert scoring.max_far_for_zone("C2-4") == config.HEIGHT_DISTRICT_FAR["4"]
    assert scoring.max_far_for_zone("nonsense") is None


@pytest.mark.parametrize(
    "zone,commercial",
    [
        ("C2-1", True),
        ("[Q]C4-2", True),
        ("M1-1", True),
        ("CM-1", True),
        ("R3-1", False),
        ("PF-1", False),
        ("A1-1", False),
        (None, False),
    ],
)
def test_is_commercial_zone(zone, commercial):
    assert scoring.is_commercial_zone(zone) is commercial


# ── Score combination ────────────────────────────────────────────────────


def test_combine_renormalizes_when_components_missing():
    index = pd.RangeIndex(3)
    components = {
        "a": pd.Series([1.0, 1.0, 1.0], index=index),
        "b": pd.Series([np.nan] * 3, index=index),
    }
    result = scoring._combine(components, {"a": 0.25, "b": 0.75}, index)
    # 'b' contributes nothing, so 'a' carries the full weight and a perfect
    # 'a' must still score 100 rather than 25.
    assert result.scores.tolist() == [100.0, 100.0, 100.0]
    assert result.skipped == ["b"]
    assert result.used_weights == {"a": 1.0}
    assert result.notes


def test_combine_treats_individual_gaps_as_neutral():
    index = pd.RangeIndex(2)
    components = {"a": pd.Series([1.0, np.nan], index=index)}
    result = scoring._combine(components, {"a": 1.0}, index)
    assert result.scores.tolist() == [100.0, 50.0]


def test_combine_with_no_usable_components_returns_nulls():
    index = pd.RangeIndex(2)
    result = scoring._combine({"a": pd.Series([np.nan, np.nan], index=index)},
                              {"a": 1.0}, index)
    assert result.scores.isna().all()
    assert result.notes


def test_weights_sum_to_one():
    acquisition = config.ACQUISITION_WEIGHTS
    development = config.DEVELOPMENT_WEIGHTS
    assert sum(vars(acquisition).values()) == pytest.approx(1.0)
    assert sum(vars(development).values()) == pytest.approx(1.0)


# ── FAR headroom ─────────────────────────────────────────────────────────


def test_far_gap_uses_zoned_ceiling_with_buildable_haircut():
    frame = pd.DataFrame(
        {
            "zone": ["C2-2"],  # 6.0 FAR
            "built_far": [1.0],
            "general_use": ["Commercial"],
            "situs_city": ["LOS ANGELES"],
            "specific_use": ["Store"],
        }
    )
    gap, max_far, _ = scoring.far_gap(frame)
    expected_ceiling = 6.0 * config.BUILDABLE_AREA_FACTOR
    assert max_far.iloc[0] == pytest.approx(expected_ceiling)
    assert gap.iloc[0] == pytest.approx((expected_ceiling - 1.0) / expected_ceiling)


def test_far_gap_is_bounded_when_overbuilt():
    frame = pd.DataFrame(
        {"zone": ["C2-1"], "built_far": [99.0], "general_use": ["Commercial"]}
    )
    gap, _, _ = scoring.far_gap(frame)
    assert gap.iloc[0] == 0.0  # already exceeds the ceiling; no headroom


def test_far_gap_treats_vacant_land_as_full_headroom():
    frame = pd.DataFrame(
        {
            "zone": [None],
            "built_far": [np.nan],
            "general_use": ["Vacant"],
            "situs_city": ["LOS ANGELES"],
            "specific_use": ["Vacant Land"],
        }
    )
    gap, _, _ = scoring.far_gap(frame)
    assert gap.iloc[0] == 1.0


def test_far_gap_falls_back_to_neighbors_without_zoning():
    frame = pd.DataFrame(
        {
            "built_far": [0.1, 1.0, 1.0, 1.0, 1.0],
            "general_use": ["Commercial"] * 5,
            "situs_city": ["POMONA"] * 5,
            "specific_use": ["Store"] * 5,
        }
    )
    gap, max_far, notes = scoring.far_gap(frame)
    assert max_far.notna().all()
    assert gap.iloc[0] > gap.iloc[1]  # the underbuilt parcel ranks higher
    assert any("neighbor-built" in note for note in notes)


# ── End-to-end scoring ───────────────────────────────────────────────────


def _screening_frame(n: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    frame = pd.DataFrame(
        {
            "parcel_id": [f"{i:010d}" for i in range(n)],
            "general_use": ["Commercial"] * n,
            "specific_use": ["Store"] * n,
            "situs_city": ["LOS ANGELES"] * n,
            "building_sqft": rng.integers(2_000, 30_000, n).astype(float),
            "lot_sqft": rng.integers(5_000, 60_000, n).astype(float),
            "total_value": rng.integers(400_000, 9_000_000, n).astype(float),
            "land_value": rng.integers(200_000, 5_000_000, n).astype(float),
            "improvement_value": rng.integers(50_000, 4_000_000, n).astype(float),
            "land_base_year": rng.integers(1975, 2024, n).astype(float),
            "imp_base_year": rng.integers(1975, 2024, n).astype(float),
            "year_built": rng.integers(1920, 2020, n).astype(float),
        }
    )
    return transform.add_derived_metrics(frame, roll_year=2025)


def test_acquisition_scores_are_bounded_and_ranked():
    frame = _screening_frame()
    result = scoring.acquisition_score(frame)
    assert result.scores.between(0, 100).all()
    assert result.scores.nunique() > 1  # actually discriminates


def test_development_scores_are_bounded():
    frame = _screening_frame()
    result = scoring.development_score(frame)
    assert result.scores.between(0, 100).all()


def test_longest_held_parcel_scores_above_newest():
    frame = _screening_frame()
    result = scoring.acquisition_score(frame)
    longest = frame["tenure_years"].idxmax()
    newest = frame["tenure_years"].idxmin()
    assert result.scores[longest] > result.scores[newest]


def test_scoring_survives_a_single_row():
    frame = _screening_frame(1)
    assert scoring.acquisition_score(frame).scores.notna().all()
    assert scoring.development_score(frame).scores.notna().all()


def test_scoring_survives_an_empty_frame():
    empty = transform.add_derived_metrics(
        pd.DataFrame({"parcel_id": pd.Series(dtype="string"),
                      "general_use": pd.Series(dtype="string")})
    )
    assert scoring.acquisition_score(empty).scores.empty


def test_explain_produces_readable_reasons():
    frame = _screening_frame()
    frame["zone"] = "C2-2"
    result = scoring.acquisition_score(frame)
    reasons = scoring.explain(frame, frame.index[0], result.components)
    assert reasons
    assert any("Held" in reason for reason in reasons)


# ── SoQL construction ────────────────────────────────────────────────────


def test_build_where_escapes_single_quotes():
    where = assessor.build_where("UseType", ["O'Brien"], "City", ["L'A"])
    assert "O''Brien" in where
    assert "L''A" in where


def test_build_where_omits_city_clause_when_unfiltered():
    where = assessor.build_where("UseType", ["Commercial"])
    assert "City" not in where
    assert where == "UseType in ('Commercial')"


# ── Permit joins ─────────────────────────────────────────────────────────


def test_build_ain_zero_pads_each_component():
    frame = pd.DataFrame(
        {"assessor_book": ["5"], "assessor_page": ["12"], "assessor_parcel": ["3"]}
    )
    assert permits.build_ain(frame).iloc[0] == "0005012003"


def test_build_ain_returns_nulls_when_columns_absent():
    assert permits.build_ain(pd.DataFrame({"x": [1]})).isna().all()


def test_attach_permits_computes_years_since():
    parcels = pd.DataFrame({"parcel_id": ["0005012003", "1111111111"]})
    recency = pd.DataFrame(
        {
            "parcel_id": ["0005012003"],
            "last_permit_date": [pd.Timestamp.now() - pd.Timedelta(days=730)],
            "permit_count": [3],
        }
    )
    out = permits.attach(parcels, recency)
    assert out.loc[0, "years_since_permit"] == pytest.approx(2.0, abs=0.05)
    assert pd.isna(out.loc[1, "years_since_permit"])


def test_attach_permits_with_empty_recency_is_safe():
    parcels = pd.DataFrame({"parcel_id": ["0005012003"]})
    out = permits.attach(parcels, pd.DataFrame())
    assert "years_since_permit" in out.columns
    assert out["years_since_permit"].isna().all()


# ── MLS joins ────────────────────────────────────────────────────────────


def test_address_key_requires_a_city():
    assert mls._address_key("123 Main St", None) is None
    assert mls._address_key("123 Main St", "Pasadena") == "123 MAIN ST|PASADENA"


def test_address_key_normalizes_street_suffixes():
    assert mls._address_key("123 Main Street", "LA") == mls._address_key("123 MAIN ST", "la")


def test_mls_attach_does_not_match_same_street_in_another_city():
    parcels = pd.DataFrame(
        {
            "parcel_id": ["1111111111", "2222222222"],
            "situs_address": ["9 Oak Ave", "9 Oak Ave"],
            "situs_city": ["PASADENA", "LONG BEACH"],
        }
    )
    listings = pd.DataFrame(
        {
            "ParcelNumber": [None],
            "ListPrice": [800_000],
            "ListingId": ["B2"],
            "UnparsedAddress": ["9 Oak Avenue, Pasadena, CA 91101"],
            "City": ["Pasadena"],
        }
    )
    out = mls.attach(parcels, listings)
    assert bool(out.loc[0, "mls_listed"]) is True
    assert bool(out.loc[1, "mls_listed"]) is False


def test_mls_attach_prefers_apn_over_address():
    parcels = pd.DataFrame(
        {
            "parcel_id": ["5001002003"],
            "situs_address": ["123 Main St"],
            "situs_city": ["LOS ANGELES"],
        }
    )
    listings = pd.DataFrame(
        {
            "ParcelNumber": ["5001-002-003"],
            "ListPrice": [2_500_000],
            "ListingId": ["A1"],
            "UnparsedAddress": ["999 Elsewhere Blvd, Burbank, CA"],
            "City": ["Burbank"],
        }
    )
    out = mls.attach(parcels, listings)
    assert bool(out.loc[0, "mls_listed"]) is True
    assert out.loc[0, "mls_listing_id"] == "A1"


def test_mls_attach_without_listings_is_safe():
    parcels = pd.DataFrame({"parcel_id": ["1"], "situs_address": ["x"], "situs_city": ["y"]})
    out = mls.attach(parcels, pd.DataFrame())
    assert out["mls_listed"].eq(False).all()


def test_mls_unconfigured_credentials_are_not_an_error():
    creds = mls.credentials_from_env(lambda key, default="": default)
    assert not creds.configured
    frame, result = mls.fetch_listings(creds)
    assert frame.empty
    assert not result.errors  # absence is a normal state
    assert result.notes


def test_mls_filter_escapes_quotes():
    assert "O''Brien" in mls.build_filter(property_types=["O'Brien"])


# ── Pipeline helpers ─────────────────────────────────────────────────────


def test_build_use_types_deduplicates():
    types = pipeline.build_use_types(True, True, True)
    assert len(types) == len(set(types))
    assert "Vacant" in types


def test_build_use_types_can_exclude_everything():
    assert pipeline.build_use_types(False, False, False) == []


def test_display_frame_only_returns_present_columns():
    frame = pd.DataFrame({"parcel_id": ["a"], "irrelevant": [1]})
    out = pipeline.display_frame(frame)
    assert list(out.columns) == ["parcel_id"]


# ── Band scoring ─────────────────────────────────────────────────────────


def test_band_score_plateaus_across_the_sweet_spot():
    values = pd.Series([15_000.0, 60_000.0, 150_000.0])
    scored = scoring.band_score(values, 5_000, 15_000, 150_000, 400_000)
    assert scored.tolist() == [1.0, 1.0, 1.0]


def test_band_score_is_zero_outside_the_band():
    values = pd.Series([1_000.0, 500_000.0])
    scored = scoring.band_score(values, 5_000, 15_000, 150_000, 400_000)
    assert scored.tolist() == [0.0, 0.0]


def test_band_score_tapers_rather_than_ranking():
    """A regional mall must not outrank a neighborhood center on scale."""
    values = pd.Series([60_000.0, 300_000.0])
    scored = scoring.band_score(values, 5_000, 15_000, 150_000, 400_000)
    assert scored.iloc[0] > scored.iloc[1]


def test_band_score_handles_degenerate_edges():
    values = pd.Series([10.0, 50.0])
    scored = scoring.band_score(values, 10, 10, 50, 50)
    assert scored.notna().all()


def test_band_score_preserves_nulls():
    scored = scoring.band_score(pd.Series([np.nan]), 1, 2, 3, 4)
    assert scored.isna().all()


# ── Retail identification ────────────────────────────────────────────────


@pytest.mark.parametrize(
    "use,retail",
    [
        ("Store", True),
        ("Shopping Center (Neighborhood)", True),
        ("Restaurant", True),
        ("Supermarket", True),
        ("Warehousing", False),
        ("Office Building", False),
        ("Light Manufacturing", False),
        (None, False),
        ("", False),
        (42, False),
    ],
)
def test_is_retail_use(use, retail):
    assert scoring.is_retail_use(use) is retail


def test_shopping_center_gates_exclude_small_storefronts():
    frame = pd.DataFrame(
        {
            "specific_use": ["Store", "Store", "Warehousing"],
            "lot_sqft": [2_000.0, 40_000.0, 80_000.0],
            "building_sqft": [1_200.0, 30_000.0, 50_000.0],
        }
    )
    out = scoring.shopping_center_candidates(frame)
    # Only the mid-size retail parcel survives: the first is too small, the
    # third is not retail.
    assert len(out) == 1
    assert out.index.tolist() == [1]


def test_shopping_center_gates_exclude_regional_malls():
    frame = pd.DataFrame(
        {
            "specific_use": ["Shopping Center"],
            "lot_sqft": [2_000_000.0],
            "building_sqft": [900_000.0],
        }
    )
    assert scoring.shopping_center_candidates(frame).empty


def test_shopping_center_gates_are_overridable():
    frame = pd.DataFrame(
        {
            "specific_use": ["Store"],
            "lot_sqft": [3_000.0],
            "building_sqft": [2_000.0],
        }
    )
    assert scoring.shopping_center_candidates(frame).empty
    widened = scoring.shopping_center_candidates(
        frame, min_lot_sqft=1_000, min_building_sqft=1_000
    )
    assert len(widened) == 1


# ── Renovation history ───────────────────────────────────────────────────


def test_renovation_gap_detects_never_improved_buildings():
    frame = pd.DataFrame(
        {
            "parcel_id": ["a", "b"],
            "general_use": ["Commercial"] * 2,
            "year_built": [1972.0, 1972.0],
            "effective_year": [1972.0, 2015.0],
        }
    )
    out = transform.add_derived_metrics(frame, roll_year=2025)
    assert out.loc[0, "renovation_gap"] == 0.0
    assert out.loc[1, "renovation_gap"] == 43.0


def test_years_since_improvement_uses_the_later_of_the_two_years():
    frame = pd.DataFrame(
        {
            "parcel_id": ["a", "b"],
            "general_use": ["Commercial"] * 2,
            "year_built": [1972.0, 1972.0],
            "effective_year": [1972.0, 2015.0],
        }
    )
    out = transform.add_derived_metrics(frame, roll_year=2025)
    assert out.loc[0, "years_since_improvement"] == 53.0
    assert out.loc[1, "years_since_improvement"] == 10.0


def test_renovation_metrics_degrade_without_effective_year():
    frame = pd.DataFrame(
        {
            "parcel_id": ["a"],
            "general_use": ["Commercial"],
            "year_built": [1972.0],
        }
    )
    out = transform.add_derived_metrics(frame, roll_year=2025)
    assert pd.isna(out.loc[0, "renovation_gap"])
    assert out.loc[0, "years_since_improvement"] == 53.0


# ── Location quality ─────────────────────────────────────────────────────


def test_location_index_uses_submarket_median_not_own_assessment():
    """
    The whole point: a long-held parcel's own land assessment is frozen by
    Prop 13, so a great location can look cheap. The index must reflect the
    neighborhood, not the subject.
    """
    frame = pd.DataFrame(
        {
            "situs_zip": ["90402"] * 10,
            "situs_city": ["SANTA MONICA"] * 10,
            # One long-held parcel assessed far below its neighbors.
            "land_value_per_lot_sqft": [30.0] + [500.0] * 9,
        }
    )
    index = transform.location_quality_index(frame)
    # The under-assessed parcel inherits its neighborhood's value, not its own.
    assert index.iloc[0] > 100.0
    assert index.iloc[0] == index.iloc[1]


def test_location_index_without_land_values_returns_nulls():
    frame = pd.DataFrame({"situs_zip": ["90001"]})
    assert transform.location_quality_index(frame).isna().all()


# ── Rehab scoring ────────────────────────────────────────────────────────


def _retail_frame(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    frame = pd.DataFrame(
        {
            "parcel_id": [f"{i:010d}" for i in range(n)],
            "general_use": ["Commercial"] * n,
            "specific_use": ["Shopping Center"] * n,
            "situs_city": ["LOS ANGELES"] * n,
            "situs_zip": rng.choice(["90001", "90210", "91401"], n),
            "building_sqft": rng.integers(8_000, 200_000, n).astype(float),
            "lot_sqft": rng.integers(20_000, 500_000, n).astype(float),
            "year_built": rng.integers(1950, 2015, n).astype(float),
            "effective_year": rng.integers(1950, 2020, n).astype(float),
            "land_value": rng.integers(500_000, 20_000_000, n).astype(float),
            "improvement_value": rng.integers(100_000, 9_000_000, n).astype(float),
            "total_value": rng.integers(1_000_000, 30_000_000, n).astype(float),
            "land_base_year": rng.integers(1975, 2024, n).astype(float),
            "imp_base_year": rng.integers(1975, 2024, n).astype(float),
        }
    )
    frame["effective_year"] = frame[["effective_year", "year_built"]].max(axis=1)
    return transform.add_derived_metrics(frame, roll_year=2025)


def test_rehab_scores_are_bounded_and_discriminate():
    result = scoring.rehab_score(_retail_frame())
    assert result.scores.between(0, 100).all()
    assert result.scores.nunique() > 1


def test_rehab_rewards_expensive_land_under_a_tired_building():
    """
    The inversion that separates this from the acquisition score: identical
    worn-out centers, one neighborhood on expensive dirt, one on cheap. The
    expensive one must rank higher — cheap land means the corridor cannot carry
    the tenants the whole thesis depends on.

    Each neighborhood needs enough parcels to clear the peer-group floor;
    below it the location index falls back to a global median and stops
    discriminating, which is correct behavior but untestable at n=2.
    """
    per_zip = config.MIN_PEER_GROUP_SIZE + 2
    rows = []
    for zip_code, city, land_psf in (
        ("90001", "LOS ANGELES", 10.0),
        ("90402", "SANTA MONICA", 400.0),
    ):
        for i in range(per_zip):
            rows.append(
                {
                    "parcel_id": f"{zip_code}{i:04d}",
                    "general_use": "Commercial",
                    "specific_use": "Shopping Center",
                    "situs_city": city,
                    "situs_zip": zip_code,
                    "building_sqft": 50_000.0,
                    "lot_sqft": 200_000.0,
                    "year_built": 1968.0,
                    "effective_year": 1968.0,
                    "land_value": land_psf * 200_000.0,
                    "improvement_value": 1_000_000.0,
                    "total_value": land_psf * 200_000.0 + 1_000_000.0,
                    "land_base_year": 1980.0,
                    "imp_base_year": 1980.0,
                }
            )
    enriched = transform.add_derived_metrics(pd.DataFrame(rows), roll_year=2025)
    scores = scoring.rehab_score(enriched).scores

    cheap = scores.iloc[:per_zip].mean()
    expensive = scores.iloc[per_zip:].mean()
    assert expensive > cheap


def test_constant_components_are_dropped_not_diluted():
    """
    A component identical for every row conveys no ranking information. It must
    be skipped and its weight redistributed, otherwise it drags every score
    toward the middle and mutes the components that do discriminate.
    """
    index = pd.RangeIndex(4)
    components = {
        "varies": pd.Series([0.0, 0.33, 0.67, 1.0], index=index),
        "constant": pd.Series([0.5] * 4, index=index),
    }
    result = scoring._combine(components, {"varies": 0.5, "constant": 0.5}, index)
    assert result.skipped == ["constant"]
    assert result.used_weights == {"varies": 1.0}
    # With the constant dropped, the spread is full rather than halved.
    assert result.scores.iloc[0] == 0.0
    assert result.scores.iloc[3] == 100.0


def test_single_row_keeps_constant_components():
    """A one-row frame is constant by definition; it must still score."""
    index = pd.RangeIndex(1)
    result = scoring._combine(
        {"a": pd.Series([1.0], index=index)}, {"a": 1.0}, index
    )
    assert result.scores.notna().all()


def test_rehab_flags_identify_never_renovated_and_excess_parking():
    frame = pd.DataFrame(
        {
            "parcel_id": ["a", "b"],
            "general_use": ["Commercial"] * 2,
            "specific_use": ["Shopping Center"] * 2,
            "situs_city": ["LOS ANGELES"] * 2,
            "renovation_gap": [0.0, 30.0],
            "built_far": [0.18, 0.90],
            "improvement_per_sqft": [10.0, 200.0],
        }
    )
    flags = scoring.rehab_flags(frame)
    assert bool(flags.loc[0, "never_renovated"]) is True
    assert bool(flags.loc[1, "never_renovated"]) is False
    assert bool(flags.loc[0, "excess_parking"]) is True
    assert bool(flags.loc[1, "excess_parking"]) is False


def test_rehab_survives_a_single_candidate():
    frame = _retail_frame(1)
    assert scoring.rehab_score(frame).scores.notna().all()


def test_rehab_candidates_scores_after_filtering():
    """
    Percentiles must be computed within the retail set, so the top-ranked
    center scores near 100 even when it is unremarkable countywide.
    """
    parcels = pd.concat([_retail_frame(30), _retail_frame(30)], ignore_index=True)
    parcels.loc[:29, "specific_use"] = "Warehousing"
    candidates, result = pipeline.rehab_candidates(parcels)
    assert not candidates.empty
    assert (candidates["specific_use"] == "Shopping Center").all()
    assert candidates["rehab_score"].max() > 70


def test_rehab_candidates_with_no_matches_returns_empty_not_error():
    frame = pd.DataFrame(
        {
            "specific_use": ["Warehousing"],
            "lot_sqft": [50_000.0],
            "building_sqft": [30_000.0],
        }
    )
    candidates, result = pipeline.rehab_candidates(frame)
    assert candidates.empty
    assert result.notes


def test_rehab_weights_sum_to_one():
    assert sum(vars(config.REHAB_WEIGHTS).values()) == pytest.approx(1.0)


def test_peer_median_fallback_broadens_with_prefix_ordering():
    """
    peer_median walks key *prefixes*, so keys must be ordered broadest-first
    for the fallback to widen. Ordering them narrowest-first makes every
    fallback level equally narrow, and thin groups collapse to a global median
    instead of a city-level one.
    """
    frame = pd.DataFrame(
        {
            "situs_city": ["SANTA MONICA"] * 10 + ["POMONA"] * 10,
            # Each ZIP holds 2 parcels — well below the floor — so grouping
            # must fall back to city, not to the whole frame.
            "situs_zip": [f"9040{i // 2}" for i in range(10)]
            + [f"9176{i // 2}" for i in range(10)],
            "land_value_per_lot_sqft": [500.0] * 10 + [20.0] * 10,
        }
    )
    broad_first = transform.peer_median(
        frame, "land_value_per_lot_sqft", group_keys=["situs_city", "situs_zip"]
    )
    # Santa Monica keeps its own level; it does not collapse toward Pomona.
    assert broad_first.iloc[0] == pytest.approx(500.0)
    assert broad_first.iloc[10] == pytest.approx(20.0)


def test_location_index_config_is_ordered_broadest_first():
    assert config.LOCATION_PEER_KEYS[0] == "situs_city"


def test_location_index_discriminates_on_demo_data():
    """Regression: the component was silently dropped for lack of variation."""
    from cre import demo

    parcels = demo.generate(1500)
    index = transform.location_quality_index(parcels)
    assert index.nunique() > 1


def test_rehab_uses_every_component_on_demo_data():
    """All five inputs should contribute; a skipped one means a data problem."""
    from cre import demo

    candidates, result = pipeline.rehab_candidates(demo.generate(3000))
    assert not candidates.empty
    assert result.skipped == []
    assert set(result.used_weights) == {
        "obsolescence",
        "renovation_dormancy",
        "location_quality",
        "under_management",
        "rehab_scale",
    }


# ── Market value estimation ──────────────────────────────────────────────

from cre import valuation  # noqa: E402


def _market_frame(n_recent: int = 40, n_held: int = 20) -> pd.DataFrame:
    """
    Recent transfers assessed at their true price, plus long-held parcels
    assessed at a fraction of it — the Prop 13 pattern the estimator exists
    to correct for.
    """
    true_psf = 300.0
    rows = []
    for i in range(n_recent):
        sqft = 10_000.0 + i * 100
        rows.append(
            {
                "parcel_id": f"R{i:04d}",
                "situs_city": "LOS ANGELES",
                "specific_use": "Store",
                "building_sqft": sqft,
                "lot_sqft": sqft * 4,
                "total_value": sqft * true_psf,
                "land_base_year": 2023.0,
                "imp_base_year": 2023.0,
            }
        )
    for i in range(n_held):
        sqft = 10_000.0 + i * 100
        rows.append(
            {
                "parcel_id": f"H{i:04d}",
                "situs_city": "LOS ANGELES",
                "specific_use": "Store",
                "building_sqft": sqft,
                "lot_sqft": sqft * 4,
                # Held since 1978, assessed at a quarter of market.
                "total_value": sqft * true_psf * 0.25,
                "land_base_year": 1978.0,
                "imp_base_year": 1978.0,
            }
        )
    return transform.add_derived_metrics(pd.DataFrame(rows), roll_year=2025)


def test_recent_transfer_mask_uses_land_base_year():
    """
    A sale reassesses land; construction reassesses only improvements. Using
    the improvement year would mistake a renovation for a purchase.
    """
    frame = pd.DataFrame(
        {
            "land_base_year": [2023.0, 1975.0],
            "imp_base_year": [2023.0, 2022.0],  # second is new construction
        }
    )
    mask = valuation.recent_transfer_mask(frame, roll_year=2025, lookback_years=5)
    assert bool(mask.iloc[0]) is True
    assert bool(mask.iloc[1]) is False


def test_recent_transfer_mask_excludes_future_base_years():
    frame = pd.DataFrame({"land_base_year": [2099.0]})
    assert not valuation.recent_transfer_mask(frame, roll_year=2025).iloc[0]


def test_estimate_reproduces_price_for_recently_traded_parcels():
    """The core sanity check: a recent transfer's assessment IS its price."""
    frame = _market_frame()
    result = valuation.estimate_market_value(frame, roll_year=2025)
    recent = frame["land_base_year"] == 2023.0
    ratio = result.estimate[recent] / frame.loc[recent, "total_value"]
    assert ratio.median() == pytest.approx(1.0, abs=0.05)


def test_estimate_exceeds_assessed_value_for_long_held_parcels():
    """The Prop 13 correction: long holds are assessed well below market."""
    frame = _market_frame()
    result = valuation.estimate_market_value(frame, roll_year=2025)
    held = frame["land_base_year"] == 1978.0
    ratio = result.estimate[held] / frame.loc[held, "total_value"]
    assert ratio.median() == pytest.approx(4.0, rel=0.15)


def test_estimate_returns_nothing_when_comps_are_too_few():
    frame = pd.DataFrame(
        {
            "parcel_id": ["a", "b"],
            "situs_city": ["LOS ANGELES"] * 2,
            "specific_use": ["Store"] * 2,
            "building_sqft": [10_000.0, 12_000.0],
            "lot_sqft": [40_000.0, 48_000.0],
            "total_value": [3_000_000.0, 3_600_000.0],
            "land_base_year": [1975.0, 1980.0],
        }
    )
    result = valuation.estimate_market_value(frame, roll_year=2025)
    assert result.estimate.isna().all()
    assert result.coverage == 0.0
    assert any("too few" in note for note in result.notes)


def test_estimate_discards_implausible_price_per_foot():
    """A data error must not poison the benchmark median."""
    frame = _market_frame()
    frame.loc[0, "total_value"] = 1e12  # absurd
    result = valuation.estimate_market_value(frame, roll_year=2025)
    recent = frame["land_base_year"] == 2023.0
    ratio = (result.estimate[recent] / (frame.loc[recent, "building_sqft"] * 300.0))
    assert ratio.median() == pytest.approx(1.0, abs=0.1)


def test_estimate_uses_lot_basis_for_land_dominant_parcels():
    frame = _market_frame()
    vacant = pd.DataFrame(
        [
            {
                "parcel_id": "V0001",
                "situs_city": "LOS ANGELES",
                "specific_use": "Store",
                "building_sqft": np.nan,
                "lot_sqft": 40_000.0,
                "total_value": np.nan,
                "land_base_year": 1990.0,
                "imp_base_year": 1990.0,
            }
        ]
    )
    combined = transform.add_derived_metrics(
        pd.concat([frame, vacant], ignore_index=True), roll_year=2025
    )
    result = valuation.estimate_market_value(combined, roll_year=2025)
    assert result.basis.iloc[-1] == "lot sqft"
    assert result.estimate.iloc[-1] > 0


def test_estimate_reports_comp_counts():
    result = valuation.estimate_market_value(_market_frame(), roll_year=2025)
    assert (result.comp_count[result.estimate.notna()] > 0).all()


def test_attach_adds_columns_and_ratio():
    out, result = valuation.attach(_market_frame(), roll_year=2025)
    for column in ("estimated_value", "value_comp_count", "value_basis",
                   "assessed_to_estimate_ratio"):
        assert column in out.columns
    held = out["land_base_year"] == 1978.0
    assert out.loc[held, "assessed_to_estimate_ratio"].median() > 2.0


def test_attach_without_assessed_value_degrades_cleanly():
    frame = pd.DataFrame({"parcel_id": ["a"], "situs_city": ["LA"]})
    out, result = valuation.attach(frame)
    assert out["estimated_value"].isna().all()
    assert result.notes


def test_valuation_never_produces_infinities():
    frame = _market_frame()
    frame.loc[0, "building_sqft"] = 0.0
    frame.loc[1, "lot_sqft"] = 0.0
    out, _ = valuation.attach(frame, roll_year=2025)
    assert not np.isinf(out["estimated_value"].astype(float).fillna(0)).any()


def test_budget_filter_on_estimate_differs_from_assessed():
    """
    The whole reason this module exists: the two bases select different
    parcels, so filtering on assessed value would return the wrong list.
    """
    out, _ = valuation.attach(_market_frame(), roll_year=2025)
    by_assessed = set(
        out.index[out["total_value"].between(1e6, 4e6)]
    )
    by_estimate = set(
        out.index[out["estimated_value"].between(1e6, 4e6)]
    )
    assert by_assessed != by_estimate
