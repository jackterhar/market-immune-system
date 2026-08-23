"""
Synthetic parcel data.

Exists for two reasons: it lets you evaluate the interface without waiting on
a multi-minute countywide pull, and it makes the UI exercisable in environments
with no network access at all. The distributions are plausible but invented —
nothing here is real property data, and the demo banner says so on every screen.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cre import transform

CITIES = {
    "LOS ANGELES": (34.05, -118.24),
    "LONG BEACH": (33.77, -118.19),
    "PASADENA": (34.15, -118.14),
    "SANTA MONICA": (34.02, -118.48),
    "GLENDALE": (34.14, -118.25),
    "TORRANCE": (33.84, -118.34),
    "POMONA": (34.06, -117.76),
    "BURBANK": (34.18, -118.31),
}

USES = [
    ("Commercial", "Store"),
    ("Commercial", "Office Building"),
    ("Commercial", "Restaurant"),
    ("Commercial", "Shopping Center"),
    ("Industrial", "Warehousing"),
    ("Industrial", "Light Manufacturing"),
    ("Commercial", "Parking Lot"),
    ("Vacant", "Vacant Land"),
]

ZONES = ["C2-1VL", "C2-1", "C4-2", "C1.5-1", "M1-1", "M2-2", "[Q]C2-1", "C4-3"]


def _city_weights(count: int) -> np.ndarray:
    """Weight the parcel mix toward Los Angeles, as the real county roll is."""
    weights = np.full(count, 1.0)
    weights[0] = 6.0  # LOS ANGELES leads the CITIES mapping
    return weights / weights.sum()


def generate(n: int = 4000, seed: int = 11) -> pd.DataFrame:
    """Build a synthetic, already-derived parcel frame."""
    rng = np.random.default_rng(seed)

    city_names = list(CITIES)
    cities = rng.choice(city_names, size=n, p=_city_weights(len(city_names)))
    use_index = rng.integers(0, len(USES), n)
    general = np.array([USES[i][0] for i in use_index])
    specific = np.array([USES[i][1] for i in use_index])

    lat = np.array([CITIES[c][0] for c in cities]) + rng.normal(0, 0.035, n)
    lon = np.array([CITIES[c][1] for c in cities]) + rng.normal(0, 0.045, n)

    lot_sqft = np.round(rng.lognormal(9.2, 0.85, n)).clip(1_500, 900_000)
    far = rng.beta(2, 5, n) * 3.0
    building_sqft = np.round(lot_sqft * far)
    vacant = general == "Vacant"
    building_sqft[vacant] = 0

    # Prop 13 base years cluster: long-held stock plus recent transfers.
    base_year = np.where(
        rng.random(n) < 0.35,
        rng.integers(1975, 1995, n),
        rng.integers(1996, 2025, n),
    ).astype(float)

    year_built = rng.integers(1920, 2022, n).astype(float)
    year_built[vacant] = np.nan

    # Most stock has never been substantially improved; the assessor's
    # effective year built equals the original. About a third has been
    # renovated at some point since.
    renovated = rng.random(n) < 0.32
    effective_year = year_built.copy()
    for i in np.flatnonzero(renovated & ~vacant):
        earliest = int(year_built[i]) + 10
        if earliest < 2024:
            effective_year[i] = rng.integers(earliest, 2025)

    # Assessed value tracks the base year, which is the whole point of the
    # tenure signal: older base years carry structurally lower assessments.
    age_factor = np.interp(base_year, [1975, 2025], [0.22, 1.0])
    land_value = np.round(lot_sqft * rng.uniform(40, 260, n) * age_factor)
    improvement_value = np.round(building_sqft * rng.uniform(60, 320, n) * age_factor)
    improvement_value[vacant] = 0

    # Each city carries a small pool of ZIPs, as real ones do. Random ZIPs per
    # parcel would leave every peer group below the size floor, collapsing the
    # location index to a single global median.
    zip_pools = {
        city: [f"9{1000 + 37 * index + offset}" for offset in range(4)]
        for index, city in enumerate(city_names)
    }
    zips = np.array([rng.choice(zip_pools[city]) for city in cities])

    frame = pd.DataFrame(
        {
            "parcel_id": [f"{5000000000 + i}" for i in range(n)],
            "situs_address": [
                f"{rng.integers(100, 9999)} {name} {suffix}"
                for name, suffix in zip(
                    rng.choice(
                        ["MAIN", "OAK", "SPRING", "FIGUEROA", "PICO", "VERMONT",
                         "SEPULVEDA", "ALAMEDA", "OLYMPIC", "SLAUSON"], n
                    ),
                    rng.choice(["ST", "AVE", "BLVD", "WAY"], n),
                )
            ],
            "situs_city": cities,
            "situs_zip": zips,
            "general_use": general,
            "specific_use": specific,
            "year_built": year_built,
            "effective_year": effective_year,
            "building_sqft": np.where(building_sqft > 0, building_sqft, np.nan),
            "lot_sqft": lot_sqft,
            "land_value": land_value,
            "improvement_value": improvement_value,
            "total_value": land_value + improvement_value,
            "land_base_year": base_year,
            "imp_base_year": base_year,
            "lat": lat,
            "lon": lon,
            "zone": rng.choice(ZONES, n),
            "toc_tier": rng.choice([np.nan, 1, 2, 3, 4], n, p=[0.55, 0.14, 0.14, 0.10, 0.07]),
            "years_since_permit": rng.exponential(9, n).clip(0, 60),
        }
    )

    # Parcels outside the City of LA have no city zoning or TOC overlay.
    outside = frame["situs_city"] != "LOS ANGELES"
    frame.loc[outside, "zone"] = pd.NA
    frame.loc[outside, "toc_tier"] = np.nan

    return transform.add_derived_metrics(frame, roll_year=2025)
