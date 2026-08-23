"""
Configuration for the LA CRE opportunity screener.

Everything that is likely to drift over time — dataset identifiers, column
names, zoning constants, scoring weights — lives here rather than being
buried in code, so it can be corrected without touching logic.

IMPORTANT: dataset identifiers on Socrata change when agencies publish a new
annual roll. Rather than pinning a single ID, each source carries a list of
candidates plus a catalog search term; the client tries candidates in order
and falls back to catalog discovery. See cre.sources.socrata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ── Cache ────────────────────────────────────────────────────────────────
CACHE_DIR = Path(__file__).resolve().parent.parent / ".cre_cache"
CACHE_TTL_HOURS = 24 * 7  # parcel rolls update annually; a week is generous

# ── Socrata endpoints ────────────────────────────────────────────────────
LA_COUNTY_DOMAIN = "data.lacounty.gov"
LA_CITY_DOMAIN = "data.lacity.org"

# Optional. Raises rate limits from ~1000/hr to effectively unlimited.
# Set via env var SOCRATA_APP_TOKEN. Not required.
SOCRATA_APP_TOKEN_ENV = "SOCRATA_APP_TOKEN"


@dataclass(frozen=True)
class SocrataSource:
    """A Socrata dataset with fallbacks for when identifiers drift."""

    name: str
    domain: str
    candidate_ids: tuple[str, ...]
    catalog_query: str
    description: str = ""


# The LA County assessor publishes one dataset per roll year. Candidates are
# ordered newest-first; unknown/retired IDs are skipped automatically. If all
# fail, catalog discovery finds the current one by name.
ASSESSOR_PARCELS = SocrataSource(
    name="assessor_parcels",
    domain=LA_COUNTY_DOMAIN,
    candidate_ids=(
        "aprh-mnnu",
        "hpws-rjis",
        "9trm-uz8i",
    ),
    catalog_query="Assessor Parcels Data",
    description="LA County assessor secured property roll, all ~2.4M parcels.",
)

LA_CITY_PERMITS = SocrataSource(
    name="la_city_permits",
    domain=LA_CITY_DOMAIN,
    candidate_ids=("pi9x-tg5x", "yv23-pmwf"),
    catalog_query="Building and Safety Permit Information",
    description="City of LA building permits. City parcels only.",
)

# ── ArcGIS / GeoHub (City of LA only) ────────────────────────────────────
# Zoning and transit overlays exist only for the City of LA. County parcels
# outside city limits fall back to peer-relative underbuilt scoring.
ARCGIS_SERVICES = {
    "zoning": [
        "https://maps.lacity.org/lahub/rest/services/Boundaries/MapServer",
        "https://public.gis.lacity.org/arcgis/rest/services/Zoning/MapServer",
    ],
    "toc": [
        "https://maps.lacity.org/lahub/rest/services/Planning/MapServer",
    ],
}

# LA County parcel geometry — used only as a lot-size fallback when the
# assessor roll does not carry a lot square footage column.
LA_COUNTY_PARCEL_GIS = (
    "https://public.gis.lacounty.gov/public/rest/services/"
    "LACounty_Dynamic/Parcel/MapServer"
)

# ── Assessor column resolution ───────────────────────────────────────────
# Socrata column names vary between roll years. Each logical field maps to
# candidate source columns, tried in order. Resolution is reported by the
# schema probe so unmatched fields are visible rather than silently null.
COLUMN_ALIASES: dict[str, list[str]] = {
    "parcel_id": ["ain", "apn", "assessorid", "parcelid", "asmt_no"],
    "situs_address": [
        "situsaddress",
        "situsfulladdress",
        "propertylocation",
        "situs_address",
        "address",
    ],
    "situs_city": ["situscity", "situs_city", "city"],
    "situs_zip": ["situszip", "situszip5", "situs_zip", "zipcode", "zip"],
    "general_use": ["generalusetype", "general_use_type", "usetype"],
    "specific_use": [
        "specificusetype",
        "specific_use_type",
        "usecodedescchar1",
    ],
    "use_code": ["usecode", "propertyusecode", "usecode1"],
    "year_built": ["yearbuilt", "yr_built", "year_built"],
    "effective_year": ["effectiveyearbuilt", "effyearbuilt"],
    "building_sqft": ["sqftmain", "sqft_main", "buildingsqft", "improvementsqft"],
    "lot_sqft": ["sqftlot", "lotsqft", "landsqft", "parcelsqft", "shape_area"],
    "units": ["units", "unitsn", "numberofunits"],
    "land_value": ["roll_landvalue", "rolllandvalue", "landvalue"],
    "improvement_value": ["roll_impvalue", "rollimpvalue", "improvementvalue"],
    "total_value": ["roll_totlandimp", "rolltotlandimp", "totalvalue"],
    "land_base_year": ["roll_landbaseyear", "rolllandbaseyear", "landbaseyear"],
    "imp_base_year": ["roll_impbaseyear", "rollimpbaseyear", "impbaseyear"],
    "lat": ["center_lat", "latitude", "lat"],
    "lon": ["center_lon", "longitude", "lon", "lng"],
    "homeowner_exemption": ["roll_homeownersexemp", "homeownersexemption"],
    "roll_year": ["rollyear", "roll_year", "taxyear"],
    "is_taxable": ["istaxableparcel", "taxable"],
}

# Fields without which the app cannot function.
REQUIRED_FIELDS = ["parcel_id", "general_use"]

# ── Use-type filtering ───────────────────────────────────────────────────
# The assessor's `generalusetype` buckets. We keep commercial-adjacent ones.
CRE_GENERAL_USE_TYPES = [
    "Commercial",
    "Industrial",
    "Recreational",
    "Institutional",
]

# Vacant land is carried separately: irrelevant for acquisitions, central
# for development site hunting.
VACANT_USE_TYPES = ["Vacant"]

# Small multifamily straddles residential and commercial underwriting.
INCOME_RESIDENTIAL_USE_TYPES = ["Residential"]
INCOME_RESIDENTIAL_MIN_UNITS = 5

# ── LA City zoning: floor area ratio by height district ──────────────────
# LAMC 12.21.1. INDICATIVE ONLY — real entitlement depends on specific plans,
# overlays (Q/D/T conditions), CPIOs, and the fact that LA measures FAR
# against *buildable* area (lot net of setbacks and dedications), not gross
# lot area. Treat every FAR figure downstream as a screening heuristic, never
# as a zoning determination. Verify against ZIMAS before acting.
HEIGHT_DISTRICT_FAR: dict[str, float] = {
    "1": 3.0,
    "1L": 3.0,
    "1VL": 3.0,
    "1XL": 3.0,
    "1SS": 3.0,
    "2": 6.0,
    "3": 10.0,
    "4": 13.0,
}
DEFAULT_FAR_IF_UNKNOWN = 3.0

# Buildable area is typically well below gross lot area. Applying a haircut
# keeps the "underbuilt" signal conservative rather than flattering.
BUILDABLE_AREA_FACTOR = 0.85

# ── Scoring ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AcquisitionWeights:
    """
    Weights for the off-market acquisition score.

    This score identifies LONG-TENURED OWNERSHIP, not underpriced real estate.
    A low assessed value under Prop 13 is a consequence of a distant base
    year, not evidence of a discount. Long tenure correlates with below-market
    in-place rents (genuine value-add upside) and with generational transfer
    events — but it also implies a large embedded capital gain, which is a
    disincentive to sell. Read it as a sourcing signal, not a valuation.
    """

    tenure: float = 0.40           # years since Prop 13 base-year reset
    basis_gap: float = 0.25        # assessed $/sqft vs. peer median
    building_age: float = 0.15     # deferred capex / retrofit exposure
    permit_dormancy: float = 0.10  # no recent permit activity
    improvement_ratio: float = 0.10  # underimproved relative to land value


@dataclass(frozen=True)
class DevelopmentWeights:
    """Weights for the development / entitlement site score."""

    far_gap: float = 0.40          # unused floor area vs. permitted or peers
    lot_size: float = 0.20         # larger sites carry projects
    improvement_ratio: float = 0.20  # low improvement value = cheap teardown
    transit: float = 0.12          # TOC tier proximity bonus
    corridor_upzone: float = 0.08  # commercial zoning eligible for housing


ACQUISITION_WEIGHTS = AcquisitionWeights()
DEVELOPMENT_WEIGHTS = DevelopmentWeights()

@dataclass(frozen=True)
class RehabWeights:
    """
    Weights for the shopping center rehabilitation score.

    The thesis is specific: buy a physically tired retail center on a corridor
    that can support better tenants, renovate, re-tenant, sell. That makes this
    score deliberately *unlike* the acquisition score. There, a low assessed
    value is the signal. Here you want expensive dirt under a worn-out
    building — a tired center in a weak location is not a value-add play, it is
    just a bad center.
    """

    obsolescence: float = 0.28        # building age
    renovation_dormancy: float = 0.22  # years since any substantial improvement
    location_quality: float = 0.20     # submarket land values (see note below)
    under_management: float = 0.15     # ownership tenure
    rehab_scale: float = 0.15          # building size within a workable band


REHAB_WEIGHTS = RehabWeights()

# Retail use types, matched as keywords against the assessor's specific use
# description so the filter survives wording changes between roll years.
RETAIL_USE_KEYWORDS = [
    "store",
    "shopping",
    "retail",
    "market",
    "restaurant",
    "strip",
    "commercial center",
    "department",
]

# Size gates defining a shopping center rather than a single storefront.
# Exposed as sliders in the UI; these are only the defaults.
REHAB_MIN_LOT_SQFT = 15_000
REHAB_MIN_BUILDING_SQFT = 5_000
REHAB_MAX_BUILDING_SQFT = 400_000

# The scale band. Centers below the sweet spot are too small to carry the soft
# costs of a repositioning; above it you are competing with institutional
# capital for regional malls.
REHAB_SWEET_SPOT_SQFT = (15_000, 150_000)

# Surface parking share below which a center reads as auto-era and may have
# room for a pad building. Coverage = building area / lot area.
REHAB_LOW_COVERAGE_THRESHOLD = 0.25

# Location quality is measured from the *submarket's* median land value per lot
# square foot, not the subject parcel's own. Under Prop 13 the subject's land
# assessment is frozen at its base year, so using it directly would penalize
# exactly the long-held parcels this score exists to surface. The neighborhood
# median, drawn across parcels of every vintage, is not distorted that way.
# Ordered broadest-first on purpose: peer_median walks key *prefixes*, so
# ["city", "zip"] groups by city+zip and then falls back to city alone. The
# reverse order would fall back from zip to zip — never widening — and the
# whole component would collapse to a global median wherever zips are thin.
LOCATION_PEER_KEYS = ["situs_city", "situs_zip"]


# Peer groups for relative comparisons. A parcel is compared against others
# of the same specific use type in the same city; falls back to county-wide
# use-type medians when a peer group is too thin to be meaningful.
PEER_GROUP_KEYS = ["situs_city", "specific_use"]
MIN_PEER_GROUP_SIZE = 8

CURRENT_ROLL_YEAR_FALLBACK = 2025

# ── Rooftop growth ───────────────────────────────────────────────────────
# New multifamily is a demand signal for retail: rooftops arrive before the
# tenants who serve them. Measured at area level, not parcel level.

# How far back to count a delivery as "new".
NEW_MF_LOOKBACK_YEARS = 8

# Units at or above which a residential parcel counts as multifamily rather
# than a duplex or a house with a granny flat.
MIN_MF_UNITS = 5

# Areas are ZIP codes, falling back to city where ZIP is unavailable. ZIP is
# the smallest geography the assessor roll carries consistently, and it is
# roughly the catchment of a neighborhood shopping center.
AREA_KEYS = ["situs_zip", "situs_city"]

# Cap on the targeted multifamily pull. Countywide new multifamily is tens of
# thousands of parcels, not millions, so this is generous.
NEW_MF_MAX_ROWS = 150_000

# ── MLS (RESO Web API) ───────────────────────────────────────────────────
# Credentials never live in this file. See cre/sources/mls.py.
MLS_PROPERTY_TYPES = [
    "Commercial Sale",
    "Commercial Lease",
    "Business Opportunity",
    "Residential Income",
]
MLS_ACTIVE_STATUSES = ["Active", "Active Under Contract", "Coming Soon"]
