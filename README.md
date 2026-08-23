# market-immune-system

Two Streamlit dashboards:

| App | What it does |
|---|---|
| `app.py` | **Market Immune System** — dual-regime classifier for SPY and BTC |
| `app_cre.py` | **LA CRE Opportunity Screener** — commercial real estate sourcing across LA County |

```bash
pip install -r requirements.txt
streamlit run app.py       # market regime dashboard
streamlit run app_cre.py   # LA CRE screener
```

---

# LA CRE Opportunity Screener

Screens all 88 cities in Los Angeles County for two kinds of commercial real
estate opportunity, using public records as the spine and your MLS feed as
optional enrichment.

**Start here — no setup, no network:**

```bash
streamlit run app_cre.py
```

The app opens in demo mode with synthetic parcels so you can evaluate the
interface immediately. Switch off *Demo data* in the sidebar to pull real
records.

**Before your first real pull**, verify the upstream sources:

```bash
python scripts/probe_sources.py          # add --mls to test your feed too
```

This checks every endpoint, resolves the current dataset IDs, and reports
which assessor columns matched — printing the exact `cre/config.py` edit for
anything that didn't.

## The two lenses

### Off-market acquisitions

Ranks parcels by **ownership tenure signal**. Components: years since the
Prop 13 base-year reset (40%), assessed value per square foot against
same-use/same-city peers (25%), building age (15%), permit dormancy (10%),
and improvement-to-total value ratio (10%).

**Read this carefully.** A low Prop 13 assessment is evidence of a distant
base year, not of a discount. What the score reliably identifies is an owner
who has held for decades. That correlates with below-market in-place rents
(real value-add upside) and with generational transfer events — but it also
implies a large embedded capital gain, which cuts *against* a sale. It is a
sourcing list, not a valuation.

### Development sites

Ranks parcels by **unused floor area**. Components: FAR headroom (40%), lot
size (20%), improvement ratio (20%), transit proximity (12%), and commercial
zoning eligible for housing conversion (8%).

Inside the City of LA the ceiling comes from the zone's height district.
Everywhere else it comes from what comparable neighbors actually built — a
zoning-free measure, which is what makes countywide coverage possible.

### Shopping center rehab

Ranks retail centers as renovate, re-tenant and sell candidates. Components:
building age (28%), years since any substantial improvement (22%), submarket
land values (20%), ownership tenure (15%), and whether the center falls in a
workable size band (15%).

**This one is deliberately inverted from the off-market score.** There, a low
assessed value is the signal. Here you want *expensive dirt under a worn-out
building*. Cheap land beneath a tired center is not a repositioning — it means
the corridor will not support the better tenants the whole thesis depends on.

Two mechanics are worth knowing:

- **Renovation history comes from the assessor's effective year built**, which
  advances when a property is substantially improved. A gap of zero against the
  original year means the building has never been meaningfully renovated —
  the most direct "outdated facilities" signal in public data.
- **Location is scored from submarket land values, not the parcel's own.**
  Prop 13 freezes a parcel's land assessment at its base year, so a long-held
  site in an excellent location looks cheap. Using its own figure as a location
  proxy would rank down exactly the long-tenured centers the screen exists to
  find. The neighborhood median, drawn across parcels of every vintage, does
  not carry that distortion.

Size gates do most of the filtering, because the assessor codes a corner
liquor store and a 90,000 sqft neighborhood center under the same description.
They are adjustable in the tab.

The tab sizes a construction budget from a cost per square foot you supply,
and **projects no returns at all**. That would need in-place rents, a rent
roll, lease expiries and market pricing, none of which exist in public assessor
data — and assessed value is not market value, which is precisely why the
long-held centers at the top carry assessments far below what they would trade
for.

## Data sources and coverage

| Source | Coverage | Auth | Provides |
|---|---|---|---|
| LA County assessor roll | All 2.4M parcels, 88 cities | none | Use type, assessed values, Prop 13 base years, building area, coordinates |
| LA City building permits | City of LA only | none | Permit recency and count |
| LA City zoning / GeoHub | City of LA only | none | Zone string, height district, TOC tier |
| MLS (RESO Web API) | Per your feed | credentials | Active listings, list price |

Roughly 800k of the county's 2.4M parcels sit inside City of LA limits.
Outside them there is no public zoning layer, so entitlement scoring falls
back to the neighbor-relative measure and the `zone` column is null.

## MLS setup

Copy `.env.example` and fill in your feed's details. The adapter speaks the
RESO Web API and supports three auth styles (`bearer`, `oauth2`, `query`),
which covers Bridge, Trestle and direct MLS access.

> **Licensing.** MLS data is licensed, not public. IDX and VOW rules
> generally prohibit republishing it on a public site. Run an MLS-enabled
> build locally and keep it private — do not deploy it to a public URL. Your
> participation agreement governs; where it conflicts with anything here, it
> wins.

MLS commercial coverage in LA is thin — most commercial trades never touch
the MLS — so treat it as enrichment, not inventory. Its most useful function
here is *exclusion*: an active listing means the property is already being
marketed, which is exactly what you want filtered out of an off-market list.

## Design notes

**Dataset IDs drift.** LA County republishes the assessor roll annually under
a new identifier. Pinning one guarantees an annual outage, so each source
carries candidate IDs plus a catalog-search fallback.

**Column names drift too.** Logical fields resolve against alias lists, and
the Diagnostics tab reports exactly what matched. An unresolved field
degrades the feature that needs it and says so, rather than emitting nulls
that look like real zeros.

**Missing data shifts the basis, not the score.** When a component has no
data anywhere, it is dropped and the remaining weights are renormalized — so
a dead permit feed costs you the dormancy signal, not eighty points on every
parcel.

**Nothing here is a zoning determination.** FAR figures are screening
heuristics. LA measures FAR against buildable area rather than gross lot
area, and specific plans, Q/D conditions, CPIOs and overlays are invisible to
this tool. Every parcel detail view links to ZIMAS and the assessor portal.
Verify there before acting.

## Layout

```
app_cre.py              Streamlit dashboard
cre/
  config.py             Dataset IDs, column aliases, FAR table, scoring weights
  transform.py          Schema resolution, normalization, derived metrics
  scoring.py            Acquisition and development scoring, zone parsing
  pipeline.py           Orchestration
  cache.py              Parquet cache with TTL
  demo.py               Synthetic data for the no-network demo mode
  sources/
    socrata.py          SODA client with ID discovery and health reporting
    assessor.py         LA County parcel roll
    permits.py          LA City permits
    zoning.py           ArcGIS zoning + TOC, local shapely spatial join
    mls.py              RESO Web API adapter
scripts/probe_sources.py  Connectivity and schema diagnostics
tests/                    108 tests, no network required
```

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -q
```

The suite is fully offline: scoring math, schema resolution, join keys and
SoQL escaping are unit-tested, and the dashboard itself is rendered through
Streamlit's `AppTest` harness against demo data.
