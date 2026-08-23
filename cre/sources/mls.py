"""
MLS enrichment via the RESO Web API (OData).

**Licensing — read before deploying.** MLS data is licensed, not public. IDX
and VOW rules generally prohibit republishing it on a public site, and feeds
are restricted to licensed members and approved vendors. This adapter is built
for a private, local dashboard: run it on your own machine, keep credentials
out of the repository, and do not deploy an MLS-enabled build to a public URL.
Your participation agreement governs; when it conflicts with anything here,
it wins.

Practically, MLS commercial coverage in LA is thin — most commercial trades
never touch the MLS — so this is an *enrichment* over the public parcel spine,
never a replacement for it. What it adds is genuinely useful though: an active
listing on a parcel tells you it is already being marketed, which is exactly
the set you want to exclude when hunting off-market.

Credentials come from the environment (or ``.streamlit/secrets.toml``):

    MLS_BASE_URL       e.g. https://api.bridgedataoutput.com/api/v2/OData/crmls
    MLS_AUTH_MODE      bearer | oauth2 | query
    MLS_TOKEN          for bearer / query modes
    MLS_CLIENT_ID      for oauth2
    MLS_CLIENT_SECRET  for oauth2
    MLS_TOKEN_URL      for oauth2
    MLS_SCOPE          optional, for oauth2
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests

from cre import config
from cre.sources.socrata import SourceResult

REQUEST_TIMEOUT = 60
PAGE_SIZE = 200
MAX_PAGES = 50

_token_cache: dict[str, tuple[str, float]] = {}


@dataclass(frozen=True)
class MlsCredentials:
    base_url: str
    auth_mode: str
    token: str | None = None
    client_id: str | None = None
    client_secret: str | None = None
    token_url: str | None = None
    scope: str | None = None

    @property
    def configured(self) -> bool:
        if not self.base_url:
            return False
        if self.auth_mode in {"bearer", "query"}:
            return bool(self.token)
        if self.auth_mode == "oauth2":
            return bool(self.client_id and self.client_secret and self.token_url)
        return False


def credentials_from_env(getter: Any = None) -> MlsCredentials:
    """Read credentials from the environment, or any dict-like getter."""
    get = getter or (lambda key, default="": os.environ.get(key, default))
    return MlsCredentials(
        base_url=(get("MLS_BASE_URL", "") or "").rstrip("/"),
        auth_mode=(get("MLS_AUTH_MODE", "bearer") or "bearer").lower().strip(),
        token=get("MLS_TOKEN", "") or None,
        client_id=get("MLS_CLIENT_ID", "") or None,
        client_secret=get("MLS_CLIENT_SECRET", "") or None,
        token_url=get("MLS_TOKEN_URL", "") or None,
        scope=get("MLS_SCOPE", "") or None,
    )


def _access_token(creds: MlsCredentials) -> str:
    """Fetch (and briefly cache) an OAuth2 client-credentials token."""
    cache_key = f"{creds.token_url}|{creds.client_id}"
    cached = _token_cache.get(cache_key)
    if cached and cached[1] > time.time() + 60:
        return cached[0]

    payload = {
        "grant_type": "client_credentials",
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
    }
    if creds.scope:
        payload["scope"] = creds.scope

    response = requests.post(creds.token_url, data=payload, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    body = response.json()
    token = body.get("access_token")
    if not token:
        raise RuntimeError("Token endpoint returned no access_token.")
    _token_cache[cache_key] = (token, time.time() + float(body.get("expires_in", 3600)))
    return token


def _request_kwargs(creds: MlsCredentials) -> tuple[dict[str, str], dict[str, str]]:
    """Build (headers, extra query params) for the configured auth mode."""
    headers = {"Accept": "application/json"}
    params: dict[str, str] = {}

    if creds.auth_mode == "bearer":
        headers["Authorization"] = f"Bearer {creds.token}"
    elif creds.auth_mode == "query":
        params["access_token"] = creds.token or ""
    elif creds.auth_mode == "oauth2":
        headers["Authorization"] = f"Bearer {_access_token(creds)}"
    else:
        raise ValueError(f"Unsupported MLS_AUTH_MODE: {creds.auth_mode!r}")

    return headers, params


def build_filter(
    property_types: list[str] | None = None,
    statuses: list[str] | None = None,
    county: str | None = "Los Angeles",
) -> str:
    """Compose an OData $filter for active commercial and income listings."""
    property_types = property_types or config.MLS_PROPERTY_TYPES
    statuses = statuses or config.MLS_ACTIVE_STATUSES

    def escape(value: str) -> str:
        return value.replace("'", "''")

    clauses = []
    if property_types:
        clauses.append(
            "("
            + " or ".join(f"PropertyType eq '{escape(p)}'" for p in property_types)
            + ")"
        )
    if statuses:
        clauses.append(
            "("
            + " or ".join(f"StandardStatus eq '{escape(s)}'" for s in statuses)
            + ")"
        )
    if county:
        clauses.append(f"CountyOrParish eq '{escape(county)}'")
    return " and ".join(clauses)


def fetch_listings(
    creds: MlsCredentials | None = None,
    odata_filter: str | None = None,
    max_records: int = 5000,
) -> tuple[pd.DataFrame, SourceResult]:
    """
    Pull active listings. Returns an empty frame (not an error) when unconfigured.

    Absence of credentials is a normal state, not a failure: the dashboard is
    designed to be fully functional on public records alone.
    """
    creds = creds or credentials_from_env()
    result = SourceResult(frame=pd.DataFrame(), domain=creds.base_url or "mls")

    if not creds.configured:
        result.notes.append(
            "MLS is not configured. Set MLS_BASE_URL and credentials to enable "
            "listing enrichment; every other feature works without it."
        )
        return pd.DataFrame(), result

    try:
        headers, auth_params = _request_kwargs(creds)
    except (ValueError, RuntimeError, requests.RequestException) as exc:
        result.errors.append(f"MLS auth failed: {exc}"[:300])
        return pd.DataFrame(), result

    rows: list[dict[str, Any]] = []
    skip = 0
    url = f"{creds.base_url}/Property"

    for _ in range(MAX_PAGES):
        params = {
            "$filter": odata_filter or build_filter(),
            "$top": min(PAGE_SIZE, max_records - len(rows)),
            "$skip": skip,
        }
        params.update(auth_params)
        try:
            response = requests.get(
                url, params=params, headers=headers, timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            result.errors.append(f"MLS request failed: {exc}"[:300])
            break

        page = payload.get("value", [])
        if not page:
            break
        rows.extend(page)
        if len(page) < params["$top"] or len(rows) >= max_records:
            break
        skip += len(page)

    result.frame = pd.DataFrame(rows)
    result.row_count = len(result.frame)
    if result.row_count:
        result.notes.append(f"Fetched {result.row_count:,} active listings.")
    return result.frame, result


_STREET_SUFFIXES = {
    "STREET": "ST", "AVENUE": "AVE", "BOULEVARD": "BLVD", "ROAD": "RD",
    "DRIVE": "DR", "PLACE": "PL", "COURT": "CT", "LANE": "LN",
    "PARKWAY": "PKWY", "HIGHWAY": "HWY", "TERRACE": "TER",
}


def _normalize_street(value: Any) -> str | None:
    """Reduce a street address to a comparable form: alphanumerics, upper case."""
    if not isinstance(value, str) or not value.strip():
        return None
    cleaned = re.sub(r"[^A-Z0-9 ]", " ", value.upper())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return None
    return " ".join(_STREET_SUFFIXES.get(part, part) for part in cleaned.split())


def _address_key(street: Any, city: Any) -> str | None:
    """
    Build a street+city join key.

    City is not optional. Street addresses repeat constantly across the 88
    cities in LA County — there is a "1st Street" in dozens of them — so a
    street-only key produces confident false matches between unrelated
    parcels. A record without a city is better left unjoined.
    """
    normalized_street = _normalize_street(street)
    normalized_city = _normalize_street(city)
    if not normalized_street or not normalized_city:
        return None
    return f"{normalized_street}|{normalized_city}"


def _split_unparsed(value: Any) -> tuple[Any, Any]:
    """Split "123 Main St, Los Angeles, CA 90012" into (street, city)."""
    if not isinstance(value, str) or "," not in value:
        return value, None
    parts = [p.strip() for p in value.split(",")]
    return parts[0], (parts[1] if len(parts) > 1 else None)


def attach(parcels: pd.DataFrame, listings: pd.DataFrame) -> pd.DataFrame:
    """
    Flag parcels that have an active MLS listing.

    Joins on APN first — RESO's ``ParcelNumber`` is the reliable key — and
    falls back to a normalized street address for records that omit it.
    """
    out = parcels.copy()
    out["mls_listed"] = False
    out["mls_list_price"] = pd.NA
    out["mls_listing_id"] = pd.NA

    if listings.empty or "parcel_id" not in out.columns:
        return out

    price_column = next(
        (c for c in ("ListPrice", "ClosePrice", "OriginalListPrice") if c in listings.columns),
        None,
    )
    id_column = next(
        (c for c in ("ListingId", "ListingKey", "ListingKeyNumeric") if c in listings.columns),
        None,
    )

    matched_by_apn: dict[str, dict[str, Any]] = {}
    matched_by_address: dict[str, dict[str, Any]] = {}

    for _, listing in listings.iterrows():
        record = {
            "price": listing.get(price_column) if price_column else None,
            "listing_id": listing.get(id_column) if id_column else None,
        }
        apn = listing.get("ParcelNumber")
        if isinstance(apn, str) and apn.strip():
            key = re.sub(r"\D", "", apn).zfill(10)
            if len(key) == 10:
                matched_by_apn.setdefault(key, record)
        street = listing.get("StreetNumberNumeric") or listing.get("StreetNumber")
        name = listing.get("StreetName")
        if street and name:
            street_text = f"{street} {name}"
            city_text = listing.get("City")
        else:
            street_text, city_text = _split_unparsed(listing.get("UnparsedAddress"))
            city_text = listing.get("City") or city_text
        address = _address_key(street_text, city_text)
        if address:
            matched_by_address.setdefault(address, record)

    apn_keys = (
        out["parcel_id"].astype("string").str.replace(r"\D", "", regex=True).str.zfill(10)
    )
    if "situs_address" in out.columns and "situs_city" in out.columns:
        address_keys = pd.Series(
            [
                _address_key(street, city)
                for street, city in zip(out["situs_address"], out["situs_city"])
            ],
            index=out.index,
            dtype="object",
        )
    else:
        address_keys = pd.Series(None, index=out.index, dtype="object")

    for label in out.index:
        apn_key = apn_keys.at[label]
        record = matched_by_apn.get(apn_key) if apn_key else None
        if record is None:
            address_key = address_keys.at[label]
            record = matched_by_address.get(address_key) if address_key else None
        if record is None:
            continue
        out.at[label, "mls_listed"] = True
        out.at[label, "mls_list_price"] = record["price"]
        out.at[label, "mls_listing_id"] = record["listing_id"]

    return out
