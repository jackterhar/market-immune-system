#!/usr/bin/env python3
"""
Verify every upstream data source and report exactly what needs fixing.

Run this first, before the dashboard. It answers the questions that decide
whether the app will work on your machine:

  * Is each endpoint reachable at all?
  * Which dataset identifier resolved, and how?
  * Which of the expected assessor columns actually exist in this roll year?
  * Do the City of LA zoning and transit layers resolve?
  * Are MLS credentials configured and accepted?

Any unresolved column is printed with the exact config edit that fixes it.

    python scripts/probe_sources.py
    python scripts/probe_sources.py --mls
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests  # noqa: E402

from cre import config, transform  # noqa: E402
from cre.sources import mls, socrata, zoning  # noqa: E402

OK = "\033[92m✓\033[0m"
BAD = "\033[91m✗\033[0m"
WARN = "\033[93m!\033[0m"


def heading(text: str) -> None:
    print(f"\n\033[1m{text}\033[0m")
    print("─" * len(text))


def check_reachability() -> None:
    heading("Endpoint reachability")
    targets = [
        ("LA County open data", f"https://{config.LA_COUNTY_DOMAIN}/api/catalog/v1?limit=1"),
        ("LA City open data", f"https://{config.LA_CITY_DOMAIN}/api/catalog/v1?limit=1"),
        ("LA City GIS", config.ARCGIS_SERVICES["zoning"][0] + "?f=json"),
        ("LA County GIS", config.LA_COUNTY_PARCEL_GIS + "?f=json"),
    ]
    for label, url in targets:
        try:
            response = requests.get(url, timeout=30)
            mark = OK if response.status_code == 200 else WARN
            print(f"  {mark} {label}: HTTP {response.status_code}")
        except requests.RequestException as exc:
            print(f"  {BAD} {label}: {type(exc).__name__}")
            print(f"      {str(exc)[:160]}")


def check_socrata(source: config.SocrataSource) -> str | None:
    heading(f"Socrata dataset — {source.name}")
    dataset_id, log = socrata.discover_dataset_id(source)
    for line in log:
        print(f"    · {line}")
    if dataset_id is None:
        print(f"  {BAD} No usable dataset ID.")
        print(
            f"      Fix: find the current dataset on https://{source.domain} and add "
            f"its ID to {source.name.upper()}.candidate_ids in cre/config.py"
        )
        return None
    print(f"  {OK} Resolved: {dataset_id}")
    return dataset_id


def check_assessor_schema(dataset_id: str) -> None:
    heading("Assessor column resolution")
    try:
        columns = socrata.probe_schema(config.LA_COUNTY_DOMAIN, dataset_id)
    except socrata.SocrataError as exc:
        print(f"  {BAD} Schema probe failed: {exc}")
        return

    print(f"  Dataset exposes {len(columns)} columns.")
    import pandas as pd

    report = transform.resolve_columns(pd.DataFrame(columns=columns))

    print(f"  {OK} Resolved {len(report.resolved)} of {len(config.COLUMN_ALIASES)} fields.")
    for field, source_column in sorted(report.resolved.items()):
        print(f"      {field:24s} → {source_column}")

    if report.missing:
        print(f"\n  {WARN} Unresolved fields — each costs a feature:")
        for field in report.missing:
            print(f"      {field}")
        for consequence in report.degradations():
            print(f"        · {consequence}")
        print("\n  Fix: pick the right column from the list below and add it to")
        print("  COLUMN_ALIASES[<field>] in cre/config.py.")
        print(f"\n  Unused columns in this dataset:\n      {', '.join(report.unmapped_source_columns)}")

    if not report.ok:
        print(f"\n  {BAD} Required fields missing — the app cannot run until this is fixed.")


def check_arcgis() -> None:
    heading("ArcGIS layers (City of LA)")
    if not zoning.SHAPELY_AVAILABLE:
        print(f"  {WARN} shapely not installed — zoning join disabled.")
        print("      Fix: pip install shapely")
    for label, services in config.ARCGIS_SERVICES.items():
        needle = label if label != "toc" else "transit"
        resolved = False
        for service_url in services:
            layer_url, log = zoning.discover_layer(service_url, needle)
            for line in log:
                print(f"    · {label}: {line}")
            if layer_url:
                print(f"  {OK} {label}: {layer_url}")
                resolved = True
                break
        if not resolved:
            print(f"  {BAD} {label}: no layer resolved.")
            print(
                f"      Fix: browse the service catalog, then update "
                f"ARCGIS_SERVICES['{label}'] in cre/config.py"
            )


def check_mls() -> None:
    heading("MLS (RESO Web API)")
    creds = mls.credentials_from_env()
    if not creds.configured:
        print(f"  {WARN} Not configured — this is fine; the app works without it.")
        print("      To enable, set: MLS_BASE_URL, MLS_AUTH_MODE, and credentials.")
        return
    print(f"  Base URL: {creds.base_url}  (auth: {creds.auth_mode})")
    frame, result = mls.fetch_listings(creds, max_records=5)
    if result.errors:
        for error in result.errors:
            print(f"  {BAD} {error}")
        return
    print(f"  {OK} Returned {len(frame)} listings.")
    if not frame.empty:
        has_apn = "ParcelNumber" in frame.columns and frame["ParcelNumber"].notna().any()
        mark = OK if has_apn else WARN
        print(f"  {mark} ParcelNumber present: {has_apn}"
              f"{'' if has_apn else ' — joins will fall back to street+city matching.'}")
        print(f"      Fields available: {', '.join(list(frame.columns)[:15])}…")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mls", action="store_true", help="also test the MLS feed")
    parser.add_argument("--skip-arcgis", action="store_true")
    args = parser.parse_args()

    print("\033[1mLA CRE screener — source probe\033[0m")

    check_reachability()

    assessor_id = check_socrata(config.ASSESSOR_PARCELS)
    if assessor_id:
        check_assessor_schema(assessor_id)

    check_socrata(config.LA_CITY_PERMITS)

    if not args.skip_arcgis:
        check_arcgis()

    if args.mls:
        check_mls()

    print("\nDone. Fix anything marked ✗ before running the dashboard.")
    return 0 if assessor_id else 1


if __name__ == "__main__":
    raise SystemExit(main())
