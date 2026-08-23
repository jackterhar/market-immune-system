"""
Socrata (SODA 2.1) client.

Two things make this more than a thin requests wrapper:

1. **Identifier drift.** LA County republishes the assessor roll annually under
   a new four-by-four ID. Pinning one ID guarantees the app breaks on a
   schedule. This client tries configured candidates, then falls back to
   searching the domain's catalog by name.

2. **Honest failure.** Every fetch returns a ``SourceResult`` carrying the
   diagnostics — which identifier resolved, how many rows came back, what
   errored. The dashboard surfaces this instead of rendering an empty table
   that looks like "no opportunities found".
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Iterator

import pandas as pd
import requests

from cre import config

PAGE_SIZE = 50_000
MAX_RETRIES = 4
BACKOFF_BASE_SECONDS = 2
REQUEST_TIMEOUT = 90


class SocrataError(RuntimeError):
    """Raised when a Socrata request fails in a way retries cannot fix."""


@dataclass
class SourceResult:
    """Data plus the diagnostics needed to explain an empty or partial result."""

    frame: pd.DataFrame
    dataset_id: str | None = None
    domain: str | None = None
    row_count: int = 0
    truncated: bool = False
    errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.frame.empty and not self.errors

    @property
    def status(self) -> str:
        if self.ok:
            return "ok"
        if self.errors and self.frame.empty:
            return "failed"
        if self.frame.empty:
            return "empty"
        return "partial"


def _headers() -> dict[str, str]:
    headers = {"Accept": "application/json", "User-Agent": "la-cre-screener/0.1"}
    token = os.environ.get(config.SOCRATA_APP_TOKEN_ENV, "").strip()
    if token:
        headers["X-App-Token"] = token
    return headers


def _get(url: str, params: dict[str, Any] | None = None) -> requests.Response:
    """GET with exponential backoff on transient failures."""
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(
                url, params=params, headers=_headers(), timeout=REQUEST_TIMEOUT
            )
        except requests.RequestException as exc:
            last_error = exc
        else:
            if response.status_code == 200:
                return response
            # 4xx other than rate limiting will not improve with retries.
            if response.status_code == 429 or response.status_code >= 500:
                last_error = SocrataError(
                    f"HTTP {response.status_code} from {url}: {response.text[:200]}"
                )
            else:
                raise SocrataError(
                    f"HTTP {response.status_code} from {url}: {response.text[:300]}"
                )
        if attempt < MAX_RETRIES - 1:
            time.sleep(BACKOFF_BASE_SECONDS * (2**attempt))
    raise SocrataError(f"Request to {url} failed after {MAX_RETRIES} attempts: {last_error}")


def discover_dataset_id(source: config.SocrataSource) -> tuple[str | None, list[str]]:
    """
    Resolve a usable dataset identifier for ``source``.

    Tries configured candidates first (cheap HEAD-ish probe of one row), then
    the domain catalog. Returns the identifier and a log of what was tried, so
    the diagnostics tab can explain how it got there.
    """
    log: list[str] = []

    for candidate in source.candidate_ids:
        url = f"https://{source.domain}/resource/{candidate}.json"
        try:
            _get(url, {"$limit": 1})
        except (SocrataError, requests.RequestException) as exc:
            log.append(f"candidate {candidate}: {type(exc).__name__} {exc}"[:200])
            continue
        log.append(f"candidate {candidate}: ok")
        return candidate, log

    log.append(f"all candidates failed; searching catalog for {source.catalog_query!r}")
    try:
        response = _get(
            f"https://{source.domain}/api/catalog/v1",
            {"q": source.catalog_query, "only": "dataset", "limit": 20},
        )
        results = response.json().get("results", [])
    except (SocrataError, requests.RequestException, ValueError) as exc:
        log.append(f"catalog search failed: {exc}"[:200])
        return None, log

    # Prefer the most recently updated dataset whose name matches the query.
    query_terms = source.catalog_query.lower().split()
    scored: list[tuple[str, str, str]] = []
    for item in results:
        resource = item.get("resource", {})
        name = (resource.get("name") or "").lower()
        if not all(term in name for term in query_terms):
            continue
        scored.append(
            (
                resource.get("updatedAt") or "",
                resource.get("id") or "",
                resource.get("name") or "",
            )
        )
    if not scored:
        log.append("catalog search returned no name match")
        return None, log

    scored.sort(reverse=True)
    _, dataset_id, name = scored[0]
    log.append(f"catalog resolved {dataset_id} ({name})")
    return dataset_id, log


def probe_schema(domain: str, dataset_id: str) -> list[str]:
    """Return the actual column names of a dataset, for schema diagnostics."""
    response = _get(
        f"https://{domain}/resource/{dataset_id}.json", {"$limit": 1}
    )
    rows = response.json()
    if not rows:
        return []
    return sorted(rows[0].keys())


def _paged_rows(
    domain: str,
    dataset_id: str,
    where: str | None,
    select: str | None,
    max_rows: int | None,
    order_by: str = ":id",
) -> Iterator[list[dict[str, Any]]]:
    """
    Yield pages of rows.

    Ordering by ``:id`` matters: without a stable sort, offset paging on
    Socrata can silently skip or duplicate rows between pages.
    """
    offset = 0
    while True:
        remaining = None if max_rows is None else max_rows - offset
        if remaining is not None and remaining <= 0:
            return
        limit = PAGE_SIZE if remaining is None else min(PAGE_SIZE, remaining)

        params: dict[str, Any] = {
            "$limit": limit,
            "$offset": offset,
            "$order": order_by,
        }
        if where:
            params["$where"] = where
        if select:
            params["$select"] = select

        rows = _get(f"https://{domain}/resource/{dataset_id}.json", params).json()
        if not rows:
            return
        yield rows
        if len(rows) < limit:
            return
        offset += len(rows)


def fetch(
    source: config.SocrataSource,
    where: str | None = None,
    select: str | None = None,
    max_rows: int | None = None,
    dataset_id: str | None = None,
    progress: Any = None,
) -> SourceResult:
    """
    Fetch a dataset into a DataFrame.

    ``progress`` may be any callable taking (rows_so_far, dataset_id); it lets
    the Streamlit layer show a live row count during the multi-minute
    countywide pull without this module importing streamlit.
    """
    result = SourceResult(frame=pd.DataFrame(), domain=source.domain)

    if dataset_id is None:
        dataset_id, log = discover_dataset_id(source)
        result.notes.extend(log)
    if dataset_id is None:
        result.errors.append(
            f"Could not resolve a dataset ID for {source.name} on {source.domain}. "
            "The dataset may have been renamed or retired; set an explicit ID in "
            "cre/config.py."
        )
        return result
    result.dataset_id = dataset_id

    chunks: list[pd.DataFrame] = []
    rows_seen = 0
    try:
        for page in _paged_rows(source.domain, dataset_id, where, select, max_rows):
            chunks.append(pd.DataFrame(page))
            rows_seen += len(page)
            if progress is not None:
                progress(rows_seen, dataset_id)
    except (SocrataError, requests.RequestException, ValueError) as exc:
        result.errors.append(f"{type(exc).__name__}: {exc}"[:400])
        if chunks:
            result.notes.append(
                f"Returning {rows_seen:,} rows fetched before the failure."
            )
            result.truncated = True

    if chunks:
        result.frame = pd.concat(chunks, ignore_index=True)
    result.row_count = len(result.frame)

    if max_rows is not None and result.row_count >= max_rows:
        result.truncated = True
        result.notes.append(f"Stopped at the {max_rows:,}-row cap.")

    return result
