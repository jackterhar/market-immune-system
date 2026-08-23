"""
End-to-end checks that the dashboard actually renders.

Runs against demo data, so these need no network and are safe in CI. They
catch the class of bug unit tests miss entirely: duplicate widget IDs, bad
column configs, and formatting calls that blow up on null values.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "app_cre.py")


@pytest.fixture(scope="module")
def app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    return at


def test_app_runs_without_exception(app: AppTest):
    assert not app.exception, [str(e.value) for e in app.exception]


def test_app_renders_all_tabs(app: AppTest):
    assert len(app.tabs) == 6


def test_app_reports_no_errors(app: AppTest):
    assert [e.value for e in app.error] == []


def test_demo_mode_is_labelled_as_synthetic(app: AppTest):
    assert any("Demo data" in w.value for w in app.warning)


def test_screener_metrics_are_populated(app: AppTest):
    labels = {m.label: m.value for m in app.metric}
    assert labels.get("Parcels", "0") != "0"
    assert "Median acquisition" in labels


def test_scores_are_percentile_centered(app: AppTest):
    """Percentile scoring should put the median near 50 on a broad sample."""
    labels = {m.label: m.value for m in app.metric}
    median = float(labels["Median acquisition"])
    assert 40 <= median <= 60
