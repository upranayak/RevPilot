"""Offline tests: no LLM calls. Run: pytest -q"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import db  # noqa: E402
import tools  # noqa: E402


def test_sql_returns_rows():
    out = db.run_readonly("SELECT COUNT(*) AS n FROM transactions")
    assert out["rows"][0][0] > 120_000


@pytest.mark.parametrize("bad", [
    "DROP TABLE transactions",
    "SELECT 1; DROP TABLE transactions",
    "SELECT 'x'; DROP TABLE transactions",
    "INSERT INTO products VALUES (1)",
    "SELECT * FROM read_parquet('secrets.parquet')",
    "COPY transactions TO 'x.csv'",
])
def test_sql_guard_blocks_writes(bad):
    with pytest.raises(ValueError):
        db.run_readonly(bad)


def test_sql_guard_allows_keywords_inside_strings():
    out = db.run_readonly("SELECT COUNT(*) FROM transactions WHERE category = 'Set'")
    assert out["rows"][0][0] > 0


def test_sql_rows_are_capped():
    out = db.run_readonly("SELECT * FROM transactions")
    assert out["row_count"] == 50 and out["truncated"]


def test_money_is_inr_and_cancellations_excluded():
    out = db.run_readonly("SELECT SUM(revenue) FROM transactions WHERE is_cancelled")
    assert out["rows"][0][0] == 0


def test_search_finds_kurta_set():
    res = tools.search_products("blue kurta set", k=3)["results"]
    assert res[0]["description"] == "Blue Kurta Set"


def test_search_handles_typos():
    res = tools.search_products("pnik kurta", k=3)["results"]
    assert res[0]["description"] == "Pink Kurta"


def test_forecast_tool():
    code = db.run_readonly("SELECT style FROM forecasts LIMIT 1")["rows"][0][0]
    out = tools.get_forecast(code)
    assert len(out["days"]) == 14 and out["total_units_14d"] >= 0


def test_simulation_is_consistent():
    code = db.run_readonly("SELECT style FROM elasticity WHERE reliable LIMIT 1")["rows"][0][0]
    out = tools.simulate_price_change(code, [-0.1, 0.0, 0.1])
    zero = [s for s in out["scenarios"] if s["pct_change"] == 0.0][0]
    assert abs(zero["revenue_delta_vs_base_inr"]) < 1e-6
    # With elastic demand (< -1), a price cut should raise revenue.
    if out["elasticity"] < -1:
        cut = [s for s in out["scenarios"] if s["pct_change"] == -0.1][0]
        assert cut["revenue_delta_vs_base_inr"] > 0


def test_tool_errors_are_returned_not_raised():
    result, ok = tools.call("run_sql", '{"query": "DELETE FROM products"}')
    assert not ok and "error" in result
    result, ok = tools.call("nope", "{}")
    assert not ok
