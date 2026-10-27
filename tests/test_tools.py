"""Offline tests: no LLM calls. Run: pytest -q"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import db  # noqa: E402
import tools  # noqa: E402


def test_sql_returns_rows():
    out = db.run_readonly("SELECT COUNT(*) AS n FROM transactions")
    assert out["rows"][0][0] > 1_000_000


@pytest.mark.parametrize("bad", [
    "DROP TABLE transactions",
    "SELECT 1; DROP TABLE transactions",
    "INSERT INTO products VALUES (1)",
    "SELECT * FROM read_parquet('secrets.parquet')",
    "COPY transactions TO 'x.csv'",
])
def test_sql_guard_blocks_writes(bad):
    with pytest.raises(ValueError):
        db.run_readonly(bad)


def test_sql_rows_are_capped():
    out = db.run_readonly("SELECT * FROM transactions")
    assert out["row_count"] == 50 and out["truncated"]


def test_search_finds_lantern():
    res = tools.search_products("white metal lantern", k=3)["results"]
    assert any("LANTERN" in r["description"] for r in res)


def test_search_handles_paraphrase():
    res = tools.search_products("heart shaped tea light holder", k=5)["results"]
    assert any("T-LIGHT" in r["description"] and "HEART" in r["description"] for r in res)


def test_forecast_tool():
    code = db.run_readonly("SELECT stock_code FROM forecasts LIMIT 1")["rows"][0][0]
    out = tools.get_forecast(code)
    assert len(out["weeks"]) == 8 and out["total_units_8w"] >= 0


def test_simulation_is_consistent():
    code = db.run_readonly("SELECT stock_code FROM elasticity WHERE reliable LIMIT 1")["rows"][0][0]
    out = tools.simulate_price_change(code, [-0.1, 0.0, 0.1])
    zero = [s for s in out["scenarios"] if s["pct_change"] == 0.0][0]
    assert abs(zero["revenue_delta_vs_base"]) < 1e-6
    # With elastic demand (< -1), a price cut should raise revenue.
    if out["elasticity"] < -1:
        cut = [s for s in out["scenarios"] if s["pct_change"] == -0.1][0]
        assert cut["revenue_delta_vs_base"] > 0


def test_tool_errors_are_returned_not_raised():
    result, ok = tools.call("run_sql", '{"query": "DELETE FROM products"}')
    assert not ok and "error" in result
    result, ok = tools.call("nope", "{}")
    assert not ok
