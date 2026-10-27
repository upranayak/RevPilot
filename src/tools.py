"""Tools the agents can call. Each returns JSON-serialisable data."""
import json
from functools import lru_cache

import pandas as pd

import db
from config import ELASTICITY, FORECASTS
from pricing import simulate
from rag import ProductIndex


@lru_cache(maxsize=1)
def _index() -> ProductIndex:
    return ProductIndex()


@lru_cache(maxsize=1)
def _elasticity() -> pd.DataFrame:
    return pd.read_parquet(ELASTICITY).set_index("stock_code")


@lru_cache(maxsize=1)
def _forecasts() -> pd.DataFrame:
    return pd.read_parquet(FORECASTS)


def run_sql(query: str) -> dict:
    return db.run_readonly(query)


def search_products(query: str, k: int = 5) -> dict:
    return {"results": _index().search(query, k=min(int(k), 10))}


def get_forecast(stock_code: str) -> dict:
    f = _forecasts()
    g = f[f["stock_code"] == stock_code.upper()]
    if g.empty:
        return {"error": f"No forecast for {stock_code}. Forecasts exist for the top 100 products by revenue."}
    return {"stock_code": stock_code.upper(), "model": g["model"].iloc[0],
            "weeks": [{"week": str(w.date()), "units": round(float(u), 1)} for w, u in zip(g["week"], g["forecast"])],
            "total_units_8w": round(float(g["forecast"].sum()), 1)}


def simulate_price_change(stock_code: str, pct_changes: list[float]) -> dict:
    el = _elasticity()
    code = stock_code.upper()
    if code not in el.index:
        return {"error": f"No elasticity estimate for {code} (needs 30+ weeks of sales with price variation)."}
    r = el.loc[code]
    base = simulate(r["forecast_units_8w"], r["base_price"], r["elasticity"], 0.0)
    scenarios = []
    for pct in pct_changes[:8]:
        pct = max(-0.5, min(0.5, float(pct)))
        s = simulate(r["forecast_units_8w"], r["base_price"], r["elasticity"], pct)
        lo = simulate(r["forecast_units_8w"], r["base_price"], r["ci_high"], pct)  # least elastic bound
        hi = simulate(r["forecast_units_8w"], r["base_price"], r["ci_low"], pct)   # most elastic bound
        scenarios.append({"pct_change": pct, "price": round(s["price"], 2), "units_8w": round(s["units"], 1),
                          "revenue_8w": round(s["revenue"], 2),
                          "revenue_delta_vs_base": round(s["revenue"] - base["revenue"], 2),
                          "revenue_range_from_elasticity_ci": [round(min(lo["revenue"], hi["revenue"]), 2),
                                                               round(max(lo["revenue"], hi["revenue"]), 2)]})
    return {"stock_code": code, "description": r["description"], "elasticity": round(float(r["elasticity"]), 2),
            "elasticity_ci": [round(float(r["ci_low"]), 2), round(float(r["ci_high"]), 2)],
            "p_value": round(float(r["p_value"]), 4), "reliable": bool(r["reliable"]),
            "base_price": round(float(r["base_price"]), 2), "base_units_8w": round(float(r["forecast_units_8w"]), 1),
            "base_revenue_8w": round(base["revenue"], 2), "scenarios": scenarios}


REGISTRY = {"run_sql": run_sql, "search_products": search_products, "get_forecast": get_forecast,
            "simulate_price_change": simulate_price_change}

SPECS = {
    "run_sql": {"type": "function", "function": {
        "name": "run_sql",
        "description": "Run one read-only DuckDB SQL SELECT over the retail tables and return rows. "
                       "Use for any numeric question about sales, revenue, customers, countries or segments.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "A single SELECT or WITH query."}},
                       "required": ["query"]}}},
    "search_products": {"type": "function", "function": {
        "name": "search_products",
        "description": "Semantic search over product descriptions. Use to find stock codes for products described in words.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"},
                                                        "k": {"type": "integer", "description": "Results, max 10."}},
                       "required": ["query"]}}},
    "get_forecast": {"type": "function", "function": {
        "name": "get_forecast",
        "description": "Get the next 8 weeks of forecast unit demand for a product stock code.",
        "parameters": {"type": "object", "properties": {"stock_code": {"type": "string"}}, "required": ["stock_code"]}}},
    "simulate_price_change": {"type": "function", "function": {
        "name": "simulate_price_change",
        "description": "Counterfactual: simulate 8-week units and revenue if a product's price changes by each "
                       "fraction (e.g. -0.1 = 10% cut), using its estimated price elasticity.",
        "parameters": {"type": "object", "properties": {
            "stock_code": {"type": "string"},
            "pct_changes": {"type": "array", "items": {"type": "number"}, "description": "Fractions between -0.5 and 0.5."}},
            "required": ["stock_code", "pct_changes"]}}},
}


def call(name: str, arguments: str | dict) -> tuple[dict, bool]:
    """Execute a tool call. Returns (result, ok)."""
    try:
        args = json.loads(arguments) if isinstance(arguments, str) else (arguments or {})
        if name not in REGISTRY:
            return {"error": f"Unknown tool {name}"}, False
        result = REGISTRY[name](**args)
        return result, "error" not in result
    except Exception as e:  # surfaced back to the model so it can correct itself
        return {"error": f"{type(e).__name__}: {e}"}, False
