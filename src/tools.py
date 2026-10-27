"""Tools the agents can call. Each returns JSON-serialisable data."""
import json
from functools import lru_cache

import pandas as pd

import db
from config import ELASTICITY, FORECASTS, HORIZON_DAYS
from pricing import simulate
from rag import ProductIndex


@lru_cache(maxsize=1)
def _index() -> ProductIndex:
    return ProductIndex()


@lru_cache(maxsize=1)
def _elasticity() -> pd.DataFrame:
    return pd.read_parquet(ELASTICITY).set_index("style")


@lru_cache(maxsize=1)
def _forecasts() -> pd.DataFrame:
    return pd.read_parquet(FORECASTS)


def run_sql(query: str) -> dict:
    return db.run_readonly(query)


def search_products(query: str, k: int = 5) -> dict:
    return {"results": _index().search(query, k=min(int(k), 10))}


def get_forecast(style: str) -> dict:
    f = _forecasts()
    code = style.strip().upper()
    g = f[f["style"] == code]
    if g.empty:
        return {"error": f"No forecast for {style}. Forecasts exist for the top 100 styles by revenue."}
    return {"style": code, "model": g["model"].iloc[0],
            "days": [{"day": str(d.date()), "units": round(float(u), 1)} for d, u in zip(g["day"], g["forecast"])],
            f"total_units_{HORIZON_DAYS}d": round(float(g["forecast"].sum()), 1)}


def simulate_price_change(style: str, pct_changes: list[float]) -> dict:
    el = _elasticity()
    code = style.strip().upper()
    if code not in el.index:
        return {"error": f"No elasticity estimate for {code} (needs 30+ sale days with price variation; top 100 styles only)."}
    r = el.loc[code]
    units = r["forecast_units_14d"]
    base = simulate(units, r["base_price"], r["elasticity"], 0.0)
    scenarios = []
    for pct in pct_changes[:8]:
        pct = max(-0.5, min(0.5, float(pct)))
        s = simulate(units, r["base_price"], r["elasticity"], pct)
        lo = simulate(units, r["base_price"], r["ci_high"], pct)  # least elastic bound
        hi = simulate(units, r["base_price"], r["ci_low"], pct)   # most elastic bound
        scenarios.append({"pct_change": pct, "price_inr": round(s["price"], 2), "units_14d": round(s["units"], 1),
                          "revenue_14d_inr": round(s["revenue"], 2),
                          "revenue_delta_vs_base_inr": round(s["revenue"] - base["revenue"], 2),
                          "revenue_range_from_elasticity_ci": [round(min(lo["revenue"], hi["revenue"]), 2),
                                                               round(max(lo["revenue"], hi["revenue"]), 2)]})
    return {"style": code, "description": r["description"], "elasticity": round(float(r["elasticity"]), 2),
            "elasticity_ci": [round(float(r["ci_low"]), 2), round(float(r["ci_high"]), 2)],
            "p_value": round(float(r["p_value"]), 4), "reliable": bool(r["reliable"]),
            "observed_price_range_inr": [round(float(r["min_price"]), 2), round(float(r["max_price"]), 2)],
            "base_price_inr": round(float(r["base_price"]), 2), "base_units_14d": round(float(units), 1),
            "base_revenue_14d_inr": round(base["revenue"], 2), "scenarios": scenarios}


REGISTRY = {"run_sql": run_sql, "search_products": search_products, "get_forecast": get_forecast,
            "simulate_price_change": simulate_price_change}

SPECS = {
    "run_sql": {"type": "function", "function": {
        "name": "run_sql",
        "description": "Run one read-only DuckDB SQL SELECT over the Amazon India sales tables and return rows. "
                       "Use for any numeric question about sales, revenue, orders, states, cities, categories, "
                       "cancellations, returns or market segments.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "A single SELECT or WITH query."}},
                       "required": ["query"]}}},
    "search_products": {"type": "function", "function": {
        "name": "search_products",
        "description": "Semantic search over styles by colour and type (e.g. 'blue kurta set'), typo tolerant. Returns one "
                       "result per description: its best-selling style code with units, orders, revenue and median price, "
                       "plus how many styles share that description. Use to find the style of a product described in words.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"},
                                                        "k": {"type": "integer", "description": "Results, max 10."}},
                       "required": ["query"]}}},
    "get_forecast": {"type": "function", "function": {
        "name": "get_forecast",
        "description": f"Get the next {HORIZON_DAYS} days of forecast unit demand for a style code.",
        "parameters": {"type": "object", "properties": {"style": {"type": "string"}}, "required": ["style"]}}},
    "simulate_price_change": {"type": "function", "function": {
        "name": "simulate_price_change",
        "description": f"Counterfactual: simulate {HORIZON_DAYS}-day units and revenue (INR) if a style's price changes "
                       "by each fraction (e.g. -0.1 = 10% cut), using its estimated price elasticity.",
        "parameters": {"type": "object", "properties": {
            "style": {"type": "string"},
            "pct_changes": {"type": "array", "items": {"type": "number"}, "description": "Fractions between -0.5 and 0.5."}},
            "required": ["style", "pct_changes"]}}},
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
