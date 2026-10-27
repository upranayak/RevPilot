"""Price elasticity per style and counterfactual revenue simulation.

Elasticity: log-log OLS of daily units on the daily list price (median selling price of the style
that day, in INR), with day-of-week dummies and a linear time trend (controls for the weekly
cycle and the April-to-June drift in demand). Only styles with enough sale days and price variation
are fitted. These are observational estimates (no randomised price tests), so treat them as directional.
Counterfactual: units' = units * (1 + dp) ** elasticity, with the 14-day forecast as baseline.
Run after forecast.py: python src/pricing.py
"""
import json

import numpy as np
import pandas as pd
import statsmodels.api as sm

from config import ARTIFACTS, DAILY, ELASTICITY, FORECASTS, PRODUCTS

MIN_DAYS = 30
MIN_PRICE_CV = 0.03  # coefficient of variation of the daily list price


def fit_elasticity(g: pd.DataFrame) -> dict | None:
    g = g[(g["quantity"] > 0) & g["list_price"].notna() & (g["list_price"] > 0)]
    if len(g) < MIN_DAYS:
        return None
    cv = g["list_price"].std() / g["list_price"].mean()
    if cv < MIN_PRICE_CV:
        return None
    X = pd.get_dummies(g["day"].dt.dayofweek.astype(str), prefix="dow", drop_first=True, dtype=float)
    X.insert(0, "trend", (g["day"] - g["day"].min()).dt.days.to_numpy(dtype=float) / 30)
    X.insert(0, "log_price", np.log(g["list_price"].to_numpy()))
    X = sm.add_constant(X)
    res = sm.OLS(np.log(g["quantity"].to_numpy()), X).fit()
    lo, hi = res.conf_int().loc["log_price"]
    return {"elasticity": float(res.params["log_price"]), "ci_low": float(lo), "ci_high": float(hi),
            "p_value": float(res.pvalues["log_price"]), "r2": float(res.rsquared), "n_days": int(len(g)),
            "price_cv": float(cv), "min_price": float(g["list_price"].min()), "max_price": float(g["list_price"].max())}


def simulate(base_units: float, base_price: float, elasticity: float, pct_change: float) -> dict:
    new_price = base_price * (1 + pct_change)
    new_units = base_units * (1 + pct_change) ** elasticity
    return {"price": new_price, "units": new_units, "revenue": new_price * new_units}


def main() -> None:
    daily = pd.read_parquet(DAILY)
    products = pd.read_parquet(PRODUCTS).set_index("style")
    forecasts = pd.read_parquet(FORECASTS)
    rows = []
    for style, g in daily.groupby("style"):
        fit = fit_elasticity(g)
        if fit is None:
            continue
        recent = g[g["list_price"].notna()].tail(14)
        fc_units = float(forecasts.loc[forecasts["style"] == style, "forecast"].sum())
        rows.append({"style": style, "description": products.at[style, "description"],
                     "category": products.at[style, "category"], "base_price": float(recent["list_price"].median()),
                     "forecast_units_14d": fc_units, **fit})
    el = pd.DataFrame(rows)
    # Only trust significant, economically plausible estimates.
    el["reliable"] = (el["p_value"] < 0.05) & (el["elasticity"] < 0) & (el["elasticity"] > -8)
    el.to_parquet(ELASTICITY, index=False)
    rel = el[el["reliable"]]
    summary = {"styles_with_estimates": int(len(el)), "reliable_estimates": int(len(rel)),
               "median_reliable_elasticity": round(float(rel["elasticity"].median()), 2) if len(rel) else None,
               "elastic_share": round(float((rel["elasticity"] < -1).mean()), 2) if len(rel) else None,
               "positive_significant": int(((el["p_value"] < 0.05) & (el["elasticity"] > 0)).sum())}
    (ARTIFACTS / "pricing_metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(el.sort_values("p_value").head(10)[["style", "description", "base_price", "elasticity", "ci_low", "ci_high",
                                              "p_value", "n_days", "price_cv"]].round(3).to_string())


if __name__ == "__main__":
    main()
