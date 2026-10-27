"""Price elasticity per product and counterfactual revenue simulation.

Elasticity: log-log OLS of weekly units on the weekly *list* price with month-of-year dummies
(controls for seasonality), fitted on products with enough price variation.
List price = median unit price on retail-sized lines (<= RETAIL_QTY units). The weekly
average price paid is NOT used: this store gives volume discounts, so average price falls
whenever bulk orders rise, which would fake a strongly negative elasticity.
These are observational estimates (no randomised price tests), so treat them as directional.
Counterfactual: units' = units * (1 + dp) ** elasticity, with the 8-week forecast as baseline.
Run after forecast.py: python src/pricing.py
"""
import json

import numpy as np
import pandas as pd
import statsmodels.api as sm

from config import ARTIFACTS, ELASTICITY, FORECASTS, PRODUCTS, TRANSACTIONS, WEEKLY

MIN_WEEKS = 30
MIN_PRICE_CV = 0.03  # coefficient of variation of weekly list price
RETAIL_QTY = 12


def weekly_list_price(codes: list[str]) -> pd.DataFrame:
    tx = pd.read_parquet(TRANSACTIONS, columns=["stock_code", "invoice_date", "quantity", "price", "is_cancellation"])
    tx = tx[tx["stock_code"].isin(codes) & ~tx["is_cancellation"] & tx["quantity"].between(1, RETAIL_QTY)]
    tx["week"] = tx["invoice_date"].dt.to_period("W-SUN").dt.end_time.dt.normalize()
    return tx.groupby(["stock_code", "week"])["price"].median().rename("list_price").reset_index()


def fit_elasticity(g: pd.DataFrame) -> dict | None:
    g = g[(g["quantity"] > 0) & g["list_price"].notna() & (g["list_price"] > 0)]
    if len(g) < MIN_WEEKS:
        return None
    cv = g["list_price"].std() / g["list_price"].mean()
    if cv < MIN_PRICE_CV:
        return None
    X = pd.get_dummies(g["week"].dt.month.astype(str), prefix="m", drop_first=True, dtype=float)
    X.insert(0, "log_price", np.log(g["list_price"].to_numpy()))
    X = sm.add_constant(X)
    res = sm.OLS(np.log(g["quantity"].to_numpy()), X).fit()
    lo, hi = res.conf_int().loc["log_price"]
    return {"elasticity": float(res.params["log_price"]), "ci_low": float(lo), "ci_high": float(hi),
            "p_value": float(res.pvalues["log_price"]), "r2": float(res.rsquared), "n_weeks": int(len(g)),
            "price_cv": float(cv)}


def simulate(base_units: float, base_price: float, elasticity: float, pct_change: float) -> dict:
    new_price = base_price * (1 + pct_change)
    new_units = base_units * (1 + pct_change) ** elasticity
    return {"price": new_price, "units": new_units, "revenue": new_price * new_units}


def main() -> None:
    weekly = pd.read_parquet(WEEKLY)
    weekly["week"] = pd.to_datetime(weekly["week"]).dt.normalize()
    weekly = weekly.merge(weekly_list_price(weekly["stock_code"].unique().tolist()),
                          on=["stock_code", "week"], how="left")
    products = pd.read_parquet(PRODUCTS).set_index("stock_code")
    forecasts = pd.read_parquet(FORECASTS)
    rows = []
    for code, g in weekly.groupby("stock_code"):
        fit = fit_elasticity(g)
        if fit is None:
            continue
        recent = g[g["list_price"].notna()].tail(8)
        base_price = float(recent["list_price"].median())
        fc_units = float(forecasts.loc[forecasts["stock_code"] == code, "forecast"].sum())
        rows.append({"stock_code": code, "description": products.at[code, "description"],
                     "base_price": base_price, "forecast_units_8w": fc_units, **fit})
    el = pd.DataFrame(rows)
    # Only trust significant, economically plausible estimates.
    el["reliable"] = (el["p_value"] < 0.05) & (el["elasticity"] < 0) & (el["elasticity"] > -6)
    el.to_parquet(ELASTICITY, index=False)
    rel = el[el["reliable"]]
    summary = {"products_with_estimates": int(len(el)), "reliable_estimates": int(len(rel)),
               "median_reliable_elasticity": round(float(rel["elasticity"].median()), 2) if len(rel) else None,
               "elastic_share": round(float((rel["elasticity"] < -1).mean()), 2) if len(rel) else None}
    (ARTIFACTS / "pricing_metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
