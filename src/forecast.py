"""Daily demand forecasting for the top styles, with a holdout backtest.

The data covers ~3 months, so forecasts are daily and short-range (14 days). Models: seasonal-naive
(same weekday last week), 7-day moving average, and Holt-Winters with weekly seasonality.
Selection is honest: one model for all styles is chosen by volume-weighted WAPE on a validation
window (days -28..-15), then scored on a later, untouched test window (last 14 days) against the
baselines. The chosen model is refit on all data to forecast HORIZON_DAYS ahead.
Run after prepare_data.py: python src/forecast.py
"""
import json
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from config import ARTIFACTS, DAILY, FORECASTS, HORIZON_DAYS, PRODUCTS, TRANSACTIONS

TOP_N = 100
HOLDOUT = 14


def daily_demand(tx: pd.DataFrame, styles: list[str]) -> pd.DataFrame:
    sales = tx[~tx["is_cancelled"] & tx["style"].isin(styles) & (tx["quantity"] > 0)]
    # The first and last days in the report are partial (a few hours of orders), so drop them.
    first, last = sales["order_date"].min(), sales["order_date"].max()
    sales = sales[(sales["order_date"] > first) & (sales["order_date"] < last)]
    days = pd.date_range(sales["order_date"].min(), sales["order_date"].max(), freq="D")
    d = sales.groupby(["style", "order_date"]).agg(quantity=("quantity", "sum"), revenue=("revenue", "sum"),
                                                   list_price=("unit_price", "median"))
    full = pd.MultiIndex.from_product([styles, days], names=["style", "order_date"])
    d = d.reindex(full).reset_index().rename(columns={"order_date": "day"})
    d[["quantity", "revenue"]] = d[["quantity", "revenue"]].fillna(0)
    return d


def fc_seasonal_naive(y: np.ndarray, h: int) -> np.ndarray:
    return np.resize(y[-7:], h)


def fc_moving_avg(y: np.ndarray, h: int) -> np.ndarray:
    return np.repeat(y[-7:].mean(), h)


def fc_holt_winters(y: np.ndarray, h: int) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = ExponentialSmoothing(y, trend=None, seasonal="add", seasonal_periods=7)
        return np.clip(m.fit(optimized=True).forecast(h), 0, None)


MODELS = {"seasonal_naive": fc_seasonal_naive, "moving_avg_7d": fc_moving_avg, "holt_winters": fc_holt_winters}


def wape(actual: np.ndarray, pred: np.ndarray) -> float:
    denom = np.abs(actual).sum()
    return float(np.abs(actual - pred).sum() / denom) if denom > 0 else np.nan


def main() -> None:
    tx = pd.read_parquet(TRANSACTIONS)
    products = pd.read_parquet(PRODUCTS)
    styles = products.head(TOP_N)["style"].tolist()
    daily = daily_demand(tx, styles)
    daily.to_parquet(DAILY, index=False)

    rows, series = [], {}
    for style, g in daily.groupby("style"):
        y = g["quantity"].to_numpy(dtype=float)
        series[style] = (y, g["day"].max())
        fit, val = y[:-2 * HOLDOUT], y[-2 * HOLDOUT:-HOLDOUT]
        train, test = y[:-HOLDOUT], y[-HOLDOUT:]
        rows.append({"style": style, "val_units": float(val.sum()), "test_units": float(test.sum()),
                     **{f"val_wape_{k}": wape(val, f(fit, HOLDOUT)) for k, f in MODELS.items()},
                     **{f"wape_{k}": wape(test, f(train, HOLDOUT)) for k, f in MODELS.items()}})
    m = pd.DataFrame(rows)
    val_w = {k: float(np.average(m[f"val_wape_{k}"].fillna(1), weights=m["val_units"])) for k in MODELS}
    chosen = min(val_w, key=val_w.get)
    test_w = {k: round(float(np.average(m[f"wape_{k}"].fillna(1), weights=m["test_units"])), 3) for k in MODELS}

    out = []
    for style, (y, last_day) in series.items():
        days = pd.date_range(last_day + pd.Timedelta(days=1), periods=HORIZON_DAYS, freq="D")
        out.append(pd.DataFrame({"style": style, "day": days, "forecast": MODELS[chosen](y, HORIZON_DAYS),
                                 "model": chosen}))
    pd.concat(out).to_parquet(FORECASTS, index=False)

    baseline = test_w["seasonal_naive"]
    summary = {
        "styles_forecast": int(len(m)), "history_days": int(daily["day"].nunique()), "holdout_days": HOLDOUT,
        "horizon_days": HORIZON_DAYS, "chosen_model": chosen,
        "validation_wape": {k: round(v, 3) for k, v in val_w.items()}, "test_wape": test_w,
        "test_wape_improvement_vs_seasonal_naive": round((baseline - test_w[chosen]) / baseline, 3),
    }
    (ARTIFACTS / "forecast_metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
