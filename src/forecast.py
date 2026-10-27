"""Weekly demand forecasting for the top products, with a holdout backtest.

Models: seasonal-naive (52w, falling back to naive), 4-week moving average, Holt-Winters.
Selection is honest: one model for all products is chosen by volume-weighted WAPE on a
validation window (weeks -16..-9), then scored on a later, untouched test window (last 8
weeks) against the baselines. (Per-product selection was tried and scored worse on test:
it overfits the short validation window.) The chosen model is refit on all data to
forecast HORIZON weeks ahead.
Run after prepare_data.py: python src/forecast.py
"""
import json
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from config import ARTIFACTS, FORECASTS, PRODUCTS, TRANSACTIONS, WEEKLY

TOP_N = 100
HOLDOUT = 8
HORIZON = 8


def weekly_demand(tx: pd.DataFrame, codes: list[str]) -> pd.DataFrame:
    sales = tx[~tx["is_cancellation"] & tx["stock_code"].isin(codes) & (tx["quantity"] > 0)]
    # Drop the final partial week so the last point isn't artificially low.
    last_full = sales["invoice_date"].max().to_period("W").start_time
    sales = sales[sales["invoice_date"] < last_full]
    w = (sales.set_index("invoice_date").groupby("stock_code")
         .resample("W-SUN").agg({"quantity": "sum", "revenue": "sum"}).reset_index())
    w["avg_price"] = (w["revenue"] / w["quantity"]).replace([np.inf, -np.inf], np.nan)
    full = []
    for code, g in w.groupby("stock_code"):
        idx = pd.date_range(g["invoice_date"].min(), g["invoice_date"].max(), freq="W-SUN")
        g = g.set_index("invoice_date").reindex(idx)
        g["stock_code"] = code
        g["quantity"] = g["quantity"].fillna(0)
        g["revenue"] = g["revenue"].fillna(0)
        full.append(g.rename_axis("week").reset_index())
    return pd.concat(full, ignore_index=True)


def fc_seasonal_naive(y: np.ndarray, h: int) -> np.ndarray:
    if len(y) >= 52 + h:
        return y[-52:-52 + h] if h < 52 else y[-52:][:h]
    return np.repeat(y[-1], h)


def fc_moving_avg(y: np.ndarray, h: int) -> np.ndarray:
    return np.repeat(y[-4:].mean(), h)


def fc_holt_winters(y: np.ndarray, h: int) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if len(y) >= 2 * 52:
            m = ExponentialSmoothing(y, trend="add", damped_trend=True, seasonal="add", seasonal_periods=52)
        else:
            # Without two full years, a fitted trend extrapolates seasonal spikes (e.g. a pre-Christmas
            # ramp keeps rising into January), so use level-only smoothing.
            m = ExponentialSmoothing(y, trend=None)
        return np.clip(m.fit(optimized=True).forecast(h), 0, None)


MODELS = {"seasonal_naive": fc_seasonal_naive, "moving_avg_4w": fc_moving_avg, "holt_winters": fc_holt_winters}


def wape(actual: np.ndarray, pred: np.ndarray) -> float:
    denom = np.abs(actual).sum()
    return float(np.abs(actual - pred).sum() / denom) if denom > 0 else np.nan


def main() -> None:
    tx = pd.read_parquet(TRANSACTIONS)
    products = pd.read_parquet(PRODUCTS)
    codes = products.head(TOP_N)["stock_code"].tolist()
    weekly = weekly_demand(tx, codes)
    weekly.to_parquet(WEEKLY, index=False)

    rows, series = [], {}
    for code, g in weekly.groupby("stock_code"):
        y = g["quantity"].to_numpy(dtype=float)
        if len(y) < 2 * HOLDOUT + 12:
            continue
        series[code] = (y, g["week"].max())
        fit, val = y[:-2 * HOLDOUT], y[-2 * HOLDOUT:-HOLDOUT]
        train, test = y[:-HOLDOUT], y[-HOLDOUT:]
        rows.append({"stock_code": code, "val_units": float(val.sum()), "test_units": float(test.sum()),
                     **{f"val_wape_{k}": wape(val, f(fit, HOLDOUT)) for k, f in MODELS.items()},
                     **{f"wape_{k}": wape(test, f(train, HOLDOUT)) for k, f in MODELS.items()}})
    m = pd.DataFrame(rows)
    val_w = {k: float(np.average(m[f"val_wape_{k}"].fillna(1), weights=m["val_units"])) for k in MODELS}
    chosen = min(val_w, key=val_w.get)
    test_w = {k: round(float(np.average(m[f"wape_{k}"].fillna(1), weights=m["test_units"])), 3) for k in MODELS}

    out = []
    for code, (y, last_week) in series.items():
        weeks = pd.date_range(last_week + pd.Timedelta(weeks=1), periods=HORIZON, freq="W-SUN")
        out.append(pd.DataFrame({"stock_code": code, "week": weeks, "forecast": MODELS[chosen](y, HORIZON),
                                 "model": chosen}))
    pd.concat(out).to_parquet(FORECASTS, index=False)
    m.to_parquet(ARTIFACTS / "forecast_backtest.parquet", index=False)

    baseline = test_w["seasonal_naive"]
    summary = {
        "products_forecast": int(len(m)), "holdout_weeks": HOLDOUT, "horizon_weeks": HORIZON,
        "chosen_model": chosen, "validation_wape": {k: round(v, 3) for k, v in val_w.items()},
        "test_wape": test_w,
        "test_wape_improvement_vs_seasonal_naive": round((baseline - test_w[chosen]) / baseline, 3),
    }
    (ARTIFACTS / "forecast_metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
