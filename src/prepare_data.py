"""Load the Amazon India sales report (Mar to Jun 2022, INR), clean it, and write processed parquet tables.

Run once: python src/prepare_data.py
"""
import re

import pandas as pd

from config import PROCESSED, PRODUCTS, RAW_CATALOG, RAW_ORDERS, TRANSACTIONS

# Typos and abbreviations seen in ship-state.
STATE_FIX = {
    "RAJSHTHAN": "RAJASTHAN", "RAJSTHAN": "RAJASTHAN", "RJ": "RAJASTHAN", "ORISSA": "ODISHA",
    "PONDICHERRY": "PUDUCHERRY", "NEW DELHI": "DELHI", "PB": "PUNJAB", "PUNJAB/MOHALI/ZIRAKPUR": "PUNJAB",
    "NL": "NAGALAND", "AR": "ARUNACHAL PRADESH", "DADRA AND NAGAR": "DADRA AND NAGAR HAVELI AND DAMAN AND DIU",
}
SMALL_WORDS = {"and", "of"}


def title(s: str) -> str:
    return " ".join(w if w.lower() in SMALL_WORDS else w.capitalize() for w in s.lower().split())


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(RAW_ORDERS, low_memory=False, dtype={"ship-postal-code": str})
    df.columns = [c.strip() for c in df.columns]
    return df.rename(columns={
        "Order ID": "order_id", "Date": "order_date", "Status": "status", "Fulfilment": "fulfilment",
        "Sales Channel": "sales_channel", "ship-service-level": "service_level", "Style": "style", "SKU": "sku",
        "Category": "category", "Size": "size", "Qty": "quantity", "Amount": "amount", "ship-city": "city",
        "ship-state": "state", "ship-postal-code": "postal_code", "promotion-ids": "promotion_ids", "B2B": "is_b2b",
    })


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset=[c for c in df.columns if c != "index"]).copy()
    df["order_date"] = pd.to_datetime(df["order_date"], format="%m-%d-%y")
    df["style"] = df["style"].str.strip().str.upper()
    df["sku"] = df["sku"].str.strip().str.upper()
    df["category"] = df["category"].str.strip().map(title)
    state = df["state"].fillna("").str.upper().str.strip().replace(STATE_FIX)
    df["state"] = state.map(title).replace({"": None, "Apo": None})
    df["city"] = df["city"].fillna("").str.upper().str.replace(r"[^A-Z ]", " ", regex=True) \
        .map(lambda s: re.sub(r"\s+", " ", s).strip()).map(title).replace({"": None})
    df["postal_code"] = df["postal_code"].str.replace(r"\.0$", "", regex=True)
    df["is_cancelled"] = df["status"].eq("Cancelled")
    df["is_returned"] = df["status"].str.contains("Return", na=False)
    df["has_promotion"] = df["promotion_ids"].notna()
    df["amount"] = df["amount"].fillna(0.0)
    # Net revenue: shipped/pending lines only. Cancelled lines keep their order value in `amount`.
    df["revenue"] = df["amount"].where(~df["is_cancelled"] & (df["quantity"] > 0), 0.0)
    df["unit_price"] = (df["amount"] / df["quantity"]).where(df["quantity"] > 0)

    cat = pd.read_csv(RAW_CATALOG)
    cat["sku"] = cat["SKU Code"].astype(str).str.strip().str.upper()
    cat = cat.drop_duplicates("sku")[["sku", "Color", "Category"]].rename(
        columns={"Color": "color", "Category": "sub_category"})
    df = df.merge(cat, on="sku", how="left")
    df["color"] = df["color"].fillna("").str.strip().map(title).replace({"": None})
    df["sub_category"] = df["sub_category"].fillna("").str.replace("AN : ", "").str.strip().map(title).replace({"": None})

    cols = ["order_id", "order_date", "status", "is_cancelled", "is_returned", "fulfilment", "sales_channel",
            "service_level", "style", "sku", "category", "sub_category", "color", "size", "quantity", "amount",
            "unit_price", "revenue", "city", "state", "postal_code", "has_promotion", "is_b2b"]
    return df[cols].sort_values(["order_date", "order_id"]).reset_index(drop=True)


def build_products(tx: pd.DataFrame) -> pd.DataFrame:
    """One row per style (design). Description = colour + finer category, used by product search."""
    sales = tx[~tx["is_cancelled"] & (tx["quantity"] > 0)]
    agg = sales.groupby("style").agg(
        category=("category", lambda s: s.mode().iat[0]), units=("quantity", "sum"), revenue=("revenue", "sum"),
        median_price=("unit_price", "median"), orders=("order_id", "nunique"), n_skus=("sku", "nunique"),
        first_sold=("order_date", "min"), last_sold=("order_date", "max"),
    ).reset_index()
    attrs = tx.groupby("style").agg(
        color=("color", lambda s: s.mode().iat[0] if s.notna().any() else ""),
        sub_category=("sub_category", lambda s: s.mode().iat[0] if s.notna().any() else ""),
        sizes=("size", lambda s: ", ".join(sorted(s.dropna().unique(), key=SIZE_ORDER.index))),
    ).reset_index()
    out = agg.merge(attrs, on="style", how="left")
    kind = out["sub_category"].where(out["sub_category"] != "", out["category"])
    out["description"] = (out["color"] + " " + kind).str.strip()
    out["revenue"] = out["revenue"].round(2)
    return out.sort_values("revenue", ascending=False).reset_index(drop=True)


SIZE_ORDER = ["Free", "XS", "S", "M", "L", "XL", "XXL", "3XL", "4XL", "5XL", "6XL"]


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    tx = clean(load_raw())
    tx.to_parquet(TRANSACTIONS, index=False)
    products = build_products(tx)
    products.to_parquet(PRODUCTS, index=False)
    net = tx[~tx["is_cancelled"]]
    print(f"lines: {len(tx):,} | orders: {tx['order_id'].nunique():,} | styles: {len(products):,} | "
          f"SKUs: {tx['sku'].nunique():,} | states: {tx['state'].nunique()} | cities: {tx['city'].nunique():,} | "
          f"net revenue: INR {net['revenue'].sum():,.0f} | "
          f"{tx['order_date'].min():%Y-%m-%d} to {tx['order_date'].max():%Y-%m-%d}")


if __name__ == "__main__":
    main()
