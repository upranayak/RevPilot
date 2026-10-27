"""Load UCI Online Retail II, clean it, and write processed parquet tables.

Run once: python src/prepare_data.py
"""
import re

import pandas as pd

from config import CUSTOMERS, NON_PRODUCT_CODES, PROCESSED, PRODUCTS, RAW_XLSX, TRANSACTIONS


def load_raw() -> pd.DataFrame:
    sheets = pd.read_excel(RAW_XLSX, sheet_name=None, dtype={"Invoice": str, "StockCode": str})
    df = pd.concat(sheets.values(), ignore_index=True)
    df = df.rename(columns={
        "Invoice": "invoice", "StockCode": "stock_code", "Description": "description",
        "Quantity": "quantity", "InvoiceDate": "invoice_date", "Price": "price",
        "Customer ID": "customer_id", "Country": "country",
    })
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()
    df["stock_code"] = df["stock_code"].str.strip().str.upper()
    df["description"] = df["description"].fillna("").str.strip()
    df["is_cancellation"] = df["invoice"].str.startswith("C")
    is_product = ~df["stock_code"].isin(NON_PRODUCT_CODES) & df["stock_code"].str.match(r"^\d{5}")
    df = df[is_product & (df["price"] > 0)].copy()
    df["customer_id"] = df["customer_id"].astype("Int64")
    df["revenue"] = df["quantity"] * df["price"]
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    return df.reset_index(drop=True)


def build_products(tx: pd.DataFrame) -> pd.DataFrame:
    sales = tx[~tx["is_cancellation"]]
    # The most common description per stock code (descriptions drift over time).
    desc = (sales[sales["description"] != ""]
            .groupby(["stock_code", "description"]).size().rename("n").reset_index()
            .sort_values("n", ascending=False).drop_duplicates("stock_code"))
    agg = sales.groupby("stock_code").agg(
        units=("quantity", "sum"), revenue=("revenue", "sum"),
        median_price=("price", "median"), orders=("invoice", "nunique"),
        first_sold=("invoice_date", "min"), last_sold=("invoice_date", "max"),
    ).reset_index()
    out = agg.merge(desc[["stock_code", "description"]], on="stock_code", how="left")
    out["description"] = out["description"].fillna("").map(lambda s: re.sub(r"\s+", " ", s))
    return out.sort_values("revenue", ascending=False).reset_index(drop=True)


def build_customers(tx: pd.DataFrame) -> pd.DataFrame:
    sales = tx[~tx["is_cancellation"] & tx["customer_id"].notna()]
    snapshot = sales["invoice_date"].max() + pd.Timedelta(days=1)
    rfm = sales.groupby("customer_id").agg(
        last_purchase=("invoice_date", "max"), first_purchase=("invoice_date", "min"),
        frequency=("invoice", "nunique"), monetary=("revenue", "sum"),
        country=("country", "first"),
    ).reset_index()
    rfm["recency_days"] = (snapshot - rfm["last_purchase"]).dt.days
    return rfm


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    tx = clean(load_raw())
    tx.to_parquet(TRANSACTIONS, index=False)
    build_products(tx).to_parquet(PRODUCTS, index=False)
    build_customers(tx).to_parquet(CUSTOMERS, index=False)
    print(f"transactions: {len(tx):,} rows | invoices: {tx['invoice'].nunique():,} | "
          f"customers: {tx['customer_id'].nunique():,} | products: {tx['stock_code'].nunique():,} | "
          f"{tx['invoice_date'].min():%Y-%m-%d} to {tx['invoice_date'].max():%Y-%m-%d}")


if __name__ == "__main__":
    main()
