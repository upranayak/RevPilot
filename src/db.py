"""In-memory DuckDB over the processed parquet files, with a read-only query guard."""
import re
import threading

import duckdb

from config import CUSTOMERS, ELASTICITY, FORECASTS, PRODUCTS, TRANSACTIONS

SCHEMA = """
transactions(invoice, stock_code, description, quantity, invoice_date TIMESTAMP, price, customer_id,
             country, is_cancellation BOOLEAN, revenue)  -- one row per invoice line, Dec 2009 to Dec 2011, GBP.
             Cancellations have is_cancellation = true and negative quantity.
products(stock_code, description, units, revenue, median_price, orders, first_sold, last_sold)
customers(customer_id, first_purchase, last_purchase, frequency, monetary, country, recency_days,
          segment, segment_name)  -- segment_name in Champions, Loyal, Occasional, At risk, Lost
forecasts(stock_code, week DATE, forecast, model)  -- next 8 weeks of unit demand, top products only
elasticity(stock_code, description, base_price, forecast_units_8w, elasticity, ci_low, ci_high,
           p_value, r2, n_weeks, reliable BOOLEAN)
""".strip()

_FORBIDDEN = re.compile(r"\b(insert|update|delete|drop|create|alter|attach|copy|pragma|install|load|export|"
                        r"call|set|read_csv|read_parquet|read_json|glob)\b", re.I)
_lock = threading.Lock()
_con: duckdb.DuckDBPyConnection | None = None


def connection() -> duckdb.DuckDBPyConnection:
    global _con
    with _lock:
        if _con is None:
            con = duckdb.connect(":memory:")
            for name, path in [("transactions", TRANSACTIONS), ("products", PRODUCTS),
                               ("customers", CUSTOMERS), ("forecasts", FORECASTS), ("elasticity", ELASTICITY)]:
                cols = "* EXCLUDE (x, y, " + ", ".join(f"emb_{i}" for i in range(16)) + ")" \
                    if name == "customers" else "*"
                con.execute(f"CREATE TABLE {name} AS SELECT {cols} FROM read_parquet('{path.as_posix()}')")
            _con = con
        return _con


def run_readonly(sql: str, max_rows: int = 50) -> dict:
    """Run one SELECT/WITH statement and return columns + rows (capped)."""
    stmt = sql.strip().rstrip(";")
    if ";" in stmt or not re.match(r"^\s*(select|with)\b", stmt, re.I) or _FORBIDDEN.search(stmt):
        raise ValueError("Only a single read-only SELECT/WITH query is allowed.")
    cur = connection().cursor()
    df = cur.execute(stmt).fetch_df()
    truncated = len(df) > max_rows
    df = df.head(max_rows)
    return {"columns": list(df.columns), "rows": df.astype(object).where(df.notna(), None).values.tolist(),
            "row_count": len(df), "truncated": truncated}
