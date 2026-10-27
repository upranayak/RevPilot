"""In-memory DuckDB over the processed parquet files, with a read-only query guard."""
import re
import threading

import duckdb

from config import CITIES, ELASTICITY, FORECASTS, PRODUCTS, TRANSACTIONS

SCHEMA = """
transactions(order_id, order_date DATE, status, is_cancelled BOOLEAN, is_returned BOOLEAN, fulfilment, sales_channel,
             service_level, style, sku, category, sub_category, color, size, quantity, amount, unit_price, revenue,
             city, state, postal_code, has_promotion BOOLEAN, is_b2b BOOLEAN)
  -- one row per order line, Amazon.in, 31 Mar to 29 Jun 2022, money in INR (Indian rupees).
  -- revenue = net sales (0 for cancelled lines); amount = order value incl. cancelled lines.
  -- fulfilment in ('Amazon','Merchant'); service_level in ('Standard','Expedited'). Most orders have 1 line.
  -- category in (Set, Kurta, Western Dress, Top, Ethnic Dress, Blouse, Bottom, Saree, Dupatta).
  -- state and city are title case, e.g. 'Maharashtra', 'Tamil Nadu', 'Bengaluru'.
products(style, category, sub_category, color, sizes, description, units, revenue, median_price, orders, n_skus,
         first_sold, last_sold)  -- one row per style (design), net sales only
cities(state, city, orders, revenue, units, cancel_rate, return_rate, b2b_share, promo_share, amazon_fulfilled_share,
       expedited_share, aov, first_order, last_order, segment, segment_name)
  -- cities with 30+ orders; rates are shares of order lines (0 to 1); segment_name in
  -- ('Metro hubs', 'High-return markets', 'High-cancellation markets', 'B2B buyers', 'Steady tier-2/3 markets')
forecasts(style, day DATE, forecast, model)  -- next 14 days of unit demand, top 100 styles only
elasticity(style, description, category, base_price, forecast_units_14d, elasticity, ci_low, ci_high, p_value, r2,
           n_days, price_cv, min_price, max_price, reliable BOOLEAN)
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
                               ("cities", CITIES), ("forecasts", FORECASTS), ("elasticity", ELASTICITY)]:
                cols = "* EXCLUDE (x, y)" if name == "cities" else "*"
                con.execute(f"CREATE TABLE {name} AS SELECT {cols} FROM read_parquet('{path.as_posix()}')")
            _con = con
        return _con


def run_readonly(sql: str, max_rows: int = 50) -> dict:
    """Run one SELECT/WITH statement and return columns + rows (capped)."""
    stmt = sql.strip().rstrip(";")
    # Check keywords outside string literals only, so values like category = 'Set' are allowed.
    code = re.sub(r"'(?:[^']|'')*'", "''", stmt)
    if ";" in code or not re.match(r"^\s*(select|with)\b", code, re.I) or _FORBIDDEN.search(code):
        raise ValueError("Only a single read-only SELECT/WITH query is allowed.")
    cur = connection().cursor()
    df = cur.execute(stmt).fetch_df()
    truncated = len(df) > max_rows
    df = df.head(max_rows)
    return {"columns": list(df.columns), "rows": df.astype(object).where(df.notna(), None).values.tolist(),
            "row_count": len(df), "truncated": truncated}
