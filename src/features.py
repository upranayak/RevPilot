"""City market embeddings + segments.

The Amazon India report has no customer IDs, so the unit of segmentation is the delivery city.
City embedding: TF-IDF-weighted city x style purchase matrix reduced with TruncatedSVD (16 dims),
i.e. what each city buys. Segments: KMeans on the embedding plus scaled market KPIs
(orders, average order value, cancellation and return rates, B2B and Amazon-fulfilled shares).
Run after prepare_data.py: python src/features.py
"""
import json

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from config import ARTIFACTS, CITIES, TRANSACTIONS

N_DIMS = 16
N_SEGMENTS = 5
MIN_ORDERS = 30  # cities with fewer orders are too noisy to profile
SEED = 42
KPIS = ["orders", "aov", "cancel_rate", "return_rate", "b2b_share", "amazon_fulfilled_share"]


def city_kpis(tx: pd.DataFrame) -> pd.DataFrame:
    tx = tx[tx["city"].notna() & tx["state"].notna()]
    g = tx.groupby(["state", "city"])
    out = g.agg(orders=("order_id", "nunique"), revenue=("revenue", "sum"), units=("quantity", "sum"),
                cancel_rate=("is_cancelled", "mean"), return_rate=("is_returned", "mean"),
                b2b_share=("is_b2b", "mean"), promo_share=("has_promotion", "mean"),
                amazon_fulfilled_share=("fulfilment", lambda s: (s == "Amazon").mean()),
                expedited_share=("service_level", lambda s: (s == "Expedited").mean()),
                first_order=("order_date", "min"), last_order=("order_date", "max")).reset_index()
    net_orders = tx[~tx["is_cancelled"]].groupby(["state", "city"])["order_id"].nunique().rename("net_orders")
    out = out.join(net_orders, on=["state", "city"])
    out["aov"] = out["revenue"] / out["net_orders"].replace(0, np.nan)
    out = out[out["orders"] >= MIN_ORDERS].drop(columns="net_orders").reset_index(drop=True)
    out["revenue"] = out["revenue"].round(2)
    out["aov"] = out["aov"].round(2)
    return out


def city_embeddings(tx: pd.DataFrame, cities: pd.DataFrame) -> np.ndarray:
    sales = tx[~tx["is_cancelled"] & (tx["quantity"] > 0)]
    key = cities["state"] + "|" + cities["city"]
    idx = {k: i for i, k in enumerate(key)}
    styles = {s: i for i, s in enumerate(sales["style"].unique())}
    counts = sales.assign(k=sales["state"] + "|" + sales["city"]).groupby(["k", "style"]).size().reset_index(name="n")
    counts = counts[counts["k"].isin(idx)]
    m = sparse.csr_matrix((counts["n"].astype(float), (counts["k"].map(idx), counts["style"].map(styles))),
                          shape=(len(cities), len(styles)))
    weighted = TfidfTransformer(sublinear_tf=True).fit_transform(m)
    return TruncatedSVD(n_components=N_DIMS, random_state=SEED).fit_transform(weighted)


# Each cluster is named after the trait that most sets it apart, assigned greedily in this order.
SEGMENT_RULES = [
    ("Metro hubs", "orders"),                     # highest median order volume
    ("High-return markets", "return_rate"),
    ("High-cancellation markets", "cancel_rate"),
    ("B2B buyers", "b2b_share"),
]
FALLBACK_NAME = "Steady tier-2/3 markets"


def name_segments(seg: pd.DataFrame) -> dict[int, str]:
    prof = seg.groupby("segment")[["orders", "return_rate", "cancel_rate", "b2b_share"]].median()
    prof["b2b_share"] = seg.groupby("segment")["b2b_share"].mean()  # median is 0 for most clusters
    names: dict[int, str] = {}
    for name, col in SEGMENT_RULES:
        free = prof.drop(index=list(names))
        names[int(free[col].idxmax())] = name
    for s in prof.index.difference(list(names)):
        names[int(s)] = FALLBACK_NAME
    return names


def build_segments(tx: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    cities = city_kpis(tx)
    emb = city_embeddings(tx, cities)
    kpi = cities[KPIS].copy()
    kpi["orders"] = np.log1p(kpi["orders"])
    features = np.hstack([StandardScaler().fit_transform(emb), StandardScaler().fit_transform(kpi) * 2])
    km = KMeans(n_clusters=N_SEGMENTS, n_init=10, random_state=SEED).fit(features)
    out = cities.copy()
    out["segment"] = km.labels_
    out["segment_name"] = out["segment"].map(name_segments(out))
    xy = TruncatedSVD(n_components=2, random_state=SEED).fit_transform(features)
    out["x"], out["y"] = xy[:, 0], xy[:, 1]
    covered = tx[tx["city"].notna()]
    share = covered.merge(out[["state", "city"]], on=["state", "city"])["order_id"].nunique() / tx["order_id"].nunique()
    metrics = {"n_cities": int(len(out)), "min_orders_per_city": MIN_ORDERS, "order_coverage": round(float(share), 3),
               "embedding_dims": N_DIMS, "n_segments": N_SEGMENTS,
               "silhouette": round(float(silhouette_score(features, km.labels_)), 3),
               "segments": out["segment_name"].value_counts().to_dict()}
    return out, metrics


def main() -> None:
    tx = pd.read_parquet(TRANSACTIONS)
    seg, metrics = build_segments(tx)
    seg.to_parquet(CITIES, index=False)
    ARTIFACTS.mkdir(exist_ok=True)
    (ARTIFACTS / "segments_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(seg.groupby("segment_name")[["orders", "aov", "cancel_rate", "return_rate", "b2b_share",
                                       "amazon_fulfilled_share"]].median().round(3).to_string())
    print(seg.sort_values("orders", ascending=False).head(10)[["city", "state", "orders", "segment_name"]].to_string())


if __name__ == "__main__":
    main()
