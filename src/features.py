"""Customer embeddings + segments, and monthly acquisition cohorts.

Customer embedding: TF-IDF-weighted customer x product purchase matrix reduced with
TruncatedSVD (16 dims). Segments: KMeans on the embedding plus scaled RFM.
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

from config import ARTIFACTS, COHORTS, CUSTOMERS, TRANSACTIONS

N_DIMS = 16
N_SEGMENTS = 5
SEED = 42


def customer_embeddings(tx: pd.DataFrame, customer_ids: pd.Series) -> np.ndarray:
    sales = tx[~tx["is_cancellation"] & tx["customer_id"].notna() & (tx["quantity"] > 0)]
    cust_index = {c: i for i, c in enumerate(customer_ids)}
    prod_codes = sales["stock_code"].unique()
    prod_index = {p: i for i, p in enumerate(prod_codes)}
    counts = sales.groupby(["customer_id", "stock_code"]).size().reset_index(name="n")
    counts = counts[counts["customer_id"].isin(cust_index)]
    m = sparse.csr_matrix(
        (counts["n"].astype(float),
         (counts["customer_id"].map(cust_index), counts["stock_code"].map(prod_index))),
        shape=(len(customer_ids), len(prod_codes)),
    )
    weighted = TfidfTransformer(sublinear_tf=True).fit_transform(m)
    return TruncatedSVD(n_components=N_DIMS, random_state=SEED).fit_transform(weighted)


SEGMENT_NAMES = ["Champions", "Loyal", "Occasional", "At risk", "Lost"]


def name_segments(seg: pd.DataFrame) -> dict[int, str]:
    """Rank clusters by a combined RFM score (frequent, high-spend, recent first) and name them in order."""
    prof = np.log1p(seg.groupby("segment")[["recency_days", "frequency", "monetary"]].median())
    z = (prof - prof.mean()) / prof.std(ddof=0)
    score = z["frequency"] + z["monetary"] - z["recency_days"]
    return {s: SEGMENT_NAMES[i] for i, s in enumerate(score.sort_values(ascending=False).index)}


def build_segments(tx: pd.DataFrame, customers: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    emb = customer_embeddings(tx, customers["customer_id"])
    rfm = np.log1p(customers[["recency_days", "frequency", "monetary"]].clip(lower=0))
    features = np.hstack([StandardScaler().fit_transform(emb), StandardScaler().fit_transform(rfm) * 2])
    km = KMeans(n_clusters=N_SEGMENTS, n_init=10, random_state=SEED).fit(features)
    out = customers.copy()
    out["segment"] = km.labels_
    names = name_segments(out)
    out["segment_name"] = out["segment"].map(names)
    xy = TruncatedSVD(n_components=2, random_state=SEED).fit_transform(features)
    out["x"], out["y"] = xy[:, 0], xy[:, 1]
    for i in range(N_DIMS):
        out[f"emb_{i}"] = emb[:, i]
    sample = np.random.default_rng(SEED).choice(len(features), size=min(3000, len(features)), replace=False)
    metrics = {"n_customers": int(len(out)), "embedding_dims": N_DIMS, "n_segments": N_SEGMENTS,
               "silhouette": round(float(silhouette_score(features[sample], km.labels_[sample])), 3)}
    return out, metrics


def build_cohorts(tx: pd.DataFrame) -> pd.DataFrame:
    sales = tx[~tx["is_cancellation"] & tx["customer_id"].notna()].copy()
    sales["month"] = sales["invoice_date"].dt.to_period("M")
    first = sales.groupby("customer_id")["month"].min().rename("cohort")
    sales = sales.join(first, on="customer_id")
    sales["age"] = (sales["month"] - sales["cohort"]).apply(lambda d: d.n)
    active = sales.groupby(["cohort", "age"])["customer_id"].nunique().reset_index(name="customers")
    size = active[active["age"] == 0].set_index("cohort")["customers"]
    active["retention"] = active["customers"] / active["cohort"].map(size)
    active["cohort"] = active["cohort"].astype(str)
    return active


def main() -> None:
    tx = pd.read_parquet(TRANSACTIONS)
    customers = pd.read_parquet(CUSTOMERS)
    seg, metrics = build_segments(tx, customers)
    seg.to_parquet(CUSTOMERS, index=False)
    cohorts = build_cohorts(tx)
    cohorts.to_parquet(COHORTS, index=False)
    m3 = cohorts[cohorts["age"] == 3]["retention"].mean()
    metrics["avg_month3_retention"] = round(float(m3), 3)
    metrics["segments"] = seg["segment_name"].value_counts().to_dict()
    ARTIFACTS.mkdir(exist_ok=True)
    (ARTIFACTS / "segments_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
