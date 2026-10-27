"""Semantic product search (the RAG retriever).

Products are embedded from their description plus sales context with word + character
TF-IDF reduced by TruncatedSVD (latent semantic indexing, 128 dims). Lightweight enough
to run on a free cloud instance without a GPU or an embedding API.
Build: python src/rag.py
"""
import pickle

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from config import PRODUCT_INDEX, PRODUCTS

N_DIMS = 128


def product_text(p: pd.DataFrame) -> pd.Series:
    return p["description"].str.lower()


def build() -> None:
    products = pd.read_parquet(PRODUCTS)
    products = products[products["description"] != ""].reset_index(drop=True)
    text = product_text(products)
    word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)
    X = sparse.hstack([word.fit_transform(text), char.fit_transform(text)]).tocsr()
    svd = TruncatedSVD(n_components=N_DIMS, random_state=42)
    emb = normalize(svd.fit_transform(X))
    np.savez_compressed(PRODUCT_INDEX, emb=emb.astype(np.float32),
                        codes=products["stock_code"].to_numpy())
    with open(PRODUCT_INDEX.with_suffix(".pkl"), "wb") as f:
        pickle.dump({"word": word, "char": char, "svd": svd}, f)
    print(f"indexed {len(products):,} products, {N_DIMS} dims")


class ProductIndex:
    def __init__(self) -> None:
        data = np.load(PRODUCT_INDEX, allow_pickle=True)
        self.emb, self.codes = data["emb"], data["codes"]
        with open(PRODUCT_INDEX.with_suffix(".pkl"), "rb") as f:
            m = pickle.load(f)
        self.word, self.char, self.svd = m["word"], m["char"], m["svd"]
        self.products = pd.read_parquet(PRODUCTS).set_index("stock_code")

    def search(self, query: str, k: int = 5) -> list[dict]:
        q = query.lower()
        X = sparse.hstack([self.word.transform([q]), self.char.transform([q])]).tocsr()
        v = normalize(self.svd.transform(X))[0]
        scores = self.emb @ v
        top = np.argsort(-scores)[:k]
        out = []
        for i in top:
            code = self.codes[i]
            p = self.products.loc[code]
            out.append({"stock_code": code, "description": p["description"], "score": round(float(scores[i]), 3),
                        "revenue": round(float(p["revenue"]), 2), "units": int(p["units"]),
                        "median_price": round(float(p["median_price"]), 2)})
        return out


if __name__ == "__main__":
    build()
