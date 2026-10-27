"""Semantic product search (the RAG retriever).

Styles are described by colour + type (e.g. "Navy Blue Kurta Set"). Descriptions are embedded with
word + character TF-IDF reduced by TruncatedSVD (latent semantic indexing, 128 dims). Lightweight enough
to run on a free cloud instance without a GPU or an embedding API.
Because descriptions are only 1 to 4 words, a single typo in the colour can lose the match, so query
words that are not in the catalogue vocabulary are first spell-corrected to the closest catalogue word.
Many styles share a description, so results are grouped by description, best seller first.
Build: python src/rag.py
"""
import difflib
import pickle
import re

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
                        codes=products["style"].to_numpy())
    vocab = sorted({w for t in text for w in t.split()})
    with open(PRODUCT_INDEX.with_suffix(".pkl"), "wb") as f:
        pickle.dump({"word": word, "char": char, "svd": svd, "vocab": vocab}, f)
    print(f"indexed {len(products):,} styles ({text.nunique()} distinct descriptions), {N_DIMS} dims")


class ProductIndex:
    def __init__(self) -> None:
        data = np.load(PRODUCT_INDEX, allow_pickle=True)
        self.emb, self.codes = data["emb"], data["codes"]
        with open(PRODUCT_INDEX.with_suffix(".pkl"), "rb") as f:
            m = pickle.load(f)
        self.word, self.char, self.svd, self.vocab = m["word"], m["char"], m["svd"], m["vocab"]
        self.products = pd.read_parquet(PRODUCTS).set_index("style")
        self.rev = self.products.loc[self.codes, "revenue"].to_numpy()

    def correct(self, query: str) -> str:
        words = re.findall(r"[a-z]+", query.lower())
        fixed = [w if w in self.vocab else (difflib.get_close_matches(w, self.vocab, n=1, cutoff=0.7) or [w])[0]
                 for w in words]
        return " ".join(fixed)

    def search(self, query: str, k: int = 5) -> list[dict]:
        q = self.correct(query)
        X = sparse.hstack([self.word.transform([q]), self.char.transform([q])]).tocsr()
        v = normalize(self.svd.transform(X))[0]
        scores = self.emb @ v
        order = np.lexsort((-self.rev, -np.round(scores, 3)))  # best score, then best seller
        out, seen = [], {}
        for i in order:
            p = self.products.loc[self.codes[i]]
            if p["description"] in seen:
                seen[p["description"]]["matching_styles"] += 1
                continue
            if len(out) == k:
                continue
            item = {"style": self.codes[i], "description": p["description"], "category": p["category"],
                    "score": round(float(scores[i]), 3), "revenue_inr": round(float(p["revenue"]), 2),
                    "units": int(p["units"]), "orders": int(p["orders"]),
                    "median_price_inr": round(float(p["median_price"]), 2), "matching_styles": 1}
            seen[p["description"]] = item
            out.append(item)
        return out


if __name__ == "__main__":
    build()
