"""Shared paths and settings."""
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
RAW_XLSX = ROOT / "data" / "raw" / "online_retail_II.xlsx"
PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"

TRANSACTIONS = PROCESSED / "transactions.parquet"
PRODUCTS = PROCESSED / "products.parquet"
CUSTOMERS = PROCESSED / "customers.parquet"
COHORTS = PROCESSED / "cohorts.parquet"
FORECASTS = PROCESSED / "forecasts.parquet"
WEEKLY = PROCESSED / "weekly_demand.parquet"
ELASTICITY = PROCESSED / "elasticity.parquet"
PRODUCT_INDEX = PROCESSED / "product_index.npz"

# Stock codes that are fees/adjustments, not products.
NON_PRODUCT_CODES = {
    "POST", "DOT", "M", "D", "C2", "BANK CHARGES", "AMAZONFEE", "CRUK", "PADS",
    "S", "B", "ADJUST", "ADJUST2", "TEST001", "TEST002", "SP1002", "GIFT",
}

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
AGENT_MODEL = "openai/gpt-oss-120b"  # open-weights, served by Groq, supports tool calling
FAST_MODEL = "openai/gpt-oss-20b"
