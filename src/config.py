"""Shared paths and settings."""
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
RAW_DIR = ROOT / "data" / "raw" / "india"
RAW_ORDERS = RAW_DIR / "Amazon Sale Report.csv"
RAW_CATALOG = RAW_DIR / "Sale Report.csv"  # SKU -> colour and finer category, same Kaggle dataset
PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"

TRANSACTIONS = PROCESSED / "transactions.parquet"
PRODUCTS = PROCESSED / "products.parquet"
CITIES = PROCESSED / "cities.parquet"
FORECASTS = PROCESSED / "forecasts.parquet"
DAILY = PROCESSED / "daily_demand.parquet"
ELASTICITY = PROCESSED / "elasticity.parquet"
PRODUCT_INDEX = PROCESSED / "product_index.npz"

HORIZON_DAYS = 14  # forecast horizon; the data covers ~3 months, so forecasts are short-range and daily

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
AGENT_MODEL = "openai/gpt-oss-120b"  # open-weights, served by Groq, supports tool calling
FAST_MODEL = "openai/gpt-oss-20b"
