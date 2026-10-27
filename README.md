# RevPilot: Autonomous E-Commerce Revenue Optimization (Amazon India)

An agentic AI analyst for an Indian women's ethnic-wear brand selling on Amazon.in. Ask a business question in
plain English and an LLM agent plans, calls tools (read-only SQL, semantic product search, demand forecasts,
price simulation), checks its own errors and answers in ₹ with numbers you can trace. A multi-agent workflow
reviews pricing decisions with counterfactual "what if we change the price" simulations.

**Live app:** https://revpilot.streamlit.app (free tier: the first visit after idle takes ~30 s to wake up)

Data: Amazon India sales report from the Kaggle
[E-Commerce Sales Dataset](https://www.kaggle.com/datasets/thedevastator/unlock-profits-with-e-commerce-sales-data)
(uploaded by The Devastator; original source [Anil Sharma on data.world](https://data.world/anilsharma87); licence
"Other", credit the original author): 128,969 order lines, 120,378 orders, 1,371 styles (7,195 SKUs) shipped to
36 states and union territories, 31 Mar to 29 Jun 2022, ₹7.17 crore net revenue. Colours come from the dataset's
`Sale Report.csv` (matched by SKU).

## What it does

| Component | How |
|---|---|
| **LLM agent with tool use** | `openai/gpt-oss-120b` on Groq with function calling, in a ReAct-style loop (max 6 steps). Tools: `run_sql`, `search_products`, `get_forecast`, `simulate_price_change`. Tool errors are returned to the model so it can fix its own queries. |
| **Safe SQL** | In-memory DuckDB over parquet. Single read-only `SELECT`/`WITH` only; writes, file readers and multi-statements are blocked (checked outside string literals, so `category = 'Set'` works); results capped. |
| **RAG retriever** | Style descriptions (colour + type) embedded with word + character TF-IDF reduced by TruncatedSVD (128-d latent semantic index), with spelling correction against the catalogue vocabulary; one result per description, best seller first. |
| **City market embeddings & segments** | No customer IDs in the report, so the unit is the delivery city: TF-IDF-weighted city × style matrix → TruncatedSVD (16-d), plus KPIs (orders, AOV, cancellation, returns, B2B, FBA share), clustered with K-Means into 5 named segments (Metro hubs, High-return, High-cancellation, B2B buyers, Steady tier-2/3). |
| **Demand forecasting** | Daily units for the top 100 styles. Seasonal-naive (same weekday last week), 7-day moving average and Holt-Winters (weekly seasonality) compared; model chosen on a validation window and scored on an untouched 14-day test window. |
| **Price elasticity** | Per style, log-log OLS of daily units on the daily selling price with day-of-week and trend controls. |
| **Multi-agent pricing** | Analyst (SQL history) → Forecaster → Pricing Strategist (simulates −15%…+15%) → Critic (checks elasticity confidence, CI range and observed price range; gives the final call). Each agent only gets the tools its role needs. |
| **Evaluation** | 30 fixed questions with gold answers computed by SQL; plus an offline retrieval benchmark. |

## Results

| Area | Metric | Result |
|---|---|---|
| Agent (30 questions: SQL, retrieval, forecast, pricing) | Answer accuracy vs SQL-computed gold | **96.7%** (29/30) |
| | Tool-routing F1 (tools used vs tools needed) | **0.99** |
| | Tool-call reliability (calls that executed without error) | **100%** |
| | End-to-end latency, Groq free tier | **p50 8.0 s · p95 23.2 s** |
| Retriever (300 noisy queries: shuffled words, one typo) | Recall@1 / Recall@5 / MRR | **0.74 / 0.96 / 0.83** |
| Forecasting (100 styles, last 14 days held out) | Volume-weighted WAPE, 7-day moving average vs seasonal-naive | **0.692 vs 0.827 (16.3% lower)** |
| Market segmentation (339 cities, 81% of orders) | Silhouette | 0.11 |
| Pricing | Styles with reliable elasticity (p < 0.05) · median | 18 of 79 · −2.75 |

Notes on honesty: with only 89 full days of history, Holt-Winters did not beat a 7-day moving average on the
validation window, so the simpler model is used. Elasticities come from day-to-day Amazon price changes, which
often coincide with deals, so they are directional rather than causal. The 30 agent questions are a development set. The one miss (rag06) was a real error: asked how many orders included the best-selling green kurta, the agent re-queried with SQL and counted cancelled orders too (4,205 vs 3,567 net).

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt pytest
cp .env.example .env              # paste a free key from https://console.groq.com/keys
# rebuild the data (optional: processed data is committed)
#   download the Kaggle dataset and put "Amazon Sale Report.csv" and "Sale Report.csv" in data/raw/india/, then:
python src/prepare_data.py && python src/features.py && python src/forecast.py && python src/pricing.py && python src/rag.py
streamlit run app/streamlit_app.py
pytest -q                         # 15 offline tests (no LLM calls)
python evals/run_evals.py         # agent + retrieval benchmark (needs the key)
```

## Deploy (free)

Streamlit Community Cloud → New app → this repo → main file `app/streamlit_app.py` →
Advanced settings → Python 3.11 → Secrets: `GROQ_API_KEY = "..."`.

## Limitations

- Three months of 2022 data from one seller; forecasts start after the last day in the data.
- No customer IDs, so there is no customer-level retention or lifetime value.
- Elasticities are observational (no randomised price tests): directional, not causal.
- The agent can make mistakes; every number in an answer comes from a tool call shown in the trace.

## Project layout

| Path | What it does |
|---|---|
| `src/prepare_data.py` | Load and clean the raw CSVs (state names, cities, colours) → parquet |
| `src/features.py` | City market embeddings and segments |
| `src/forecast.py` | Daily demand backtest and 14-day forecasts |
| `src/pricing.py` | Price elasticity estimates |
| `src/rag.py` | Product search index |
| `src/db.py`, `src/tools.py` | DuckDB with read-only guard; agent tools |
| `src/llm.py`, `src/agent.py`, `src/multi_agent.py` | Groq client, tool-using agent, pricing agents |
| `app/streamlit_app.py` | The web app |
| `evals/` | Benchmark questions and runner |
| `tests/` | Offline tests |
