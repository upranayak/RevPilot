# RevPilot: Autonomous E-Commerce Revenue Optimization

An agentic AI analyst for an online retailer. Ask a business question in plain English and an LLM agent
plans, calls tools (read-only SQL, semantic product search, demand forecasts, price simulation), checks its
own errors and answers with numbers you can trace. A multi-agent workflow reviews pricing decisions with
counterfactual "what if we change the price" simulations.

**Live app:** https://revpilot.streamlit.app (free tier: the first visit after idle takes ~30 s to wake up)

Data: [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) (CC BY 4.0):
1,021,128 transaction lines, 46,921 invoices, 5,875 customers and 4,721 products from a UK online gift
retailer, Dec 2009 to Dec 2011.

## What it does

| Component | How |
|---|---|
| **LLM agent with tool use** | `openai/gpt-oss-120b` on Groq with function calling, in a ReAct-style loop (max 6 steps). Tools: `run_sql`, `search_products`, `get_forecast`, `simulate_price_change`. Tool errors are returned to the model so it can fix its own queries. |
| **Safe SQL** | In-memory DuckDB over parquet. Single read-only `SELECT`/`WITH` only; writes, file readers and multi-statements are blocked; results capped. |
| **RAG retriever** | Product descriptions embedded with word + character TF-IDF reduced by TruncatedSVD (128-d latent semantic index), cosine top-k. |
| **Customer embeddings & segments** | TF-IDF-weighted customer × product purchase matrix → TruncatedSVD (16-d), combined with RFM, clustered with K-Means into 5 named segments. Monthly cohort retention. |
| **Demand forecasting** | Weekly units for the top 99 products. Seasonal-naive, 4-week moving average and Holt-Winters compared; model chosen on a validation window and scored on an untouched test window. |
| **Price elasticity** | Per product, log-log OLS of weekly units on weekly *list* price with month-of-year controls (list price from retail-sized lines, so volume discounts don't fake elasticity). |
| **Multi-agent pricing** | Analyst (SQL history) → Forecaster → Pricing Strategist (simulates −15%…+15%) → Critic (checks elasticity confidence, CI range and price history; gives the final call). Each agent only gets the tools its role needs. |
| **Evaluation** | 30 fixed questions with gold answers computed by SQL; plus an offline retrieval benchmark. |

## Results

| Area | Metric | Result |
|---|---|---|
| Agent (30 questions: SQL, retrieval, forecast, pricing) | Answer accuracy vs SQL-computed gold | **96.7%** (29/30) |
| | Tool-routing F1 (tools used vs tools needed) | **0.96** |
| | Tool-call reliability (calls that executed without error) | **100%** |
| | End-to-end latency, Groq free tier | **p50 3.7 s · p95 11.3 s** |
| Retriever (300 noisy product queries) | Recall@1 / Recall@5 / MRR | **0.49 / 0.79 / 0.61** |
| Forecasting (99 products, last 8 weeks held out) | Volume-weighted WAPE, Holt-Winters vs seasonal-naive | **0.585 vs 0.739 (20.8% lower)** |
| Segmentation (5,852 customers) | Silhouette · avg month-3 cohort retention | 0.11 · 22% |
| Pricing | Products with reliable elasticity (p < 0.05) · median | 56 of 91 · −2.03 |
| Multi-agent pricing review | 4 agents, end to end | ~12 s per product |

Notes on honesty: the 30 agent questions are a development set. After the first run, 3 of 4 misses
turned out to be scorer bugs (unicode minus signs) and 1 was a real error (guest checkouts counted as a
customer), which led to a one-line prompt rule. LLM answers vary between runs: the one remaining miss
(filtering to reliable estimates) passed on the first run.

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt pytest
cp .env.example .env              # paste a free key from https://console.groq.com/keys
# rebuild the data (optional: processed data is committed)
#   download online_retail_II.xlsx into data/raw/, then:
python src/prepare_data.py && python src/features.py && python src/forecast.py && python src/pricing.py && python src/rag.py
streamlit run app/streamlit_app.py
pytest -q                         # 12 offline tests (no LLM calls)
python evals/run_evals.py         # agent + retrieval benchmark (needs the key)
```

## Deploy (free)

Streamlit Community Cloud → New app → this repo → main file `app/streamlit_app.py` →
Advanced settings → Secrets: `GROQ_API_KEY = "..."`.

## Limitations

- Historical data (2009 to 2011) only; forecasts start after the last week in the data.
- Elasticities are observational (no randomised price tests): directional, not causal.
- The agent can make mistakes; every number in an answer comes from a tool call shown in the trace.

## Project layout

| Path | What it does |
|---|---|
| `src/prepare_data.py` | Load and clean the raw Excel file → parquet |
| `src/features.py` | Customer embeddings, segments, cohorts |
| `src/forecast.py` | Weekly demand backtest and 8-week forecasts |
| `src/pricing.py` | Price elasticity estimates |
| `src/rag.py` | Product search index |
| `src/db.py`, `src/tools.py` | DuckDB with read-only guard; agent tools |
| `src/llm.py`, `src/agent.py`, `src/multi_agent.py` | Groq client, tool-using agent, pricing agents |
| `app/streamlit_app.py` | The web app |
| `evals/` | Benchmark questions and runner |
| `tests/` | Offline tests |
