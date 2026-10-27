"""RevPilot: autonomous e-commerce revenue optimization (Streamlit app)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import db  # noqa: E402
import tools  # noqa: E402
from config import ARTIFACTS, COHORTS, CUSTOMERS, ELASTICITY, FORECASTS, PRODUCTS, WEEKLY  # noqa: E402
import llm  # noqa: E402
from llm import api_key  # noqa: E402
from pricing import simulate  # noqa: E402

GOLD = "#C99700"
st.set_page_config(page_title="RevPilot: E-Commerce Revenue Agent", page_icon="📈", layout="wide")


@st.cache_data
def load(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


@st.cache_data
def metrics(name: str) -> dict:
    p = ARTIFACTS / name
    return json.loads(p.read_text()) if p.exists() else {}


@st.cache_resource(show_spinner="Loading 1M transactions and the product index (first visit only)...")
def warm() -> bool:
    db.connection()
    tools._index()
    return True


def sql(q: str) -> pd.DataFrame:
    out = db.run_readonly(q, max_rows=10_000)
    return pd.DataFrame(out["rows"], columns=out["columns"])


warm()
products = load(str(PRODUCTS))
has_key = bool(api_key())

st.title("RevPilot")
st.caption("An agentic AI analyst for an online retailer: SQL + retrieval tool use, customer embeddings, "
           "demand forecasting and multi-agent counterfactual pricing. Data: UCI Online Retail II "
           "(1.02M transaction lines, Dec 2009 to Dec 2011, GBP).")

tab_over, tab_agent, tab_cust, tab_fc, tab_price, tab_eval = st.tabs(
    ["Overview", "Ask the agent", "Customers", "Forecasts", "Pricing lab", "Evaluation"])

# ---------------- Overview ----------------
with tab_over:
    k = sql("SELECT SUM(revenue) rev, COUNT(DISTINCT invoice) orders, COUNT(DISTINCT customer_id) customers, "
            "COUNT(*) lines FROM transactions WHERE NOT is_cancellation").iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Net revenue", f"£{k['rev'] / 1e6:.2f}M")
    c2.metric("Orders", f"{int(k['orders']):,}")
    c3.metric("Customers", f"{int(k['customers']):,}")
    c4.metric("Transaction lines", f"{k['lines'] / 1e6:.2f}M")
    monthly = sql("SELECT date_trunc('month', invoice_date) AS month, SUM(revenue) AS revenue FROM transactions "
                  "WHERE NOT is_cancellation GROUP BY 1 ORDER BY 1")
    fig = px.bar(monthly, x="month", y="revenue", title="Monthly net revenue (GBP)", color_discrete_sequence=[GOLD])
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Top products")
    st.dataframe(products.head(15)[["stock_code", "description", "units", "revenue", "median_price"]],
                 hide_index=True, use_container_width=True)

# ---------------- Agent ----------------
with tab_agent:
    st.markdown("Ask a business question. The agent plans, calls tools (read-only SQL, product search, "
                "forecast, price simulation), checks errors, and answers with numbers.")
    if not has_key:
        st.info("The agent needs a Groq API key (GROQ_API_KEY). Dashboards work without it."
                + (f" Reason: {llm.key_error}." if llm.key_error else ""))
    examples = ["Which country outside the UK generated the most revenue in 2011?",
                "How many units of the white hanging heart tea light holder were sold, and what is its 8-week forecast?",
                "If we cut the regency cake stand price by 10%, what happens to 8-week revenue?",
                "What share of customers are in the Lost segment, and what is their median recency?"]
    choice = st.selectbox("Example questions", ["(type your own)"] + examples)
    q = st.text_input("Question", value="" if choice == "(type your own)" else choice)
    if st.button("Ask", type="primary", disabled=not (has_key and q.strip())):
        import agent
        with st.spinner("Thinking with tools..."):
            r = agent.run(q.strip())
        st.markdown(f"**Answer:** {r.answer}")
        st.caption(f"{r.seconds:.1f}s · {r.llm_calls} LLM calls · {len(r.steps)} tool calls")
        for i, s in enumerate(r.steps, 1):
            with st.expander(f"{i}. {s.tool} {'✅' if s.ok else '⚠️'} ({s.seconds:.2f}s)"):
                st.code(json.dumps(s.arguments, indent=2), language="json")
                st.text(s.result_preview)

# ---------------- Customers ----------------
with tab_cust:
    cust = load(str(CUSTOMERS))
    sm = metrics("segments_metrics.json")
    st.markdown(f"Each customer is embedded from their purchase history (TF-IDF-weighted customer × product "
                f"matrix → TruncatedSVD, {sm.get('embedding_dims', 16)} dims), combined with RFM features and "
                f"clustered with K-Means into {sm.get('n_segments', 5)} segments.")
    left, right = st.columns([3, 2])
    with left:
        samp = cust.sample(min(3000, len(cust)), random_state=1)
        fig = px.scatter(samp, x="x", y="y", color="segment_name", opacity=0.6, title="Customer embedding map (2-D projection)",
                         hover_data={"customer_id": True, "monetary": ":.0f", "frequency": True, "x": False, "y": False})
        st.plotly_chart(fig, use_container_width=True)
    with right:
        prof = cust.groupby("segment_name").agg(customers=("customer_id", "count"),
                                                median_recency_days=("recency_days", "median"),
                                                median_orders=("frequency", "median"),
                                                total_revenue=("monetary", "sum")).sort_values("total_revenue", ascending=False)
        prof["revenue_share"] = (prof["total_revenue"] / prof["total_revenue"].sum()).map("{:.0%}".format)
        st.dataframe(prof, use_container_width=True)
    coh = load(str(COHORTS))
    piv = coh[coh["age"] <= 12].pivot(index="cohort", columns="age", values="retention")
    fig = px.imshow(piv, aspect="auto", color_continuous_scale="YlOrBr", title="Monthly cohort retention",
                    labels={"x": "Months since first purchase", "y": "Cohort", "color": "Retention"})
    st.plotly_chart(fig, use_container_width=True)

# ---------------- Forecasts ----------------
with tab_fc:
    fm = metrics("forecast_metrics.json")
    weekly, fc = load(str(WEEKLY)), load(str(FORECASTS))
    names = products.set_index("stock_code")["description"]
    codes = [c for c in products["stock_code"] if c in set(fc["stock_code"])]  # best sellers first
    code = st.selectbox("Product", codes, format_func=lambda c: f"{c} · {names.get(c, '')}")
    h = weekly[weekly["stock_code"] == code]
    f = fc[fc["stock_code"] == code]
    fig = go.Figure()
    fig.add_scatter(x=h["week"], y=h["quantity"], name="Actual weekly units", line=dict(color="#555"))
    fig.add_scatter(x=f["week"], y=f["forecast"], name=f"Forecast ({f['model'].iloc[0]})", line=dict(color=GOLD, width=3))
    fig.update_layout(title="Weekly demand and 8-week forecast", height=420)
    st.plotly_chart(fig, use_container_width=True)
    if fm:
        st.markdown(f"**Backtest** on the last {fm['holdout_weeks']} weeks for {fm['products_forecast']} top products "
                    f"(model chosen on an earlier validation window). Volume-weighted WAPE, lower is better:")
        st.dataframe(pd.DataFrame({"test WAPE": fm["test_wape"]}).T, use_container_width=True)
        st.caption(f"Chosen model: {fm['chosen_model']}, "
                   f"{fm['test_wape_improvement_vs_seasonal_naive']:.0%} lower error than the seasonal-naive baseline.")

# ---------------- Pricing ----------------
with tab_price:
    el = load(str(ELASTICITY))
    rel = el[el["reliable"]].sort_values("forecast_units_8w", ascending=False)
    st.markdown("Counterfactual pricing: what happens to the next 8 weeks of revenue if the price changes? "
                "Elasticity is estimated per product from list-price variation (log-log regression with "
                "month controls). It is observational, not from price experiments, so treat results as directional.")
    code = st.selectbox("Product (reliable elasticity only)", rel["stock_code"],
                        format_func=lambda c: f"{c} · {names.get(c, '')}", key="price_code")
    r = rel.set_index("stock_code").loc[code]
    c1, c2, c3 = st.columns(3)
    c1.metric("Elasticity", f"{r['elasticity']:.2f}", help=f"95% CI {r['ci_low']:.2f} to {r['ci_high']:.2f}")
    c2.metric("Current list price", f"£{r['base_price']:.2f}")
    c3.metric("Forecast units (8w)", f"{r['forecast_units_8w']:,.0f}")
    grid = np.linspace(-0.15, 0.15, 31)  # constant-elasticity models are unreliable far from observed prices
    curve = pd.DataFrame([{"pct": p * 100,
                           "revenue": simulate(r["forecast_units_8w"], r["base_price"], r["elasticity"], p)["revenue"],
                           "lo": simulate(r["forecast_units_8w"], r["base_price"], r["ci_high"], p)["revenue"],
                           "hi": simulate(r["forecast_units_8w"], r["base_price"], r["ci_low"], p)["revenue"]}
                          for p in grid])
    curve[["lo", "hi"]] = np.sort(curve[["lo", "hi"]].to_numpy(), axis=1)
    fig = go.Figure()
    fig.add_scatter(x=curve["pct"], y=curve["hi"], line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=curve["pct"], y=curve["lo"], fill="tonexty", line=dict(width=0), name="Elasticity 95% CI",
                    fillcolor="rgba(201,151,0,0.2)")
    fig.add_scatter(x=curve["pct"], y=curve["revenue"], name="Expected revenue", line=dict(color=GOLD, width=3))
    fig.update_layout(title="8-week revenue vs price change", xaxis_title="Price change (%)",
                      yaxis_title="Revenue (GBP)", height=420)
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Multi-agent pricing review")
    st.caption("Analyst (SQL history) → Forecaster → Pricing strategist (scenarios) → Critic (checks evidence, final call).")
    if not has_key:
        st.info("Needs a Groq API key.")
    if st.button("Run multi-agent review", disabled=not has_key):
        import multi_agent
        with st.spinner("Agents are working..."):
            run = multi_agent.run(code)
        for role in ["analyst", "forecaster", "strategist"]:
            with st.expander(role.title()):
                st.markdown(run.outputs[role])
        st.markdown("**Critic's final recommendation**")
        st.markdown(run.outputs["critic"])
        st.caption(f"Finished in {run.seconds:.1f}s")

# ---------------- Evaluation ----------------
with tab_eval:
    ev = metrics("eval_report.json")
    st.markdown("Benchmarks are produced by `evals/run_evals.py`. Gold answers are computed with SQL from the data.")
    if "agent" in ev:
        a = ev["agent"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Answer accuracy", f"{a['answer_accuracy']:.0%}", help=f"{a['questions']} questions")
        c2.metric("Tool-routing F1", f"{a['routing_f1']:.2f}")
        c3.metric("Tool-call reliability", f"{a['tool_call_reliability']:.0%}")
        c4.metric("Latency p50 / p95", f"{a['latency_p50_s']:.1f}s / {a['latency_p95_s']:.1f}s")
        st.dataframe(pd.DataFrame({"accuracy": a["accuracy_by_category"]}).T, use_container_width=True)
        with st.expander("Per-question results"):
            det = pd.DataFrame(a["details"])[["id", "question", "gold", "answer", "correct", "tools_used",
                                              "routing_f1", "seconds"]]
            det["gold"] = det["gold"].astype(str)  # mixed text/number answers break Arrow serialization
            det["tools_used"] = det["tools_used"].map(", ".join)
            st.dataframe(det, hide_index=True, use_container_width=True)
    if "retrieval" in ev:
        rr = ev["retrieval"]
        st.markdown(f"**Product retrieval** on {rr['queries']} noisy queries (word dropped, shuffled, typo): "
                    f"recall@1 {rr['recall_at_1']:.0%}, recall@5 {rr['recall_at_5']:.0%}, MRR {rr['mrr']:.2f}.")
    fm, sm, pm = metrics("forecast_metrics.json"), metrics("segments_metrics.json"), metrics("pricing_metrics.json")
    if fm:
        st.markdown(f"**Forecasting:** {fm['chosen_model']} test WAPE {fm['test_wape'][fm['chosen_model']]:.3f} vs "
                    f"seasonal-naive {fm['test_wape']['seasonal_naive']:.3f}.")
    if sm:
        st.markdown(f"**Segmentation:** {sm['n_customers']:,} customers, silhouette {sm['silhouette']}, "
                    f"average month-3 cohort retention {sm['avg_month3_retention']:.0%}.")
    if pm:
        st.markdown(f"**Pricing:** {pm['reliable_estimates']} of {pm['products_with_estimates']} products have reliable "
                    f"elasticity estimates (p < 0.05); median {pm['median_reliable_elasticity']}.")
    st.markdown("**Limitations:** historical data only (2009 to 2011); elasticities are observational; "
                "the agent uses a hosted open-weights LLM and can make mistakes, so every number is traceable "
                "to a tool call shown in the trace.")
