"""RevPilot: autonomous e-commerce revenue optimization on Amazon India sales (Streamlit app)."""
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
from config import ARTIFACTS, CITIES, DAILY, ELASTICITY, FORECASTS, HORIZON_DAYS, PRODUCTS  # noqa: E402
import llm  # noqa: E402
from llm import api_key  # noqa: E402
from pricing import simulate  # noqa: E402

GOLD = "#C99700"
st.set_page_config(page_title="RevPilot: E-Commerce Revenue Agent (India)", page_icon="📈", layout="wide")


def inr(x: float) -> str:
    """Indian units: crore (1e7) and lakh (1e5)."""
    if abs(x) >= 1e7:
        return f"₹{x / 1e7:.2f} Cr"
    if abs(x) >= 1e5:
        return f"₹{x / 1e5:.2f} L"
    return f"₹{x:,.0f}"


@st.cache_data
def load(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


@st.cache_data
def metrics(name: str) -> dict:
    p = ARTIFACTS / name
    return json.loads(p.read_text()) if p.exists() else {}


@st.cache_resource(show_spinner="Loading 129K Amazon.in order lines and the product index (first visit only)...")
def warm() -> bool:
    db.connection()
    tools._index()
    return True


def sql(q: str) -> pd.DataFrame:
    out = db.run_readonly(q, max_rows=10_000)
    return pd.DataFrame(out["rows"], columns=out["columns"])


warm()
products = load(str(PRODUCTS))
names = products.set_index("style")["description"]
has_key = bool(api_key())

st.title("RevPilot")
st.caption("An agentic AI analyst for an Indian ethnic-wear brand on Amazon.in: SQL + retrieval tool use, city market "
           "embeddings, demand forecasting and multi-agent counterfactual pricing. Data: Amazon India sales report "
           "(128,969 order lines, 31 Mar to 29 Jun 2022, INR).")

tab_over, tab_agent, tab_mkt, tab_fc, tab_price, tab_eval = st.tabs(
    ["Overview", "Ask the agent", "Markets", "Forecasts", "Pricing lab", "Evaluation"])

# ---------------- Overview ----------------
with tab_over:
    k = sql("SELECT SUM(revenue) rev, COUNT(DISTINCT order_id) orders, "
            "SUM(revenue) / COUNT(DISTINCT CASE WHEN NOT is_cancelled THEN order_id END) aov, "
            "AVG(CASE WHEN is_cancelled THEN 1.0 ELSE 0 END) cancel FROM transactions").iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Net revenue", inr(k["rev"]))
    c2.metric("Orders", f"{int(k['orders']):,}")
    c3.metric("Average order value", f"₹{k['aov']:,.0f}")
    c4.metric("Cancelled lines", f"{k['cancel']:.1%}")
    daily = sql("SELECT order_date AS day, SUM(revenue) AS revenue FROM transactions GROUP BY 1 ORDER BY 1")
    fig = px.line(daily, x="day", y="revenue", title="Daily net revenue (INR)", color_discrete_sequence=[GOLD])
    st.plotly_chart(fig, use_container_width=True)
    left, right = st.columns(2)
    with left:
        states = sql("SELECT state, SUM(revenue) AS revenue FROM transactions WHERE state IS NOT NULL "
                     "GROUP BY 1 ORDER BY 2 DESC LIMIT 12")
        fig = px.bar(states, x="revenue", y="state", orientation="h", title="Top states by net revenue (INR)",
                     color_discrete_sequence=[GOLD])
        fig.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        cats = sql("SELECT category, SUM(revenue) AS revenue FROM transactions GROUP BY 1 ORDER BY 2 DESC")
        fig = px.bar(cats, x="category", y="revenue", title="Net revenue by category (INR)",
                     color_discrete_sequence=[GOLD])
        st.plotly_chart(fig, use_container_width=True)
    st.subheader("Top styles")
    top = products.head(15)[["style", "description", "category", "units", "revenue", "median_price"]].rename(
        columns={"revenue": "revenue (₹)", "median_price": "median price (₹)"})
    st.dataframe(top, hide_index=True, use_container_width=True)

# ---------------- Agent ----------------
with tab_agent:
    st.markdown("Ask a business question. The agent plans, calls tools (read-only SQL, product search, "
                "forecast, price simulation), checks errors, and answers with numbers.")
    if not has_key:
        st.info("The agent needs a Groq API key (GROQ_API_KEY). Dashboards work without it."
                + (f" Reason: {llm.key_error}." if llm.key_error else ""))
    examples = ["Which state generated the most net revenue, and what share of it was Amazon-fulfilled?",
                "How many units of the best-selling blue kurta set were sold, and what is its 14-day forecast?",
                "If we cut the price of the best-selling blue dress by 10%, what happens to 14-day revenue?",
                "Which Metro hub city has the highest cancellation rate?"]
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
                st.code(json.dumps(s.arguments, indent=2, ensure_ascii=False), language="json")
                st.text(s.result_preview)

# ---------------- Markets ----------------
with tab_mkt:
    cities = load(str(CITIES))
    sm = metrics("segments_metrics.json")
    st.markdown(f"The report has no customer IDs, so markets are segmented by delivery city. Each city with "
                f"{sm.get('min_orders_per_city', 30)}+ orders is embedded from what it buys (TF-IDF-weighted city × "
                f"style matrix → TruncatedSVD, {sm.get('embedding_dims', 16)} dims), combined with market KPIs "
                f"(orders, AOV, cancellation, returns, B2B, FBA share) and clustered with K-Means into "
                f"{sm.get('n_segments', 5)} segments.")
    left, right = st.columns([3, 2])
    with left:
        fig = px.scatter(cities, x="x", y="y", color="segment_name", size=np.sqrt(cities["orders"]), opacity=0.7,
                         title="City market map (2-D projection)",
                         hover_data={"city": True, "state": True, "orders": True, "aov": ":.0f",
                                     "cancel_rate": ":.1%", "x": False, "y": False})
        st.plotly_chart(fig, use_container_width=True)
    with right:
        prof = cities.groupby("segment_name").agg(
            cities=("city", "count"), orders=("orders", "sum"), revenue=("revenue", "sum"),
            median_aov=("aov", "median"), cancel_rate=("cancel_rate", "median"), return_rate=("return_rate", "median"),
        ).sort_values("revenue", ascending=False)
        prof["revenue_share"] = (prof["revenue"] / prof["revenue"].sum()).map("{:.0%}".format)
        for c in ["cancel_rate", "return_rate"]:
            prof[c] = prof[c].map("{:.1%}".format)
        prof["revenue"] = prof["revenue"].map(inr)
        prof["median_aov"] = prof["median_aov"].map("₹{:,.0f}".format)
        st.dataframe(prof, use_container_width=True)
    mix = sql("SELECT state, category, SUM(revenue) AS revenue FROM transactions WHERE state IN "
              "(SELECT state FROM transactions WHERE state IS NOT NULL GROUP BY 1 ORDER BY SUM(revenue) DESC LIMIT 12) "
              "GROUP BY 1, 2")
    piv = mix.pivot(index="state", columns="category", values="revenue").fillna(0)
    piv = piv.div(piv.sum(axis=1), axis=0)
    fig = px.imshow(piv, aspect="auto", color_continuous_scale="YlOrBr", title="Category mix by state (share of net revenue)",
                    labels={"x": "Category", "y": "State", "color": "Share"})
    st.plotly_chart(fig, use_container_width=True)

# ---------------- Forecasts ----------------
with tab_fc:
    fm = metrics("forecast_metrics.json")
    daily_d, fc = load(str(DAILY)), load(str(FORECASTS))
    styles = [c for c in products["style"] if c in set(fc["style"])]  # best sellers first
    style = st.selectbox("Style", styles, format_func=lambda c: f"{c} · {names.get(c, '')}")
    h = daily_d[daily_d["style"] == style]
    f = fc[fc["style"] == style]
    fig = go.Figure()
    fig.add_scatter(x=h["day"], y=h["quantity"], name="Actual daily units", line=dict(color="#555"))
    fig.add_scatter(x=f["day"], y=f["forecast"], name=f"Forecast ({f['model'].iloc[0]})", line=dict(color=GOLD, width=3))
    fig.update_layout(title=f"Daily demand and {HORIZON_DAYS}-day forecast", height=420)
    st.plotly_chart(fig, use_container_width=True)
    if fm:
        st.markdown(f"**Backtest** on the last {fm['holdout_days']} days for the top {fm['styles_forecast']} styles "
                    f"(model chosen on an earlier validation window). Volume-weighted WAPE, lower is better:")
        st.dataframe(pd.DataFrame({"test WAPE": fm["test_wape"]}).T, use_container_width=True)
        st.caption(f"Chosen model: {fm['chosen_model']}, "
                   f"{fm['test_wape_improvement_vs_seasonal_naive']:.0%} lower error than the seasonal-naive "
                   f"(same weekday last week) baseline. Only {fm['history_days']} days of history, so Holt-Winters "
                   f"has little to learn from.")

# ---------------- Pricing ----------------
with tab_price:
    el = load(str(ELASTICITY))
    rel = el[el["reliable"]].sort_values("forecast_units_14d", ascending=False)
    st.markdown(f"Counterfactual pricing: what happens to the next {HORIZON_DAYS} days of revenue if the price changes? "
                "Elasticity is estimated per style from day-to-day changes in its Amazon selling price (log-log "
                "regression with day-of-week and trend controls). It is observational, not from price experiments, and "
                "price changes often coincide with Amazon deals, so treat results as directional.")
    style = st.selectbox("Style (reliable elasticity only)", rel["style"],
                         format_func=lambda c: f"{c} · {names.get(c, '')}", key="price_style")
    r = rel.set_index("style").loc[style]
    c1, c2, c3 = st.columns(3)
    c1.metric("Elasticity", f"{r['elasticity']:.2f}", help=f"95% CI {r['ci_low']:.2f} to {r['ci_high']:.2f}")
    c2.metric("Current selling price", f"₹{r['base_price']:,.0f}",
              help=f"Observed range ₹{r['min_price']:,.0f} to ₹{r['max_price']:,.0f}")
    c3.metric(f"Forecast units ({HORIZON_DAYS}d)", f"{r['forecast_units_14d']:,.0f}")
    grid = np.linspace(-0.15, 0.15, 31)  # constant-elasticity models are unreliable far from observed prices
    curve = pd.DataFrame([{"pct": p * 100,
                           "revenue": simulate(r["forecast_units_14d"], r["base_price"], r["elasticity"], p)["revenue"],
                           "lo": simulate(r["forecast_units_14d"], r["base_price"], r["ci_high"], p)["revenue"],
                           "hi": simulate(r["forecast_units_14d"], r["base_price"], r["ci_low"], p)["revenue"]}
                          for p in grid])
    curve[["lo", "hi"]] = np.sort(curve[["lo", "hi"]].to_numpy(), axis=1)
    fig = go.Figure()
    fig.add_scatter(x=curve["pct"], y=curve["hi"], line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=curve["pct"], y=curve["lo"], fill="tonexty", line=dict(width=0), name="Elasticity 95% CI",
                    fillcolor="rgba(201,151,0,0.2)")
    fig.add_scatter(x=curve["pct"], y=curve["revenue"], name="Expected revenue", line=dict(color=GOLD, width=3))
    fig.update_layout(title=f"{HORIZON_DAYS}-day revenue vs price change", xaxis_title="Price change (%)",
                      yaxis_title="Revenue (INR)", height=420)
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Multi-agent pricing review")
    st.caption("Analyst (SQL history) → Forecaster → Pricing strategist (scenarios) → Critic (checks evidence, final call).")
    if not has_key:
        st.info("Needs a Groq API key.")
    if st.button("Run multi-agent review", disabled=not has_key):
        import multi_agent
        with st.spinner("Agents are working..."):
            run = multi_agent.run(style)
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
        st.markdown(f"**Product retrieval** on {rr['queries']} noisy queries (shuffled words, one typo): "
                    f"recall@1 {rr['recall_at_1']:.0%}, recall@5 {rr['recall_at_5']:.0%}, MRR {rr['mrr']:.2f}.")
    fm, sm, pm = metrics("forecast_metrics.json"), metrics("segments_metrics.json"), metrics("pricing_metrics.json")
    if fm:
        st.markdown(f"**Forecasting:** {fm['chosen_model']} test WAPE {fm['test_wape'][fm['chosen_model']]:.3f} vs "
                    f"seasonal-naive {fm['test_wape']['seasonal_naive']:.3f}.")
    if sm:
        st.markdown(f"**Market segmentation:** {sm['n_cities']} cities covering {sm['order_coverage']:.0%} of orders, "
                    f"silhouette {sm['silhouette']}.")
    if pm:
        st.markdown(f"**Pricing:** {pm['reliable_estimates']} of {pm['styles_with_estimates']} styles have reliable "
                    f"elasticity estimates (p < 0.05); median {pm['median_reliable_elasticity']}.")
    st.markdown("**Limitations:** three months of 2022 data from one seller; no customer IDs, so no customer-level "
                "retention; elasticities are observational; the agent uses a hosted open-weights LLM and can make "
                "mistakes, so every number is traceable to a tool call shown in the trace.")
