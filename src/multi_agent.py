"""Multi-agent counterfactual pricing workflow.

Analyst  -> pulls the style's sales history with SQL.
Forecaster -> reads the 14-day demand forecast.
Strategist -> simulates price scenarios with the elasticity model and proposes a price.
Critic   -> checks the proposal against the evidence (elasticity confidence, data limits)
            and returns the final recommendation.
Each role is a separate agent with only the tools it needs.
"""
import time
from dataclasses import dataclass, field

import agent
from agent import AgentResult
from config import AGENT_MODEL
from db import SCHEMA

ROLE_PROMPTS = {
    "analyst": f"""You are the Analyst. Use run_sql to summarise the style's sales history:
total units and net revenue (INR), median and range of unit_price, cancellation and return rates, and units by
month (April, May, June 2022). Tables: {SCHEMA}
revenue already excludes cancelled lines. Reply with a short bullet list of facts only.""",
    "forecaster": """You are the Forecaster. Use get_forecast for the style and report the 14-day total and
the daily pattern in 2 sentences. Facts only.""",
    "strategist": """You are the Pricing Strategist. Use simulate_price_change with pct_changes
[-0.15, -0.10, -0.05, 0.05, 0.10, 0.15] for the style. Recommend ONE price change that maximises 14-day
revenue, and state the expected revenue change in ₹ and %. Mention the elasticity and its confidence interval.""",
    "critic": """You are the Critic. You are given the Analyst, Forecaster and Strategist outputs. Check the
recommendation: Is the elasticity reliable (p < 0.05, CI excludes 0)? Does the revenue gain survive the CI range?
Is the new price within observed_price_range_inr? Then give the FINAL recommendation in this format:
Recommendation: <price change or 'hold price'>
Expected impact: <₹ and % over 14 days, with range>
Confidence: <high/medium/low> - <one-line reason>
Risks: <1-2 short bullets>""",
}


@dataclass
class PricingRun:
    style: str
    outputs: dict = field(default_factory=dict)
    results: dict = field(default_factory=dict)
    seconds: float = 0.0


def run(style: str, model: str = AGENT_MODEL) -> PricingRun:
    code = style.strip().upper()
    out = PricingRun(code)
    t0 = time.perf_counter()
    plan = [("analyst", ["run_sql"], f"Style {code}."),
            ("forecaster", ["get_forecast"], f"Style {code}."),
            ("strategist", ["simulate_price_change"], f"Style {code}.")]
    for role, allowed, task in plan:
        r: AgentResult = agent.run(task, allowed=allowed, system=ROLE_PROMPTS[role], model=model, max_steps=5)
        out.results[role], out.outputs[role] = r, r.answer
    context = "\n\n".join(f"## {role.title()}\n{out.outputs[role]}" for role, _, _ in plan)
    critic = agent.run(f"Style {code}.\n\n{context}", allowed=["simulate_price_change"],
                       system=ROLE_PROMPTS["critic"], model=model, max_steps=3)
    out.results["critic"], out.outputs["critic"] = critic, critic.answer
    out.seconds = round(time.perf_counter() - t0, 3)
    return out
