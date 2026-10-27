"""Multi-agent counterfactual pricing workflow.

Analyst  -> pulls the product's sales history with SQL.
Forecaster -> reads the 8-week demand forecast.
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
    "analyst": f"""You are the Analyst. Use run_sql to summarise the product's sales history:
total units and revenue, average price, price range, and the last 3 months vs the same months a year earlier.
Tables: {SCHEMA}
Exclude cancellations. Reply with a short bullet list of facts only.""",
    "forecaster": """You are the Forecaster. Use get_forecast for the product and report the 8-week total and
the trend (rising, flat or falling) in 2 sentences. Facts only.""",
    "strategist": """You are the Pricing Strategist. Use simulate_price_change with pct_changes
[-0.15, -0.10, -0.05, 0.05, 0.10, 0.15] for the product. Recommend ONE price change that maximises 8-week
revenue, and state the expected revenue change in GBP and %. Mention the elasticity and its confidence interval.""",
    "critic": """You are the Critic. You are given the Analyst, Forecaster and Strategist outputs. Check the
recommendation: Is the elasticity reliable (p < 0.05, CI excludes 0)? Does the revenue gain survive the CI range?
Is the change within the historically observed price range? Then give the FINAL recommendation in this format:
Recommendation: <price change or 'hold price'>
Expected impact: <GBP and % over 8 weeks, with range>
Confidence: <high/medium/low> - <one-line reason>
Risks: <1-2 short bullets>""",
}


@dataclass
class PricingRun:
    stock_code: str
    outputs: dict = field(default_factory=dict)
    results: dict = field(default_factory=dict)
    seconds: float = 0.0


def run(stock_code: str, model: str = AGENT_MODEL) -> PricingRun:
    code = stock_code.upper()
    out = PricingRun(code)
    t0 = time.perf_counter()
    plan = [("analyst", ["run_sql"], f"Product stock_code {code}."),
            ("forecaster", ["get_forecast"], f"Product stock_code {code}."),
            ("strategist", ["simulate_price_change"], f"Product stock_code {code}.")]
    for role, allowed, task in plan:
        r: AgentResult = agent.run(task, allowed=allowed, system=ROLE_PROMPTS[role], model=model, max_steps=5)
        out.results[role], out.outputs[role] = r, r.answer
    context = "\n\n".join(f"## {role.title()}\n{out.outputs[role]}" for role, _, _ in plan)
    critic = agent.run(f"Product {code}.\n\n{context}", allowed=["simulate_price_change"],
                       system=ROLE_PROMPTS["critic"], model=model, max_steps=3)
    out.results["critic"], out.outputs["critic"] = critic, critic.answer
    out.seconds = round(time.perf_counter() - t0, 3)
    return out
