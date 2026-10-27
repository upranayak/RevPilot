"""Single tool-using analyst agent (ReAct-style loop over function calls)."""
import json
import time
from dataclasses import dataclass, field

import tools
from config import AGENT_MODEL
from db import SCHEMA
from llm import chat

SYSTEM = f"""You are a revenue analyst for a UK online gift retailer (Dec 2009 to Dec 2011, prices in GBP).
Answer using the tools; never guess numbers. Tables (DuckDB SQL):
{SCHEMA}
Rules:
- Exclude cancellations (is_cancellation = false) when measuring sales unless asked otherwise.
- customer_id is NULL for guest checkouts: exclude NULLs in any customer-level question
  (top customers, counts, segments). The customers table already excludes them.
- Revenue = SUM(revenue). Round money to 2 decimals.
- Use search_products to turn a product described in words into a stock_code before querying it.
- If a tool returns an error, fix the call and try again.
- Final answer: 1 to 4 sentences with the key numbers, stated once, then nothing else."""


@dataclass
class Step:
    tool: str
    arguments: dict
    ok: bool
    result_preview: str
    seconds: float


@dataclass
class AgentResult:
    answer: str
    steps: list[Step] = field(default_factory=list)
    seconds: float = 0.0
    llm_calls: int = 0
    error: str | None = None


def run(question: str, allowed: list[str] | None = None, system: str = SYSTEM, model: str = AGENT_MODEL,
        max_steps: int = 6) -> AgentResult:
    names = allowed or list(tools.SPECS)
    specs = [tools.SPECS[n] for n in names]
    messages = [{"role": "system", "content": system}, {"role": "user", "content": question}]
    res = AgentResult(answer="")
    t0 = time.perf_counter()
    try:
        for _ in range(max_steps):
            msg = chat(messages, model=model, tools=specs)
            res.llm_calls += 1
            calls = msg.get("tool_calls") or []
            if not calls:
                res.answer = (msg.get("content") or "").strip()
                break
            messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
            for c in calls:
                fn = c["function"]
                t1 = time.perf_counter()
                out, ok = tools.call(fn["name"], fn.get("arguments") or "{}")
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {"raw": fn.get("arguments")}
                text = json.dumps(out, default=str)[:6000]
                res.steps.append(Step(fn["name"], args, ok, text[:400], round(time.perf_counter() - t1, 3)))
                messages.append({"role": "tool", "tool_call_id": c["id"], "content": text})
        else:
            res.answer = "I could not finish within the step limit."
    except Exception as e:
        res.error = str(e)
        res.answer = f"Error: {e}"
    res.seconds = round(time.perf_counter() - t0, 3)
    return res
