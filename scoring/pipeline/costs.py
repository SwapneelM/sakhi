"""Compute per-model generation cost from JSONL token usage.

For fresh generations: uses input_tokens / output_tokens recorded per call.
For migrated rows (no token counts): estimates tokens ≈ chars / 4 as fallback.

Pricing is hard-coded from provider pages as of 2026-04 (USD per 1M tokens).
Update PRICING when re-running for a different snapshot.
"""
from __future__ import annotations
import json, sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scoring.pipeline import checkpoint

REPO = Path(__file__).resolve().parent.parent.parent
RUNS_DIR = REPO / "runs"
OUT = REPO / "release" / "results"
OUT.mkdir(parents=True, exist_ok=True)

# USD per 1M tokens, April 2026 public provider pricing.
# (input_price, output_price). output_price covers reasoning + visible tokens where applicable.
PRICING = {
    "gpt_5_mini":            (0.25, 2.00),   # OpenAI
    "gpt_4o_mini":           (0.15, 0.60),
    "cohere_command_a":      (2.50, 10.00),  # Cohere
    "aya_expanse":           (0.50, 1.50),
    "gemini_3_pro":          (2.00, 12.00),  # Google
    "gemini_3_flash":        (0.50,  3.00),
    "gemini_3_1_flash_lite": (0.25,  1.50),
    "gemma_3_27b":           (0.20,  0.20),  # via OpenRouter (open-weight hosting)
    "llama_3_3_70b":         (0.50,  0.80),  # OpenRouter approx
    "llama_4_maverick":      (0.27,  0.85),
    "medgemma_4b":           (0.10,  0.20),  # HF Inference approx
    "medgemma_27b":          (0.30,  0.50),
    "claude_haiku_4_5":      (1.00,  5.00),
    # Claude Opus 4.7 ran through the local Claude Code subscription (CLI / agent),
    # so token counts were not surfaced per call. We treat the published Anthropic
    # API price for the Opus 4.x family as a what-if rate and estimate tokens from
    # text lengths so the cost figure has an Opus point. The result is an
    # API-equivalent estimate, not a figure paid for this paper.
    "claude_opus_4_7":       (15.00, 75.00),
    # judges (for reference in cost tables)
    "judge_gpt_4o_mini":     (0.15, 0.60),
    "judge_gpt_5_1":         (1.25, 10.00),
    "judge_claude_opus_4_6": (0.00, 0.00),
}


def token_estimate(text: str) -> int:
    return max(1, len(str(text)) // 4)


# When a row has no metered input_tokens (e.g. claude_cli / claude_code_agent
# path on Opus 4.7), we estimate input tokens from the question text alone.
# That undercounts because the model sees the full SAKHI prompt template
# (~350 tokens for EN, ~350 for ML) wrapped around each question. We add this
# constant so the API-equivalent cost estimate reflects what the API would
# actually have billed.
SAKHI_PROMPT_TEMPLATE_TOKENS_EST = 350


def main():
    per_model = defaultdict(lambda: {"calls": 0, "in_tok": 0, "out_tok": 0,
                                     "in_tok_est": 0, "out_tok_est": 0, "migrated": 0})
    for path in RUNS_DIR.glob("gen__*.jsonl"):
        parts = path.stem.split("__")
        if len(parts) != 4:
            continue
        alias = parts[1]
        for r in checkpoint.jsonl_rows(str(path)):
            if r.get("error") or not (r.get("response") or "").strip():
                continue
            m = per_model[alias]
            m["calls"] += 1
            in_t = r.get("input_tokens")
            out_t = r.get("output_tokens")
            if in_t is not None:
                m["in_tok"] += int(in_t)
            else:
                # Estimate input = SAKHI prompt template + question text.
                m["in_tok_est"] += (
                    SAKHI_PROMPT_TEMPLATE_TOKENS_EST
                    + token_estimate(r.get("question", ""))
                )
            if out_t is not None:
                m["out_tok"] += int(out_t)
            else:
                m["out_tok_est"] += token_estimate(r.get("response", ""))
            if r.get("provider") == "migrated":
                m["migrated"] += 1

    rows = []
    for alias in sorted(per_model):
        m = per_model[alias]
        price_in, price_out = PRICING.get(alias, (None, None))
        total_in = m["in_tok"] + m["in_tok_est"]
        total_out = m["out_tok"] + m["out_tok_est"]
        cost = None
        if price_in is not None:
            cost = round(total_in * price_in / 1e6 + total_out * price_out / 1e6, 4)
        rows.append({
            "model": alias,
            "calls": m["calls"],
            "input_tokens_logged": m["in_tok"],
            "output_tokens_logged": m["out_tok"],
            "input_tokens_est": m["in_tok_est"],
            "output_tokens_est": m["out_tok_est"],
            "migrated_rows": m["migrated"],
            "price_usd_per_M_in": price_in,
            "price_usd_per_M_out": price_out,
            "total_cost_usd": cost,
            "cost_per_response_usd": round(cost / m["calls"], 6) if (cost is not None and m["calls"]) else None,
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "cost_per_model.csv", index=False)
    print(df[["model", "calls", "total_cost_usd", "cost_per_response_usd"]].to_string(index=False))


if __name__ == "__main__":
    main()
