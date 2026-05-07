"""
Generate Sakhi responses via the native Cohere API.

OpenRouter does not host Aya Expanse, so the previous Aya Expanse runs in
the expert arm all came back as 400 errors ("not a valid model ID"). This
script fills that gap using the Cohere ClientV2 chat endpoint.

Usage:
    python scoring/pipeline/generate_cohere_native.py \
        --model c4ai-aya-expanse-32b --model-alias aya_expanse \
        --dataset expert --lang en --num-runs 3

Resumable: skips any (run, q_idx) already in the per-(model,dataset,lang)
JSONL file at runs/gen__<alias>__<dataset>__<lang>.jsonl.

For full regeneration of broken Aya Expanse expert arm:
    for lang in en hi mr; do
        python ... --dataset expert --lang $lang
    done
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

REPO = Path(__file__).resolve().parents[2]
RUNS_DIR = REPO / "runs"
DATA_NONEXPERT = REPO / "data2" / "sakhi_non_expert_raw_230.csv"
DATA_EXPERT = REPO / "data2" / "sakhi_expert_raw_150.csv"

load_dotenv(Path.home() / ".env.cohere")
COHERE_KEY = os.environ.get("COHERE_API_KEY", "")
if not COHERE_KEY:
    sys.exit("COHERE_API_KEY missing - source ~/.env.cohere first")

import cohere

DATASET_CONFIG = {
    "expert": {
        "path": DATA_EXPERT,
        "q_cols": {"en": "question", "hi": "question_hi", "mr": "question_mr"},
        "id_col": "q_id",
    },
    "non_expert": {
        "path": DATA_NONEXPERT,
        "q_cols": {"en": "question", "hi": "questions_hindi", "mr": "questions_marathi"},
        "id_col": None,
    },
}


def load_questions(lang: str, dataset: str) -> list[dict]:
    cfg = DATASET_CONFIG[dataset]
    df = pd.read_csv(cfg["path"])
    q_col = cfg["q_cols"][lang]
    rows = []
    for i, r in df.iterrows():
        q = r[q_col]
        if pd.isna(q) or not str(q).strip():
            continue
        q_id = str(r[cfg["id_col"]]) if cfg["id_col"] and cfg["id_col"] in df.columns else str(i)
        # Strip stray surrounding quotes from theme that some source CSV rows have.
        # Three themes in the expert CSV ship as e.g. '"Nutrition, Diet & Supplementation"';
        # the RUBRIC_MAP keys are unquoted, so we strip here to keep judge lookups working.
        theme = str(r.get("theme", "")).strip().strip('"').strip("'").strip()
        rows.append({
            "q_idx": int(i),
            "q_id": q_id,
            "question": str(q).strip(),
            "theme": theme,
        })
    return rows


def existing_keys(jsonl_path: Path) -> set[tuple[int, int]]:
    """Return {(run, q_idx)} already present and valid."""
    keys: set[tuple[int, int]] = set()
    if not jsonl_path.exists():
        return keys
    with jsonl_path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("error"):
                continue
            run = int(r.get("run", 1))
            qi = int(r.get("q_idx", -1))
            if qi >= 0:
                keys.add((run, qi))
    return keys


PROMPT_TEMPLATE = """You are a maternal-health information assistant. Answer the question below in 2-4 sentences. Use simple, patient-friendly language. If the question concerns medication, dosage, or a clinical procedure, recommend consulting a qualified doctor or healthcare provider.

Question: {question}

Answer:"""


def call_cohere(client: cohere.ClientV2, model: str, prompt: str, *, max_retries: int = 3) -> tuple[str, dict]:
    last_err = None
    for attempt in range(max_retries):
        t0 = time.time()
        try:
            resp = client.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=400,
            )
            text = resp.message.content[0].text if resp.message.content else ""
            usage = getattr(resp, "usage", None)
            meta = {
                "input_tokens": getattr(usage.tokens, "input_tokens", None) if usage else None,
                "output_tokens": getattr(usage.tokens, "output_tokens", None) if usage else None,
                "latency_s": round(time.time() - t0, 3),
            }
            return text.strip(), meta
        except Exception as e:
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise
    raise RuntimeError(f"cohere call failed after {max_retries} retries: {last_err}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, help="Cohere model id (e.g. c4ai-aya-expanse-32b)")
    ap.add_argument("--model-alias", required=True, help="alias for filename (e.g. aya_expanse)")
    ap.add_argument("--dataset", required=True, choices=["expert", "non_expert"])
    ap.add_argument("--lang", required=True, choices=["en", "hi", "mr"])
    ap.add_argument("--num-runs", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None, help="probe limit (e.g. 10 for testing)")
    args = ap.parse_args()

    rows = load_questions(args.lang, args.dataset)
    if args.limit:
        rows = rows[: args.limit]
    runs_to_do = list(range(1, args.num_runs + 1))

    out_path = RUNS_DIR / f"gen__{args.model_alias}__{args.dataset}__{args.lang}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = existing_keys(out_path)
    to_do = [(run, r) for run in runs_to_do for r in rows if (run, r["q_idx"]) not in done]

    print(f"target: gen__{args.model_alias}__{args.dataset}__{args.lang}.jsonl")
    print(f"questions: {len(rows)}, runs: {args.num_runs}, total cells: {len(rows)*args.num_runs}")
    print(f"already valid: {len(done)}; to do: {len(to_do)}")

    if not to_do:
        print("nothing to do.")
        return 0

    client = cohere.ClientV2(api_key=COHERE_KEY)
    n_done = 0
    n_err = 0
    t_start = time.time()
    with out_path.open("a") as f:
        for run_num, q in to_do:
            prompt = PROMPT_TEMPLATE.format(question=q["question"])
            try:
                text, meta = call_cohere(client, args.model, prompt)
                rec = {
                    "model": args.model,
                    "model_alias": args.model_alias,
                    "provider": "cohere_native",
                    "run": run_num,
                    "dataset": args.dataset,
                    "lang": args.lang,
                    "q_idx": q["q_idx"],
                    "q_id": q["q_id"],
                    "theme": q["theme"],
                    "question": q["question"],
                    "response": text,
                    "meta": meta,
                    "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                }
                n_done += 1
            except Exception as e:
                rec = {
                    "model": args.model,
                    "model_alias": args.model_alias,
                    "provider": "cohere_native",
                    "run": run_num,
                    "dataset": args.dataset,
                    "lang": args.lang,
                    "q_idx": q["q_idx"],
                    "q_id": q["q_id"],
                    "error": str(e)[:300],
                    "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                }
                n_err += 1
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if (n_done + n_err) % 20 == 0:
                rate = (n_done + n_err) / max(0.1, time.time() - t_start)
                remain = len(to_do) - (n_done + n_err)
                eta = remain / max(0.1, rate)
                print(f"  progress: {n_done}/{len(to_do)} ok, {n_err} err, "
                      f"{rate:.1f} req/s, ETA {eta/60:.1f} min", flush=True)
    print(f"\nfinal: {n_done} ok, {n_err} err. wrote -> {out_path.relative_to(REPO)}")
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
