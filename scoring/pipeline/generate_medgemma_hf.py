"""
Generate MedGemma 27B responses on the expert arm via Hugging Face's
chat-completions interface (provider: featherless-ai).

This fills the gap left by orchestrate.py, which deliberately scoped
MedGemma to the non-expert arm because there was no OpenRouter route
at the time. With the featherless-ai provider now reachable through
the HF token, we can produce expert-arm coverage for the central
medical-fine-tune-vs-base claim against Gemma 3 27B.

Usage:
  set -a; source ~/.env.hf.medgemma; set +a
  python -m scoring.pipeline.generate_medgemma_hf

Resumes safely: skips any (run, q_idx, lang) keys already present in
the output JSONL.

Output:
  runs/gen__medgemma_27b__expert__{en,hi,mr}.jsonl
"""
from __future__ import annotations
import argparse, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from huggingface_hub import InferenceClient

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "runs"
RUNS.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompts import build_gen_prompt  # type: ignore

MODEL = "google/medgemma-27b-text-it:featherless-ai"
MODEL_ALIAS = "medgemma_27b"
DATASET = "expert"
N_RUNS = 3
TEMPERATURE = 0.7
MAX_TOKENS = 400
LANGS = ("en", "hi", "mr")
CONCURRENCY = 4

# Per-call retry policy (provider 502/503 occasionally happens).
MAX_ATTEMPTS = 3
RETRY_BACKOFF_S = 8.0


def load_existing_keys(path: Path) -> set[tuple[int, int]]:
    """Return the set of (run, q_idx) tuples already present in the JSONL."""
    if not path.exists():
        return set()
    keys = set()
    with path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            run, qi = r.get("run"), r.get("q_idx")
            if run is None or qi is None:
                continue
            keys.add((int(run), int(qi)))
    return keys


def call_once(client: InferenceClient, prompt: str) -> tuple[str, dict | None, str | None]:
    """Single chat-completion call. Returns (response_text, usage_dict, error_str_or_None)."""
    last_err = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            comp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            )
            text = comp.choices[0].message.content or ""
            usage = {
                "prompt_tokens": getattr(comp.usage, "prompt_tokens", None),
                "completion_tokens": getattr(comp.usage, "completion_tokens", None),
                "total_tokens": getattr(comp.usage, "total_tokens", None),
            }
            return text.strip(), usage, None
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {str(e)[:200]}"
            if attempt < MAX_ATTEMPTS:
                time.sleep(RETRY_BACKOFF_S * attempt)
    return "", None, last_err


def gen_one(client: InferenceClient, lang: str, q_idx: int, q_id: str, theme: str,
            question: str, run: int) -> dict:
    """Generate one (lang, q_idx, run) cell and return a dict ready for JSONL."""
    prompt = build_gen_prompt(question, lang)
    t0 = time.time()
    response, usage, error = call_once(client, prompt)
    dt = time.time() - t0
    return {
        "model": MODEL_ALIAS,
        "model_id": MODEL,
        "provider": "huggingface_featherless",
        "run": run,
        "dataset": DATASET,
        "lang": lang,
        "q_idx": int(q_idx),
        "q_id": str(q_id),
        "theme": str(theme),
        "question": question,
        "response": response,
        "input_tokens": (usage or {}).get("prompt_tokens"),
        "output_tokens": (usage or {}).get("completion_tokens"),
        "latency_s": round(dt, 2),
        "error": error,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", choices=("en", "hi", "mr", "all"), default="all")
    parser.add_argument("--num-runs", type=int, default=N_RUNS)
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    parser.add_argument("--limit", type=int, default=None,
                        help="Only generate first N questions (smoke test).")
    args = parser.parse_args()

    bench = pd.read_csv(REPO / "release" / "sakhi_benchmark_expert.csv")
    if args.limit:
        bench = bench.head(args.limit)
    print(f"Loaded {len(bench)} expert questions.")

    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN env var not set. source ~/.env.hf.medgemma first.")
    client = InferenceClient(api_key=token)

    langs = LANGS if args.lang == "all" else (args.lang,)

    for lang in langs:
        out_path = RUNS / f"gen__{MODEL_ALIAS}__{DATASET}__{lang}.jsonl"
        existing = load_existing_keys(out_path)
        # Drop the lone stub row that was sitting in this file.
        if existing == {(1, 0)} or (out_path.exists() and out_path.stat().st_size < 500):
            out_path.write_text("")
            existing = set()

        question_col = {"en": "question_en", "hi": "question_hi", "mr": "question_mr"}[lang]

        # Build the list of (run, q_idx) cells to generate.
        cells = []
        for run in range(1, args.num_runs + 1):
            for q_idx, row in bench.iterrows():
                if (run, int(q_idx)) in existing:
                    continue
                cells.append((run, int(q_idx), str(row.get("q_id", q_idx)),
                              str(row.get("theme", "")), str(row[question_col])))

        print(f"[{lang}] {len(cells)} cells to generate ({len(existing)} already done; output -> {out_path.name})")

        if not cells:
            continue

        # Run with bounded concurrency, append each result to the JSONL.
        done = 0
        t_start = time.time()
        with out_path.open("a") as fout, ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {
                pool.submit(gen_one, client, lang, q_idx, q_id, theme, question, run): (run, q_idx)
                for (run, q_idx, q_id, theme, question) in cells
            }
            for fut in as_completed(futures):
                rec = fut.result()
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                done += 1
                if done % 25 == 0 or done == len(cells):
                    elapsed = time.time() - t_start
                    rate = done / elapsed if elapsed > 0 else 0
                    eta = (len(cells) - done) / rate if rate > 0 else float("inf")
                    err_count = sum(1 for f, k in futures.items() if f.done() and (f.result().get("error") or not f.result().get("response","").strip()))
                    print(f"  [{lang}] {done:>4}/{len(cells)}  rate={rate:.2f}/s  eta={eta/60:.1f}m  errors={err_count}",
                          flush=True)


if __name__ == "__main__":
    main()
