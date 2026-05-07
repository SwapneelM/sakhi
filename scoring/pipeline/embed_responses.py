"""Batched, resumable embedding writer for Sakhi model responses.

Reads each ``runs/gen__<model>__<dataset>__<lang>.jsonl`` from a single run
(default ``run==1``), embeds each non-empty response with
``openai/text-embedding-3-small`` via the OpenRouter embeddings endpoint,
and appends the embeddings to a per-run parquet so that a crash mid-run
does not throw away tokens already paid for.

Resumes by skipping any (gen_model, dataset, lang, q_idx) key already
present in the parquet output.

Usage:
    python -m scoring.pipeline.embed_responses --run 1 --batch 100

Cost (rough): 13 models * 380 questions * 3 langs = ~14.8k responses,
~150 tokens each -> ~2.2M tokens -> ~$0.04 on text-embedding-3-small.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

REPO = Path(__file__).resolve().parent.parent.parent
RUNS_DIR = REPO / "runs"
EMBED_DIR = REPO / "data" / "embeddings"
EMBED_DIR.mkdir(parents=True, exist_ok=True)

# Load OpenRouter key from ~/.env.openrouter, matching the rest of the project.
for p in (Path.home() / ".env.openrouter", Path.home() / ".env"):
    if p.exists():
        load_dotenv(p, override=False)

EMBED_MODEL = "openai/text-embedding-3-small"
EMBED_DIM = 1536
OR_URL = "https://openrouter.ai/api/v1/embeddings"


def _gen_files(run: int) -> list[Path]:
    return sorted(RUNS_DIR.glob("gen__*.jsonl"))


def _parse_filename(path: Path) -> tuple[str, str, str]:
    # gen__<gen_model>__<dataset>__<lang>.jsonl
    stem = path.stem
    parts = stem.split("__")
    if len(parts) != 4 or parts[0] != "gen":
        raise ValueError(f"unexpected gen filename: {path.name}")
    return parts[1], parts[2], parts[3]


def _load_existing_keys(out_path: Path) -> set[tuple]:
    if not out_path.exists():
        return set()
    df = pd.read_parquet(out_path, columns=["gen_model", "dataset", "lang", "q_idx"])
    return set(zip(df["gen_model"], df["dataset"], df["lang"], df["q_idx"]))


def _append_parquet(out_path: Path, batch: list[dict]) -> None:
    if not batch:
        return
    new_df = pd.DataFrame(batch)
    if out_path.exists():
        existing = pd.read_parquet(out_path)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_parquet(out_path, index=False)


def _read_run1_responses(paths: list[Path], run: int) -> list[dict]:
    """Yield (gen_model, dataset, lang, q_idx, q_id, theme, response) tuples."""
    import json
    out: list[dict] = []
    for p in paths:
        gen_model, dataset, lang = _parse_filename(p)
        # Dedupe within file: keep latest per (run, q_idx)
        latest: dict[tuple, dict] = {}
        with p.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if str(r.get("run")) != str(run):
                    continue
                if r.get("error") or not (r.get("response") or "").strip():
                    continue
                k = (str(r.get("run")), int(r.get("q_idx", -1)))
                latest[k] = r
        for r in latest.values():
            out.append({
                "gen_model": gen_model,
                "dataset": dataset,
                "lang": lang,
                "run": int(r.get("run") or 1),
                "q_idx": int(r.get("q_idx", -1)),
                "q_id": str(r.get("q_id", "")),
                "theme": str(r.get("theme", "")),
                "response": str(r.get("response", "")),
            })
    return out


def _embed_via_openrouter(texts: list[str], api_key: str, timeout: int = 60) -> list[list[float]]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/SwapneelM/MedicaLLM",
        "X-Title": "Sakhi Embed",
    }
    r = requests.post(OR_URL, headers=headers,
                      json={"model": EMBED_MODEL, "input": texts},
                      timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"OpenRouter {r.status_code}: {r.text[:300]}")
    data = r.json()
    rows = data.get("data") or []
    return [d["embedding"] for d in rows]


def embed_run(run: int = 1, batch_size: int = 100, max_retries: int = 4) -> Path:
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip().strip('"\'')
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY not set in environment")

    out_path = EMBED_DIR / f"embeddings_run{run}.parquet"
    done = _load_existing_keys(out_path)
    print(f"[embed_responses] resume: {len(done)} responses already embedded", flush=True)

    rows = _read_run1_responses(_gen_files(run), run)
    pending = [r for r in rows
               if (r["gen_model"], r["dataset"], r["lang"], r["q_idx"]) not in done]
    print(f"[embed_responses] total={len(rows)}  pending={len(pending)}", flush=True)
    if not pending:
        return out_path

    written = 0
    for i in range(0, len(pending), batch_size):
        chunk = pending[i:i + batch_size]
        texts = [r["response"] for r in chunk]
        last_err = None
        for attempt in range(max_retries):
            try:
                vectors = _embed_via_openrouter(texts, api_key)
                if len(vectors) != len(chunk):
                    raise RuntimeError(f"len mismatch: {len(vectors)} vs {len(chunk)}")
                batch = []
                for r, v in zip(chunk, vectors):
                    batch.append({
                        "gen_model": r["gen_model"],
                        "dataset": r["dataset"],
                        "lang": r["lang"],
                        "run": r["run"],
                        "q_idx": r["q_idx"],
                        "q_id": r["q_id"],
                        "theme": r["theme"],
                        "response": r["response"],
                        "embedding": v,
                    })
                _append_parquet(out_path, batch)
                written += len(batch)
                print(f"[embed_responses]  batch {i // batch_size + 1}: wrote +{len(batch)}, "
                      f"running total written: {written}", flush=True)
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
                wait = min(2 ** attempt, 30)
                print(f"[embed_responses]  attempt {attempt + 1}/{max_retries} failed: "
                      f"{str(e)[:200]}; sleeping {wait}s", flush=True)
                time.sleep(wait)
        else:
            print(f"[embed_responses]  giving up on chunk after {max_retries} retries: "
                  f"{last_err}", flush=True)
    return out_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", type=int, default=1, help="Run index to embed (default 1)")
    ap.add_argument("--batch", type=int, default=100,
                    help="Embeddings per OpenAI request (default 100)")
    ap.add_argument("--retries", type=int, default=4)
    args = ap.parse_args(argv)

    out = embed_run(run=args.run, batch_size=args.batch, max_retries=args.retries)
    print(f"[embed_responses] done; output: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
