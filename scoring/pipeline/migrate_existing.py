"""One-time migration: extract pre-generated model responses from llm_test_data_*.csv
into the per (model × dataset × lang) JSONL format used by the rest of the pipeline.

Applies only to the non_expert dataset (231 Qs). Models:
  cohere_command_a, gpt_5_mini, gpt_4o_mini, llama_3_3_70b, llama_4_maverick,
  aya_expanse, medgemma_4b, medgemma_27b.

Stale Gemini 2.5 columns are skipped — those will be freshly generated as Gemini 3 variants.
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parent.parent.parent
DATA_EN = REPO / "data2" / "llm_test_data_english_230.csv"
DATA_ML = REPO / "data2" / "llm_test_data_multilingual_230.csv"
DATA_NE = REPO / "data2" / "sakhi_non_expert_raw_230.csv"
RUNS_DIR = REPO / "runs"
RUNS_DIR.mkdir(exist_ok=True)

MODELS = [
    "cohere_command_a",
    "gpt_5_mini",
    "gpt_4o_mini",
    "llama_3_3_70b",
    "llama_4_maverick",
    "aya_expanse",
    "medgemma_4b",
    "medgemma_27b",
]
# Note: stale Gemini 2.5 columns are NOT migrated; we re-generate as Gemini 3 variants.

LANG_CONFIG = {
    "en": {"csv": DATA_EN, "q_col": "question", "suffix": ""},
    "hi": {"csv": DATA_ML, "q_col": "questions_hindi", "suffix": "_hindi"},
    "mr": {"csv": DATA_ML, "q_col": "questions_marathi", "suffix": "_marathi"},
}


def migrate_one(model: str, lang: str) -> int:
    cfg = LANG_CONFIG[lang]
    df = pd.read_csv(cfg["csv"])
    out_path = RUNS_DIR / f"gen__{model}__non_expert__{lang}.jsonl"
    if out_path.exists():
        out_path.unlink()
    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for i, row in df.iterrows():
            question = row.get(cfg["q_col"])
            if pd.isna(question) or not str(question).strip():
                continue
            for run in (1, 2, 3):
                col = f"{model}{cfg['suffix']}_run{run}"
                if col not in df.columns:
                    continue
                resp = row.get(col)
                if pd.isna(resp) or not str(resp).strip():
                    continue
                rec = {
                    "model": model, "provider": "migrated", "run": run,
                    "dataset": "non_expert", "lang": lang,
                    "q_idx": int(i), "q_id": str(i),
                    "theme": str(row.get("theme", "")),
                    "question": str(question).strip(),
                    "response": str(resp).strip(),
                    "migrated_from": str(cfg["csv"].name),
                    "ts": "2026-04-21T00:00:00Z",
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                count += 1
    return count


def main():
    total = 0
    for m in MODELS:
        for lang in ("en", "hi", "mr"):
            n = migrate_one(m, lang)
            print(f"{m} {lang}: {n} rows -> gen__{m}__non_expert__{lang}.jsonl")
            total += n
    print(f"\nMigrated {total} rows total across {len(MODELS)} models × 3 langs.")


if __name__ == "__main__":
    main()
