"""Linguistic + semantic metrics per response. Local only (no judge API).
  - SacreBLEU, chrF++, ROUGE-L, METEOR (when EN)
  - BERTScore F1 (xlm-roberta-base for HI/MR, roberta-base with baseline rescaling for EN)
  - OpenAI text-embedding-3-small cosine similarity

Reads:  runs/gen__<model>__<dataset>__<lang>.jsonl
Writes: runs/metrics__<model>__<dataset>__<lang>.jsonl  (one row per response, keyed by (run, q_idx))

Resumes safely. Embeddings are cached per (q_idx) reference so we don't re-embed
the same reference text for every model.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
from typing import Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scoring.pipeline import checkpoint
from scoring.pipeline.generate import DATASET_CONFIG

REPO = Path(__file__).resolve().parent.parent.parent
RUNS_DIR = REPO / "runs"
CACHE_DIR = RUNS_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)

_sacrebleu = None
_rouge = None
_bert_en = None
_bert_ml = None
_openai_client = None


def lazy_sacrebleu():
    global _sacrebleu
    if _sacrebleu is None:
        import sacrebleu as sb
        _sacrebleu = sb
    return _sacrebleu


def lazy_rouge():
    global _rouge
    if _rouge is None:
        from rouge_score import rouge_scorer
        _rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return _rouge


def lazy_bert(lang: str):
    global _bert_en, _bert_ml
    from bert_score import BERTScorer
    if lang == "en":
        if _bert_en is None:
            _bert_en = BERTScorer(lang="en", rescale_with_baseline=True)
        return _bert_en
    if _bert_ml is None:
        _bert_ml = BERTScorer(model_type="xlm-roberta-base", num_layers=9)
    return _bert_ml


def lazy_openai():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI()
    return _openai_client


def meteor_score_en(hyp: str, ref: str) -> float:
    import nltk
    for res in ("wordnet", "omw-1.4", "punkt"):
        try:
            nltk.data.find(res if res.endswith(".zip") else f"tokenizers/{res}")
        except LookupError:
            try:
                nltk.download(res, quiet=True)
            except Exception:
                pass
    from nltk.translate.meteor_score import meteor_score as ms
    from nltk.tokenize import wordpunct_tokenize
    return float(ms([wordpunct_tokenize(ref)], wordpunct_tokenize(hyp)))


def cosine(a: list[float], b: list[float]) -> float:
    import math
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def load_references(dataset: str, lang: str) -> dict[int, str]:
    cfg = DATASET_CONFIG[dataset]
    df = pd.read_csv(cfg["path"])
    ref_col = cfg["ref_cols"][lang]
    return {int(i): (str(r[ref_col]) if pd.notna(r[ref_col]) else "") for i, r in df.iterrows()}


def load_ref_embeddings(dataset: str, lang: str, refs: dict[int, str]) -> dict[int, list[float]]:
    cache_path = CACHE_DIR / f"emb_refs__{dataset}__{lang}.json"
    cache = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    missing = {i: t for i, t in refs.items() if str(i) not in cache and t.strip()}
    if missing:
        client = lazy_openai()
        items = list(missing.items())
        B = 128
        for i in range(0, len(items), B):
            batch = items[i:i + B]
            resp = client.embeddings.create(model="text-embedding-3-small", input=[t for _, t in batch])
            for (idx, _), d in zip(batch, resp.data):
                cache[str(idx)] = d.embedding
        cache_path.write_text(json.dumps(cache), encoding="utf-8")
    return {int(k): v for k, v in cache.items()}


def embed_hypotheses(texts: list[str]) -> list[list[float]]:
    client = lazy_openai()
    B = 128
    out: list[list[float]] = []
    for i in range(0, len(texts), B):
        batch = texts[i:i + B]
        resp = client.embeddings.create(model="text-embedding-3-small", input=[t if t.strip() else " " for t in batch])
        out.extend([d.embedding for d in resp.data])
    return out


def run(args):
    gen_path = RUNS_DIR / f"gen__{args.gen_model}__{args.dataset}__{args.lang}.jsonl"
    out_path = RUNS_DIR / f"metrics__{args.gen_model}__{args.dataset}__{args.lang}.jsonl"
    if not gen_path.exists():
        print(f"FATAL: gen file missing: {gen_path}", file=sys.stderr); return 1

    key_fields = ("run", "q_idx")
    done = checkpoint.completed_keys(str(out_path), key_fields)
    rows = [r for r in checkpoint.jsonl_rows(str(gen_path))
            if not r.get("error") and (r.get("response") or "").strip()]
    pending = [r for r in rows if (r.get("run"), r.get("q_idx")) not in done]
    print(f"[metrics {args.gen_model} {args.dataset} {args.lang}] total={len(rows)} pending={len(pending)}", flush=True)
    if not pending:
        return 0

    refs = load_references(args.dataset, args.lang)

    # Embeddings (batched, cached refs)
    try:
        ref_embs = load_ref_embeddings(args.dataset, args.lang, refs)
        hyp_embs = embed_hypotheses([r.get("response", "") for r in pending])
    except Exception as e:
        print(f"[metrics] embedding failed, continuing without: {e}", flush=True)
        ref_embs = {}
        hyp_embs = [None] * len(pending)

    # BERTScore (batched)
    try:
        scorer = lazy_bert(args.lang)
        cands = [r.get("response", "") for r in pending]
        refs_list = [refs.get(int(r["q_idx"]), "") for r in pending]
        P, R, F1 = scorer.score(cands, refs_list)
        bert_f1 = [float(x) for x in F1.tolist()]
    except Exception as e:
        print(f"[metrics] BERTScore failed, continuing without: {e}", flush=True)
        bert_f1 = [None] * len(pending)

    sb = lazy_sacrebleu()
    rouge = lazy_rouge()

    for idx, row in enumerate(pending):
        q_idx = int(row["q_idx"])
        hyp = row.get("response", "") or ""
        ref = refs.get(q_idx, "") or ""
        if not ref.strip():
            continue
        bleu = float(sb.sentence_bleu(hyp, [ref]).score) / 100.0
        chrfpp = float(sb.sentence_chrf(hyp, [ref], word_order=2).score) / 100.0
        rL = float(rouge.score(ref, hyp)["rougeL"].fmeasure)
        meteor = meteor_score_en(hyp, ref) if args.lang == "en" else None
        emb_sim = None
        if hyp_embs[idx] is not None and q_idx in ref_embs:
            emb_sim = cosine(hyp_embs[idx], ref_embs[q_idx])
        checkpoint.append(str(out_path), {
            "gen_model": args.gen_model, "dataset": args.dataset, "lang": args.lang,
            "run": row.get("run"), "q_idx": q_idx, "theme": row.get("theme", ""),
            "sacrebleu": bleu, "chrf_pp": chrfpp, "rouge_l": rL,
            "meteor": meteor, "bert_f1": bert_f1[idx], "emb_sim": emb_sim,
        })
    print(f"[metrics {args.gen_model} {args.dataset} {args.lang}] done", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-model", required=True)
    ap.add_argument("--dataset", required=True, choices=["expert", "non_expert"])
    ap.add_argument("--lang", required=True, choices=["en", "hi", "mr"])
    args = ap.parse_args()
    sys.exit(run(args))


if __name__ == "__main__":
    main()
