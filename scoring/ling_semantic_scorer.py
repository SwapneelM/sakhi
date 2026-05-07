import os, logging, multiprocessing, warnings, re, torch, transformers, nltk, sacrebleu, pandas as pd
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Tuple
from bert_score import BERTScorer
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from rapidfuzz import fuzz, process
from openai import OpenAI
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)
transformers.logging.set_verbosity_error()

load_dotenv()

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
if not OPENROUTER_API_KEY:
    raise EnvironmentError(
        "OPENROUTER_API_KEY is not set. "
        "Please add it to your .env file as:\n  OPENROUTER_API_KEY=your_key_here"
    )

_REPO_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = str(_REPO_ROOT / "outputs" / "rubric_scores")
OUTPUT_DIR = str(_REPO_ROOT / "outputs" / "linguistic_semantic")
BATCH_SIZE = 256
NLTK_RESOURCES = ("punkt", "punkt_tab", "wordnet", "omw-1.4")

try:
    from indicnlp.tokenize import indic_tokenize
    from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
    INDIC_AVAILABLE = True
    print(">>> IndicNLP found and loaded — will use for Hindi/Marathi tokenization")
except ImportError:
    INDIC_AVAILABLE = False
    print(">>> IndicNLP NOT found — falling back to regex tokenization for Hindi/Marathi")


def make_openrouter_client():
    return OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")


def setup_environment():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for r in NLTK_RESOURCES:
        try: nltk.data.find(f"tokenizers/{r}" if "punkt" in r else f"corpora/{r}")
        except LookupError: nltk.download(r, quiet=True)
    try:
        from nltk.corpus import wordnet; wordnet.ensure_loaded()
    except: pass


def get_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


class MultilingualTokenizer:
    def __init__(self, lang: str = "hi"):
        self.lang = lang
        self._normalizer = None
        if INDIC_AVAILABLE:
            try:
                self._normalizer = IndicNormalizerFactory().get_normalizer(lang)
            except Exception:
                self._normalizer = None

    def tokenize(self, text: str) -> List[str]:
        if INDIC_AVAILABLE and self._normalizer is not None:
            try:
                text = self._normalizer.normalize(text)
                return indic_tokenize.trivial_tokenize(text, self.lang)
            except Exception:
                pass
        return re.findall(r'\w+', text)


def init_worker(use_stemmer: bool, lang: str = "hi"):
    global scorer_global, is_multilingual_global, lang_global
    is_multilingual_global = not use_stemmer
    lang_global = lang
    scorer_global = (
        rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        if use_stemmer
        else rouge_scorer.RougeScorer(["rougeL"], tokenizer=MultilingualTokenizer(lang))
    )
    try:
        from nltk.corpus import wordnet; wordnet.ensure_loaded()
    except: pass


def _indic_tokenize_text(text: str, lang: str) -> List[str]:
    if INDIC_AVAILABLE:
        try:
            from indicnlp.tokenize import indic_tokenize
            from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
            normalizer = IndicNormalizerFactory().get_normalizer(lang)
            return indic_tokenize.trivial_tokenize(normalizer.normalize(text.lower()), lang)
        except Exception:
            pass
    return re.findall(r'\w+', text.lower())


def _calc_linguistic_metrics(args: Tuple[str, str, List[str]]) -> Tuple[float, float, float, float]:
    ref, cand, tok_ref = args
    if not cand or not ref: return 0.0, 0.0, 0.0, 0.0
    global scorer_global, is_multilingual_global, lang_global
    if scorer_global is None:
        scorer_global = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    bleu = sacrebleu.sentence_bleu(cand, [ref]).score / 100
    chrf = sacrebleu.sentence_chrf(cand, [ref]).score / 100

    meteor = 0.0
    try:
        if is_multilingual_global:
            tok_cand     = _indic_tokenize_text(cand, lang_global)
            tok_ref_used = _indic_tokenize_text(ref,  lang_global)
        else:
            tok_cand     = nltk.word_tokenize(cand.lower())
            tok_ref_used = tok_ref
        meteor = meteor_score([tok_ref_used], tok_cand)
    except LookupError:
        try:
            from nltk.corpus import wordnet
            meteor = meteor_score([tok_ref_used], tok_cand, wordnet=wordnet)
        except: meteor = 0.0
    except: meteor = 0.0

    rougel = scorer_global.score(ref, cand)["rougeL"].fmeasure
    return bleu, chrf, meteor, rougel


def _fuzzy_match_chunk(chunk_data):
    start_idx, en_chunk, multi_q_target, thresh = chunk_data
    matches = {}
    for i, query in enumerate(en_chunk):
        res = process.extractOne(query, multi_q_target, scorer=fuzz.ratio, score_cutoff=thresh)
        if res:
            matches[start_idx + i] = res[2] if len(res) == 3 else multi_q_target.index(res[0])
    return matches


def precise_repair(df: pd.DataFrame, device: str, client) -> pd.DataFrame:
    print("   > Running precise repair for any 0.0 scores...")
    en_scorer    = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    multi_scorer_hi = rouge_scorer.RougeScorer(["rougeL"], tokenizer=MultilingualTokenizer("hi"))
    multi_scorer_mr = rouge_scorer.RougeScorer(["rougeL"], tokenizer=MultilingualTokenizer("mr"))

    metrics = [c for c in df.columns if '_rougel' in c or '_openai_sim' in c]
    for col in metrics:
        indices = df.index[(df[col] == 0) | (df[col].isna())].tolist()
        if not indices: continue

        base = col.replace('_openai_sim', '').replace('_rougel', '')
        lang, ref_col = 'en', 'answer'
        if base.endswith('_hi'):   lang, ref_col, base = 'hi', 'answer_hindi',   base[:-3]
        elif base.endswith('_mr'): lang, ref_col, base = 'mr', 'answer_marathi', base[:-3]
        elif 'ideal_answer' in df.columns: ref_col = 'ideal_answer'

        run_num = '1'
        match = re.search(r'(\d+)$', base)
        if match: run_num, model_name = match.group(1), base[:-len(match.group(1))]
        else: model_name = base
        model_name = model_name.rstrip('_').replace('_hindi', '').replace('_marathi', '')

        cand_col = None
        for c in df.columns:
            if c.startswith(model_name) and str(run_num) in c and ('response' in c or 'run' in c):
                cand_col = c; break
        if not cand_col:
            cand_col = base + "_run" if base + "_run" in df.columns else base
        if cand_col not in df.columns or ref_col not in df.columns: continue

        to_rep = [i for i in indices if str(df.at[i, ref_col]).strip() and str(df.at[i, cand_col]).strip()]
        if not to_rep: continue

        if '_rougel' in col:
            if lang == 'hi':   sc = multi_scorer_hi
            elif lang == 'mr': sc = multi_scorer_mr
            else:              sc = en_scorer
            for i in to_rep:
                df.at[i, col] = sc.score(str(df.at[i, ref_col]), str(df.at[i, cand_col]))["rougeL"].fmeasure

        elif '_openai_sim' in col and client:
            for i in range(0, len(to_rep), 64):
                bx = to_rep[i:i+64]
                try:
                    emb = client.embeddings.create(
                        input=[str(df.at[x, ref_col]) for x in bx] + [str(df.at[x, cand_col]) for x in bx],
                        model="openai/text-embedding-3-small"
                    )
                    v    = torch.tensor([e.embedding for e in emb.data], device=device)
                    sims = torch.nn.functional.cosine_similarity(v[:len(bx)], v[len(bx):]).cpu().tolist()
                    for j, val in enumerate(sims): df.at[bx[j], col] = val
                except Exception as e:
                    print(f"   [repair] embedding error: {e}")
    return df


class ScorerPipeline:
    def __init__(self):
        self.device = get_device()
        print(f">>> Models on {self.device}")
        self.scorer_en    = BERTScorer(model_type="roberta-base",     lang="en", device=self.device, rescale_with_baseline=True)
        self.scorer_multi = BERTScorer(model_type="xlm-roberta-base", device=self.device)
        self.client       = make_openrouter_client()

    def calculate_linguistic_scores(self, df: pd.DataFrame, ref_col: str, suffix: str = "",
                                    use_stemmer: bool = True, output_path: str = None,
                                    col_filter: str = None, lang: str = "hi") -> pd.DataFrame:
        print(f"   > Linguistic ({suffix if suffix else 'en'}) [lang={lang if not use_stemmer else 'en'}]...")
        rcols = [c for c in df.columns
                 if ("run" in c or "response" in c)
                 and not any(t in c for t in ["judge", "axis", "rubric", "expert_response", "edited_expert_response"])]
        if col_filter: rcols = [c for c in rcols if col_filter in c]

        refs     = df[ref_col].fillna("").astype(str).tolist()
        tok_refs = [nltk.word_tokenize(r.lower()) for r in refs]

        with ProcessPoolExecutor(max_workers=os.cpu_count() or 4,
                                 initializer=init_worker, initargs=(use_stemmer, lang)) as executor:
            for col in rcols:
                p = col.replace("_run", "").replace("_response", "") + suffix
                if f"{p}_linguistic_avg" in df.columns and df[f"{p}_linguistic_avg"].notna().any(): continue
                res = list(executor.map(
                    _calc_linguistic_metrics,
                    zip(refs, df[col].fillna("").astype(str).tolist(), tok_refs),
                    chunksize=100
                ))
                b, c, m, r = zip(*res)
                df[f"{p}_sacrebleu"]      = b
                df[f"{p}_chrf_pp"]        = c
                df[f"{p}_meteor"]         = m
                df[f"{p}_rougel"]         = r
                df[f"{p}_linguistic_avg"] = (pd.Series(b)+pd.Series(c)+pd.Series(m)+pd.Series(r)) / 4.0
                if output_path: df.loc[:, ~df.columns.duplicated()].to_csv(output_path, index=False)
        return df.loc[:, ~df.columns.duplicated()]

    def calculate_semantic_scores(self, df: pd.DataFrame, ref_col: str, scorer: BERTScorer,
                                  suffix: str = "", output_path: str = None,
                                  col_filter: str = None) -> pd.DataFrame:
        print(f"   > Semantic ({suffix if suffix else 'en'})...")
        rcols = [c for c in df.columns
                 if ("run" in c or "response" in c)
                 and not any(m in c for m in {"gemini_3_pro", "gemma_3_27b"})
                 and not any(t in c for t in ["judge", "axis", "rubric", "expert_response"])]
        if col_filter: rcols = [c for c in rcols if col_filter in c]

        refs = df[ref_col].fillna(" ").astype(str).tolist()
        for col in rcols:
            p = col.replace("_run", "").replace("_response", "") + suffix
            if f"{p}_semantic_avg" in df.columns and df[f"{p}_semantic_avg"].notna().any(): continue
            cands = df[col].fillna(" ").astype(str).tolist()
            sims, f1 = [], []
            for i in range(0, len(refs), BATCH_SIZE):
                rb, cb = refs[i:i+BATCH_SIZE], cands[i:i+BATCH_SIZE]
                try:
                    emb = self.client.embeddings.create(
                        input=[t.strip() or " " for t in rb + cb],
                        model="openai/text-embedding-3-small"
                    )
                    v = torch.tensor([e.embedding for e in emb.data], device=self.device)
                    sims.extend(torch.nn.functional.cosine_similarity(v[:len(rb)], v[len(rb):]).cpu().tolist())
                except Exception as e:
                    print(f"   [semantic] embedding error: {e}")
                    sims.extend([0.0] * len(rb))
                _, _, F = scorer.score(cb, rb)
                f1.extend(F.cpu().tolist())
            df[f"{p}_openai_sim"]   = sims
            df[f"{p}_bert_f1"]      = f1
            df[f"{p}_semantic_avg"] = (pd.Series(sims) + pd.Series(f1)) / 2.0
            if output_path: df.loc[:, ~df.columns.duplicated()].to_csv(output_path, index=False)
        return df.loc[:, ~df.columns.duplicated()]


def merge_semantic_parallel(df_en: pd.DataFrame, df_multi: pd.DataFrame, key: str = "question") -> pd.DataFrame:
    print(f"   > Merging on '{key}'...")
    en_q, multi_q = df_en[key].fillna("").tolist(), df_multi[key].fillna("").tolist()
    sz = (len(en_q) // (os.cpu_count() or 4)) + 1
    with ProcessPoolExecutor() as ex:
        matches = {k: v for d in ex.map(
            _fuzzy_match_chunk,
            [(i, en_q[i:i+sz], multi_q, 95) for i in range(0, len(en_q), sz)]
        ) for k, v in d.items()}
    if not matches: return pd.DataFrame()
    return pd.concat([
        df_en.iloc[list(matches.keys())].reset_index(drop=True),
        df_multi.drop(columns=[key]).iloc[list(matches.values())].reset_index(drop=True)
    ], axis=1)


def get_ref_col(df: pd.DataFrame, is_expert: bool = False) -> str:
    if is_expert and "ideal_answer" in df.columns:
        return "ideal_answer"
    return "answer" if "answer" in df.columns else None


def get_multilingual_ref_cols(df: pd.DataFrame):
    hindi_col   = next((c for c in ["answer_hindi",   "answer_hi"] if c in df.columns), None)
    marathi_col = next((c for c in ["answer_marathi", "answer_mr"] if c in df.columns), None)
    return hindi_col, marathi_col


def main():
    setup_environment()
    pipe = ScorerPipeline()

    fs = {
        "expert_english":          "rubric_scores_expert_english_gpt.csv",
        "expert_multilingual":     "rubric_scores_expert_multilingual_gpt.csv",
        "non_expert_english":      "rubric_scores_non_expert_english_gpt.csv",
        "non_expert_multilingual": "rubric_scores_non_expert_multilingual_gpt.csv",
    }

    ds = {}
    print(">>> Loading...")
    for k, f in fs.items():
        fp = os.path.join(INPUT_DIR, f)
        if os.path.exists(fp):
            ds[k] = pd.read_csv(fp)
            print(f"   Loaded {k}: {len(ds[k])} rows from {f}")
        else:
            print(f"   Skipping (not found): {fp}")

    for k, df in ds.items():
        ds[k] = df.drop(columns=[
            c for c in df.columns
            if ("hindi" in c and "_mr" in c) or ("marathi" in c and "_hi" in c)
        ]).loc[:, ~df.columns.duplicated()]

    print(">>> Linguistic Scores...")

    if "expert_english" in ds:
        ref = get_ref_col(ds["expert_english"], is_expert=True)
        if ref: ds["expert_english"] = pipe.calculate_linguistic_scores(ds["expert_english"], ref, lang="en")

    if "non_expert_english" in ds:
        ref = get_ref_col(ds["non_expert_english"], is_expert=False)
        if ref: ds["non_expert_english"] = pipe.calculate_linguistic_scores(ds["non_expert_english"], ref, lang="en")

    if "expert_multilingual" in ds:
        hi_col, mr_col = get_multilingual_ref_cols(ds["expert_multilingual"])
        if hi_col:
            ds["expert_multilingual"] = pipe.calculate_linguistic_scores(
                ds["expert_multilingual"], hi_col, "_hi", use_stemmer=False, col_filter="hindi", lang="hi")
        if mr_col:
            ds["expert_multilingual"] = pipe.calculate_linguistic_scores(
                ds["expert_multilingual"], mr_col, "_mr", use_stemmer=False, col_filter="marathi", lang="mr")

    if "non_expert_multilingual" in ds:
        hi_col, mr_col = get_multilingual_ref_cols(ds["non_expert_multilingual"])
        if hi_col:
            ds["non_expert_multilingual"] = pipe.calculate_linguistic_scores(
                ds["non_expert_multilingual"], hi_col, "_hi", use_stemmer=False, col_filter="hindi", lang="hi")
        if mr_col:
            ds["non_expert_multilingual"] = pipe.calculate_linguistic_scores(
                ds["non_expert_multilingual"], mr_col, "_mr", use_stemmer=False, col_filter="marathi", lang="mr")

    print(">>> Semantic Scores...")

    if "expert_english" in ds:
        ref = get_ref_col(ds["expert_english"], is_expert=True)
        if ref: ds["expert_english"] = pipe.calculate_semantic_scores(ds["expert_english"], ref, pipe.scorer_en)

    if "non_expert_english" in ds:
        ref = get_ref_col(ds["non_expert_english"], is_expert=False)
        if ref: ds["non_expert_english"] = pipe.calculate_semantic_scores(ds["non_expert_english"], ref, pipe.scorer_en)

    if "expert_multilingual" in ds:
        hi_col, mr_col = get_multilingual_ref_cols(ds["expert_multilingual"])
        if hi_col:
            ds["expert_multilingual"] = pipe.calculate_semantic_scores(
                ds["expert_multilingual"], hi_col, pipe.scorer_multi, "_hi", col_filter="hindi")
        if mr_col:
            ds["expert_multilingual"] = pipe.calculate_semantic_scores(
                ds["expert_multilingual"], mr_col, pipe.scorer_multi, "_mr", col_filter="marathi")

    if "non_expert_multilingual" in ds:
        hi_col, mr_col = get_multilingual_ref_cols(ds["non_expert_multilingual"])
        if hi_col:
            ds["non_expert_multilingual"] = pipe.calculate_semantic_scores(
                ds["non_expert_multilingual"], hi_col, pipe.scorer_multi, "_hi", col_filter="hindi")
        if mr_col:
            ds["non_expert_multilingual"] = pipe.calculate_semantic_scores(
                ds["non_expert_multilingual"], mr_col, pipe.scorer_multi, "_mr", col_filter="marathi")

    print(">>> Merging and Repairing...")

    def fin(k, df):
        df = precise_repair(df, pipe.device, pipe.client)
        p = os.path.join(OUTPUT_DIR, f"{k}_scored.csv")
        df.to_csv(p, index=False)
        print(f"   Saved {k}_scored.csv")

    if "expert_english" in ds and "expert_multilingual" in ds:
        fin("expert", merge_semantic_parallel(ds["expert_english"], ds["expert_multilingual"]))
    elif "expert_english" in ds:
        fin("expert_english", ds["expert_english"])
    elif "expert_multilingual" in ds:
        fin("expert_multilingual", ds["expert_multilingual"])

    if "non_expert_english" in ds and "non_expert_multilingual" in ds:
        fin("non_expert", merge_semantic_parallel(ds["non_expert_english"], ds["non_expert_multilingual"]))
    elif "non_expert_english" in ds:
        fin("non_expert_english", ds["non_expert_english"])
    elif "non_expert_multilingual" in ds:
        fin("non_expert_multilingual", ds["non_expert_multilingual"])

    print(f">>> All Done. Output Directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()