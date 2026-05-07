# Sakhi: Replication code

Code release for the Sakhi maternal-health LLM benchmark. The benchmark itself, doctor calibration verdicts, rubric, and Croissant metadata are published as a HuggingFace dataset:

**Dataset:** https://huggingface.co/datasets/SimPPL/sakhi

This repository contains only the replication code: the generation pipeline (per-model response generation), the LLM-as-judge scoring pipeline, the rubric implementation, the linguistic and embedding-similarity metrics, and the figure / table renderers. All API keys must be supplied by the caller via `.env`.

## Layout

```
data2/             canonical raw inputs (the source CSVs that release/ is built from)
generation/        legacy entry-point scripts (English + multilingual generation)
graphs_code/       legacy figure / theme-graph scripts
release/           the published benchmark and aggregated results
  ├── sakhi_benchmark_expert.csv             149 doctor-edited Q-A pairs (EN/HI/MR)
  ├── sakhi_benchmark_non_expert.csv         231 community-sourced Q-A pairs (EN/HI/MR)
  ├── sakhi_doctor_ratings.csv               2,103 binary criterion-level verdicts from 11 doctors
  ├── sakhi_rubrics.json                     14-criterion rubric across 5 axes
  ├── sakhi_croissant.json                   Croissant 1.0 metadata
  ├── DATASHEET.md                           Datasheet for Datasets
  ├── LICENSE                                CC BY 4.0 (data)
  ├── results/                               per-model x dataset x language MQS aggregates
  └── hf_dataset/                            parquet-flavoured copy used for HF upload
scoring/
  ├── scoring_rubric.py                      RUBRIC_MAP, AXIS_MAP, calc_m, axis weights
  ├── ling_semantic_scorer.py                BLEU / METEOR / ROUGE-L / BERTScore / embedding sim
  └── pipeline/                              canonical pipeline (generate, judge, aggregate, build)
prompts.txt                                  knowledge-base prompt templates
requirements.txt                             Python dependencies
```

## Reproducing the paper numbers

The aggregated MQS table in the paper is reproducible from the released judge JSONLs (which live on HuggingFace) and `scoring/pipeline/aggregate.py`:

```bash
pip install -r requirements.txt
# Pull the dataset (questions + rubric + leaderboard subset)
python -c "from datasets import load_dataset; load_dataset('SimPPL/sakhi', 'expert', split='test')"
# Re-aggregate from raw judge JSONLs (if you have them locally under runs/)
python -m scoring.pipeline.aggregate
```

The canonical Medical Quality Score (MQS) computation is in `scoring/scoring_rubric.py` (`calc_m`); axis weights are `Accuracy 0.30 / Completeness 0.25 / Context Awareness 0.20 / Communication 0.15 / Terminology Accessibility 0.10`.

## Running the full pipeline end-to-end (requires API credits)

You will need API access to the model providers used in the paper. Set up `.env` with the keys you have:

```
OPENROUTER_API_KEY=...
COHERE_API_KEY=...
HF_TOKEN=...
```

Then for one (model, dataset arm, language) cell:

```bash
# 1. Generate model responses (writes runs/gen__<model>__<arm>__<lang>.jsonl)
python -m scoring.pipeline.generate --model gpt-4o-mini --dataset expert --lang en --num-runs 3

# 2. Score with the rubric judge (writes runs/judge__<judge>__<model>__<arm>__<lang>.jsonl)
python -m scoring.pipeline.judge --judge-model gpt-4o-mini --gen-model gpt_4o_mini --dataset expert --lang en

# 3. Aggregate to MQS-per-cell
python -m scoring.pipeline.aggregate
```

Specialised scripts cover providers without OpenRouter routes:
- `scoring/pipeline/generate_cohere_native.py` — native Cohere route (Aya Expanse, Command A)
- `scoring/pipeline/generate_medgemma_hf.py` — HuggingFace `featherless-ai` provider (MedGemma)

## Rebuilding the release artefacts

```bash
# Top-level CSV release + Croissant
python -m scoring.pipeline.build_release

# HuggingFace parquet release + Croissant
python -m scoring.pipeline.build_hf_release --repo-id SimPPL/sakhi
```

## License

Code in this repository is released under the MIT license (`LICENSE`). The benchmark data, rubric, and doctor verdicts in `release/` are released under CC BY 4.0 (`release/LICENSE`).

## Citation

See the dataset card on HuggingFace (https://huggingface.co/datasets/SimPPL/sakhi) for the canonical citation. Author and institution details are withheld during the review period.
