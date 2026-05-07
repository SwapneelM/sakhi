"""Assemble the Sakhi HuggingFace dataset directory from the canonical CSVs.

Run after build_release.py and pipeline/aggregate.py have produced the
canonical artefacts under release/. This script is deterministic: same
inputs always give the same parquet bytes and the same Croissant MD5s.

Outputs:
  release/hf_dataset/
    data/expert.parquet
    data/non_expert.parquet
    data/doctor_ratings.parquet
    data/results.parquet           (per-model MQS, derived from release/results)
    rubrics.json                   (copy of release/sakhi_rubrics.json)
    croissant.json                 (regenerated, parquet-aware, fresh MD5s)
    README.md                      (HF dataset card)

Usage:
  python -m scoring.pipeline.build_hf_release [--repo-id SimPPL/sakhi]
"""
from __future__ import annotations
import argparse, hashlib, json, shutil
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RELEASE = REPO / "release"
HF_DIR = RELEASE / "hf_dataset"
DATA_DIR = HF_DIR / "data"

PARQUET_KW = dict(index=False, compression="snappy", engine="pyarrow")


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def write_parquet(df: pd.DataFrame, name: str) -> Path:
    out = DATA_DIR / name
    df.to_parquet(out, **PARQUET_KW)
    return out


def build_parquets() -> dict[str, Path]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    expert = pd.read_csv(RELEASE / "sakhi_benchmark_expert.csv")
    non_expert = pd.read_csv(RELEASE / "sakhi_benchmark_non_expert.csv")
    doctor = pd.read_csv(RELEASE / "sakhi_doctor_ratings.csv")

    out = {}
    out["expert"] = write_parquet(expert, "expert.parquet")
    out["non_expert"] = write_parquet(non_expert, "non_expert.parquet")
    out["doctor_ratings"] = write_parquet(doctor, "doctor_ratings.parquet")
    return out


def build_results_parquet() -> Path:
    """Bundle the per-model MQS aggregates into a small benchmark-leaderboard parquet."""
    src = RELEASE / "results" / "mqs_per_model__gpt_4o_mini.csv"
    df = pd.read_csv(src)
    df = df[["gen_model", "dataset", "lang", "n", "mqs_mean", "mqs_std"]]
    df = df.rename(columns={"gen_model": "model"})
    df["judge"] = "gpt-4o-mini"
    return write_parquet(df, "results.parquet")


def copy_rubrics() -> Path:
    src = RELEASE / "sakhi_rubrics.json"
    dst = HF_DIR / "rubrics.json"
    shutil.copyfile(src, dst)
    return dst


def field(rid: str, dtype: str, parquet_id: str, column: str) -> dict:
    return {
        "@type": "cr:Field",
        "@id": rid,
        "dataType": dtype,
        "source": {
            "fileObject": {"@id": parquet_id},
            "extract": {"column": column},
        },
    }


def build_croissant(parquets: dict[str, Path], results_path: Path,
                    rubrics_path: Path, repo_id: str) -> Path:
    repo_url = f"https://huggingface.co/datasets/{repo_id}"

    file_objects = []
    for key, p in parquets.items():
        file_objects.append({
            "@type": "cr:FileObject",
            "@id": f"{key}-parquet",
            "name": f"data/{p.name}",
            "description": {
                "expert": "Expert arm: 149 doctor-edited Q&A pairs in EN/HI/MR.",
                "non_expert": "Non-expert arm: 231 community-sourced Q&A pairs with doctor-reviewed reference answers, in EN/HI/MR.",
                "doctor_ratings": "2,103 binary rubric verdicts on 148 expert-arm questions, supplied by 11 practising Indian doctors.",
            }[key],
            "encodingFormat": "application/x-parquet",
            "contentUrl": f"data/{p.name}",
            "md5": md5(p),
        })
    file_objects.append({
        "@type": "cr:FileObject",
        "@id": "results-parquet",
        "name": f"data/{results_path.name}",
        "description": "Per-model MQS aggregates produced by the GPT-4o-mini judge across all 11 generation models, two arms, and three languages.",
        "encodingFormat": "application/x-parquet",
        "contentUrl": f"data/{results_path.name}",
        "md5": md5(results_path),
    })
    file_objects.append({
        "@type": "cr:FileObject",
        "@id": "rubrics-json",
        "name": "rubrics.json",
        "description": "10 themes x 5 axes x 3 binary criteria rubric used to score model responses.",
        "encodingFormat": "application/json",
        "contentUrl": "rubrics.json",
        "md5": md5(rubrics_path),
    })

    benchmark_fields = [
        ("q_id", "sc:Text"),
        ("theme", "sc:Text"),
        ("domain", "sc:Text"),
        ("question_en", "sc:Text"),
        ("question_hi", "sc:Text"),
        ("question_mr", "sc:Text"),
        ("answer_en", "sc:Text"),
        ("answer_hi", "sc:Text"),
        ("answer_mr", "sc:Text"),
    ]

    record_sets = [
        {
            "@type": "cr:RecordSet",
            "@id": "expert-records",
            "name": "Expert arm records",
            "description": "Doctor-edited reference answers parallel across English, Hindi, and Marathi.",
            "field": [field(f"expert/{c}", t, "expert-parquet", c) for c, t in benchmark_fields]
                     + [field("expert/sources", "sc:Text", "expert-parquet", "sources")],
        },
        {
            "@type": "cr:RecordSet",
            "@id": "non-expert-records",
            "name": "Non-expert arm records",
            "description": "Community-sourced reference answers reviewed by ASHA workers and nonprofit staff, parallel across English, Hindi, and Marathi.",
            "field": [field(f"non_expert/{c}", t, "non_expert-parquet", c) for c, t in benchmark_fields]
                     + [field("non_expert/references", "sc:Text", "non_expert-parquet", "references")],
        },
        {
            "@type": "cr:RecordSet",
            "@id": "doctor-ratings-records",
            "name": "Doctor calibration verdicts",
            "description": "One row per (reviewer, question, rubric criterion). Reviewer pseudonyms run R1 to R11; reviewer roles are OB/GYN or General Practitioner. The verdict column is binary (pass/fail) on the same 14-criterion rubric the LLM judges use.",
            "field": [
                field("doc/doctor_id", "sc:Text", "doctor_ratings-parquet", "doctor_id"),
                field("doc/doctor_role", "sc:Text", "doctor_ratings-parquet", "doctor_role"),
                field("doc/doctor_experience", "sc:Text", "doctor_ratings-parquet", "doctor_experience"),
                field("doc/doctor_ai_exposure", "sc:Text", "doctor_ratings-parquet", "doctor_ai_exposure"),
                field("doc/question_id", "sc:Text", "doctor_ratings-parquet", "question_id"),
                field("doc/question_text", "sc:Text", "doctor_ratings-parquet", "question_text"),
                field("doc/ai_response", "sc:Text", "doctor_ratings-parquet", "ai_response"),
                field("doc/rubric_text", "sc:Text", "doctor_ratings-parquet", "rubric_text"),
                field("doc/axis", "sc:Text", "doctor_ratings-parquet", "axis"),
                field("doc/verdict", "sc:Text", "doctor_ratings-parquet", "verdict"),
                field("doc/theme", "sc:Text", "doctor_ratings-parquet", "theme"),
                field("doc/domain", "sc:Text", "doctor_ratings-parquet", "domain"),
                field("doc/references", "sc:Text", "doctor_ratings-parquet", "references"),
            ],
        },
        {
            "@type": "cr:RecordSet",
            "@id": "results-records",
            "name": "Per-model MQS leaderboard",
            "description": "GPT-4o-mini-judged MQS for each (model, dataset, language) cell. n is the number of model responses that received valid judge verdicts; mqs_mean and mqs_std are computed over those.",
            "field": [
                field("res/model", "sc:Text", "results-parquet", "model"),
                field("res/dataset", "sc:Text", "results-parquet", "dataset"),
                field("res/lang", "sc:Text", "results-parquet", "lang"),
                field("res/n", "sc:Integer", "results-parquet", "n"),
                field("res/mqs_mean", "sc:Float", "results-parquet", "mqs_mean"),
                field("res/mqs_std", "sc:Float", "results-parquet", "mqs_std"),
                field("res/judge", "sc:Text", "results-parquet", "judge"),
            ],
        },
    ]

    croissant = {
        "@context": {
            "@language": "en",
            "@vocab": "https://schema.org/",
            "citeAs": "cr:citeAs",
            "column": "cr:column",
            "conformsTo": "dct:conformsTo",
            "cr": "http://mlcommons.org/croissant/",
            "rai": "http://mlcommons.org/croissant/RAI/",
            "data": {"@id": "cr:data", "@type": "@json"},
            "dataType": {"@id": "cr:dataType", "@type": "@vocab"},
            "dct": "http://purl.org/dc/terms/",
            "examples": {"@id": "cr:examples", "@type": "@json"},
            "extract": "cr:extract",
            "field": "cr:field",
            "fileProperty": "cr:fileProperty",
            "fileObject": "cr:fileObject",
            "fileSet": "cr:fileSet",
            "format": "cr:format",
            "includes": "cr:includes",
            "isLiveDataset": "cr:isLiveDataset",
            "jsonPath": "cr:jsonPath",
            "key": "cr:key",
            "md5": "cr:md5",
            "parentField": "cr:parentField",
            "path": "cr:path",
            "recordSet": "cr:recordSet",
            "references": "cr:references",
            "regex": "cr:regex",
            "repeated": "cr:repeated",
            "replace": "cr:replace",
            "sc": "https://schema.org/",
            "separator": "cr:separator",
            "source": "cr:source",
            "subField": "cr:subField",
            "transform": "cr:transform",
        },
        "@type": "sc:Dataset",
        "name": "Sakhi",
        "alternateName": ["sakhi-benchmark"],
        "description": "Sakhi is a parallel English, Hindi, and Marathi maternal and reproductive-health benchmark with two reference-answer tracks (149 doctor-edited expert pairs; 231 community-sourced non-expert pairs). It is grounded in a deployed bot serving rural Indian populations and validated through a three-stakeholder review pipeline (doctors, ASHA workers, healthcare-nonprofit staff). Released for the NeurIPS 2026 Evaluations and Datasets track.",
        "conformsTo": "http://mlcommons.org/croissant/1.0",
        "citeAs": "To appear, NeurIPS 2026 Evaluations and Datasets track. Author and institution details are withheld during review.",
        "creator": [
            {"@type": "Organization", "name": "Authors withheld for review"},
        ],
        "keywords": [
            "maternal health", "reproductive health", "multilingual evaluation",
            "Hindi", "Marathi", "rural India", "non-WEIRD", "LLM benchmark",
            "clinical rubric", "ASHA worker",
        ],
        "license": "https://creativecommons.org/licenses/by/4.0/",
        "url": repo_url,
        "version": "1.1",
        "datePublished": "2026-05-06",
        "isLiveDataset": False,

        "rai:dataCollection": "Generated by Aya Expanse from a curated knowledge corpus of public maternal-health guidelines (WHO, India National ANC Guideline, ANM Training Manual, NHM protocols), validated by MedGemma, and routed through a three-stakeholder human review pipeline. Translations to Hindi and Marathi performed by professional native-speaker translators and reviewed by ASHA workers for patient-voice fidelity.",
        "rai:dataCollectionType": ["Generated by an automated process", "Curated by humans"],
        "rai:dataCollectionMissingData": "Coverage is complete in expert and non-expert tracks across all three languages, except a small number of pairs where translation review flagged ambiguity; these rows are explicitly marked.",
        "rai:dataCollectionRawData": "Raw 822-pair generation pool and per-stage drop logs are retained internally for audit but are not part of the public release.",
        "rai:dataCollectionTimeframe": "2025-10-31/2025-12-01",
        "rai:dataAnnotationProtocol": "Theme labels are produced by a DSPy few-shot classifier (10-15 expert-labelled seeds per theme); low-confidence predictions reviewed by clinicians. Reference answers in the expert arm are written or rewritten by practising doctors. Reference answers in the non-expert arm are doctor-reviewed but not doctor-authored.",
        "rai:dataAnnotationPlatform": "Purpose-built Q&A review platform; not a public crowd-work platform. Doctors, ASHA workers, and nonprofit staff are project collaborators, not crowd workers.",
        "rai:dataAnnotationAnalysis": "Three-stakeholder review (clinical, sociolinguistic, cultural-infrastructural). Each review action logged with reviewer role, timestamp, edit diff for internal audit; reviewer identity is NOT included in the released artefact.",
        "rai:dataAnnotationDemographics": "Doctors are practising physicians in Indian district and tertiary-care settings. ASHA workers are National Health Mission frontline workers in rural districts. Nonprofit staff have local cultural and infrastructural expertise relevant to maternal-health deployment.",
        "rai:dataAnnotationTools": "Internal review platform; the codebase is part of the deploying nonprofit's infrastructure and is not part of this release.",
        "rai:dataPreprocessingProtocol": [
            "Embedding-similarity deduplication against the source corpus",
            "Fuzzy string-match deduplication within the pool",
            "Per-stage drop based on stakeholder review",
        ],
        "rai:dataReleaseMaintenancePlan": "Versioned releases tagged on the Hugging Face dataset repo; errata in the GitHub issue tracker. Approximate cadence: annual. Two follow-up releases planned: Indic-first native-authored extension and a human-rater scoring subset.",
        "rai:dataUseCases": "Cross-lingual maternal-health LLM evaluation; calibration of LLM-as-judge protocols on multilingual medical content; audit of clinical-rubric performance across model families.",
        "rai:dataLimitations": "Hindi and Marathi are professional translations of an English seed set; the benchmark does not yet include native-authored Indic queries. Reviewer pool is volunteer rather than paid, and is therefore smaller than a crowd-recruited pool would be. Single-turn only; no multi-turn clarification dynamics.",
        "rai:dataSocialImpact": "The intended use is to support safer deployment of AI maternal-health systems in multilingual rural settings, where the population that bears the safety risk is currently under-represented in evaluation data. Misuse risk: a deploying party may read the headline MQS without reading the axis decomposition and conclude a model is ready when it is not. We mitigate by releasing per-axis, per-language, per-theme breakdowns alongside the headline.",
        "rai:dataBiases": "By design, the question population reflects rural and semi-urban Indian women aged 23-33 (the deployed-bot user base). The benchmark is not representative of urban Indian, non-Indian South Asian, or non-South-Asian maternal health contexts. The reference answers reflect doctor and ASHA-worker judgments at the time of review and may not capture future updates to clinical guidelines.",
        "rai:personalSensitiveInformation": "None. The dataset contains no personally identifying information about any user, doctor, ASHA worker, or other reviewer. Reviewer identity is logged internally during the review pipeline but is not part of the released artefact.",

        "distribution": file_objects,
        "recordSet": record_sets,
    }

    out = HF_DIR / "croissant.json"
    out.write_text(json.dumps(croissant, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


README_TMPL = """---
license: cc-by-4.0
language:
- en
- hi
- mr
language_creators:
- expert-generated
- found
multilinguality:
- multilingual
pretty_name: Sakhi — A Community-Validated Multilingual Maternal-Health Benchmark
size_categories:
- n<1K
source_datasets:
- original
tags:
- maternal-health
- reproductive-health
- women-health
- india
- indic
- low-resource
- evaluation
- benchmark
- llm-as-judge
- clinical
task_categories:
- question-answering
- text-generation
task_ids:
- closed-domain-qa
- open-book-qa
configs:
- config_name: expert
  data_files:
  - split: test
    path: data/expert.parquet
- config_name: non_expert
  data_files:
  - split: test
    path: data/non_expert.parquet
- config_name: doctor_ratings
  data_files:
  - split: test
    path: data/doctor_ratings.parquet
- config_name: results
  data_files:
  - split: test
    path: data/results.parquet
---

# Sakhi: A Community-Validated Multilingual Maternal-Health Benchmark

Sakhi is a benchmark for evaluating large language models on maternal and reproductive-health questions in three languages spoken in low-resource settings: English, Hindi, and Marathi. It was built around a deployed WhatsApp-based maternal-health chatbot reaching rural mothers in Hindi- and Marathi-speaking districts of India, with a three-channel review pipeline: practising Indian doctors, Accredited Social Health Activist (ASHA) workers, and healthcare-nonprofit frontline staff.

## What is in the dataset

- **expert** — 149 questions with doctor-edited reference answers, parallel across English, Hindi, and Marathi. The expert track is the harder, doctor-validated side of the benchmark.
- **non_expert** — 231 questions with community-sourced reference answers reviewed by healthcare-nonprofit staff and ASHA workers, parallel across English, Hindi, and Marathi. The non-expert track approximates the deployment-facing question distribution.
- **doctor_ratings** — 2,103 binary rubric verdicts (169 question-level verdicts × ~13 rubric criteria each) supplied by 11 practising Indian doctors (2 OB/GYN, 9 General Practitioners) on 148 of the 149 expert-track questions; 21 of those questions received a second independent verdict, supporting an inter-rater agreement floor.
- **results** — per-model MQS leaderboard from the companion paper. One row per (model, dataset arm, language) cell, judged by GPT-4o-mini.

Total release: 380 question-answer pairs across three languages (3,420 query-language pairs in total) plus the doctor-rating record. Each question is annotated with one of ten maternal-health themes (Antenatal Care, Nutrition & Diet, Mental Well-being, Clinical Procedures, Medication Safety, Reproductive Health, Health Systems Access, Infection Prevention, Symptom Interpretation, Risk & Complication Management) and rated under a 14-criterion clinical rubric covering Accuracy, Completeness, Context Awareness, Communication, and Terminology Accessibility.

## How to load

```python
from datasets import load_dataset

# Expert arm: 149 doctor-edited Q-A pairs
expert = load_dataset("__REPO__", "expert", split="test")

# Non-expert arm: 231 community-sourced Q-A pairs
non_expert = load_dataset("__REPO__", "non_expert", split="test")

# Doctor calibration: 2,103 binary rubric verdicts
doctor_ratings = load_dataset("__REPO__", "doctor_ratings", split="test")

# Per-model MQS leaderboard from the companion paper
results = load_dataset("__REPO__", "results", split="test")
```

## Schema

### expert / non_expert

| field | type | description |
|---|---|---|
| `q_id` | string | unique question identifier |
| `theme` | string | one of ten maternal-health themes |
| `domain` | string | sub-domain tag |
| `question_en` | string | question in English |
| `question_hi` | string | question in Hindi (Devanagari) |
| `question_mr` | string | question in Marathi (Devanagari) |
| `answer_en` | string | reference answer in English |
| `answer_hi` | string | reference answer in Hindi |
| `answer_mr` | string | reference answer in Marathi |
| `sources` / `references` | string | clinical sources cited by the reviewer (where applicable) |

### doctor_ratings

| field | type | description |
|---|---|---|
| `doctor_id` | string | anonymised reviewer pseudonym (R1 to R11) |
| `doctor_role` | string | OB/GYN or General Practitioner |
| `doctor_experience` | string | self-reported clinical experience bracket |
| `doctor_ai_exposure` | string | self-reported prior LLM exposure |
| `question_id` | string | links back to the expert track |
| `question_text` | string | the question shown to the reviewer |
| `ai_response` | string | the AI draft shown to the reviewer |
| `rubric_text` | string | the rubric criterion text |
| `axis` | string | rubric axis (Accuracy / Completeness / Context Awareness / Communication / Terminology Accessibility) |
| `verdict` | string | `pass` or `fail` (binary) |
| `theme` | string | one of ten maternal-health themes |
| `domain` | string | sub-domain tag |
| `references` | string | clinical sources |

### results

| field | type | description |
|---|---|---|
| `model` | string | generation model alias (e.g. `gpt_5_mini`, `medgemma_27b`) |
| `dataset` | string | `expert` or `non_expert` |
| `lang` | string | `en`, `hi`, or `mr` |
| `n` | int | number of valid model responses scored |
| `mqs_mean` | float | mean Medical Quality Score (axis-weighted) |
| `mqs_std` | float | standard deviation across scored responses |
| `judge` | string | LLM judge identifier (`gpt-4o-mini`) |

## Intended use

This dataset is intended for evaluating large language models on maternal and reproductive-health questions in cross-cultural, multilingual, low-resource settings. It is suitable for:

- Benchmarking LLMs on a doctor-validated rubric across English / Hindi / Marathi.
- Studying judge-vs-doctor calibration of LLM-as-judge scoring in clinical QA.
- Cross-lingual analyses of model behaviour on Indic languages relative to English.

It is **not** suitable as clinical advice and must not be deployed in any clinical workflow without further validation by qualified medical professionals practising in the deployment context.

## Out-of-scope use

- Direct deployment of any LLM that scores well on this benchmark in a patient-facing setting without local clinical validation, regulatory clearance, and continuous human-in-the-loop oversight.
- Substituting model-generated answers for advice from qualified healthcare providers.
- Generalising results to maternal-health populations outside South Asia or to languages other than English, Hindi, and Marathi.

## Provenance

- **Question pool**: drafted by an Aya Expanse generator and validated by MedGemma over a public maternal-health knowledge corpus, then routed through three review channels.
- **Doctor edits**: 11 practising Indian doctors (OB/GYN and General Practitioners) edited the reference answers in the expert track and rated AI drafts on a 14-criterion clinical rubric.
- **ASHA workers and nonprofit staff**: provided sociolinguistic and cultural review on the reference answers and translations.
- **Translation**: professional human translation from English into Hindi and Marathi, then ASHA-reviewed for patient-voice fidelity.

A full pipeline diagram and per-stage drop counts are in the companion paper.

## Ethics

- The dataset contains no personally identifying information about any individual user, doctor, ASHA worker, or other reviewer. Reviewer identity is logged internally during the review pipeline but is not part of this released artefact.
- The medical content is an evaluation reference, not deployable clinical guidance.
- Reviewers consented to having their rubric verdicts published in anonymised form.

## Citation

```bibtex
@inproceedings{sakhi2026,
  title     = {Large Language Models Still Fail Sensitive Maternal-Health Questions in Non-English Languages},
  author    = {Anonymous},
  booktitle = {Submitted to the Conference on Neural Information Processing Systems (NeurIPS 2026), Track on Evaluations and Datasets},
  year      = {2026},
  note      = {Author and institution details withheld during review.}
}
```

## License

Released under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

## Croissant metadata

A Croissant 1.0 metadata file (`croissant.json`) is included alongside the data files. Loaders that understand Croissant can pick up file URLs, schemas, MD5 checksums, and Responsible-AI metadata directly.
"""


def write_readme(repo_id: str) -> Path:
    out = HF_DIR / "README.md"
    out.write_text(README_TMPL.replace("__REPO__", repo_id), encoding="utf-8")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default="SimPPL/sakhi",
                    help="HF dataset repo id used in load_dataset() examples (default: SimPPL/sakhi)")
    args = ap.parse_args()

    parquets = build_parquets()
    results_path = build_results_parquet()
    rubrics_path = copy_rubrics()
    croissant_path = build_croissant(parquets, results_path, rubrics_path, args.repo_id)
    readme_path = write_readme(args.repo_id)

    print("HF release built under", HF_DIR)
    for k, p in parquets.items():
        print(f"  data/{p.name:>26}  rows={len(pd.read_parquet(p)):>6}  md5={md5(p)}")
    print(f"  data/{results_path.name:>26}  rows={len(pd.read_parquet(results_path)):>6}  md5={md5(results_path)}")
    print(f"  {rubrics_path.name:>31}                md5={md5(rubrics_path)}")
    print(f"  {croissant_path.name:>31}")
    print(f"  {readme_path.name:>31}")


if __name__ == "__main__":
    main()
