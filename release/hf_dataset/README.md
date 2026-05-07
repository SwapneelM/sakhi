---
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
expert = load_dataset("SimPPL/sakhi", "expert", split="test")

# Non-expert arm: 231 community-sourced Q-A pairs
non_expert = load_dataset("SimPPL/sakhi", "non_expert", split="test")

# Doctor calibration: 2,103 binary rubric verdicts
doctor_ratings = load_dataset("SimPPL/sakhi", "doctor_ratings", split="test")

# Per-model MQS leaderboard from the companion paper
results = load_dataset("SimPPL/sakhi", "results", split="test")
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
