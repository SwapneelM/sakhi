"""Assemble the Sakhi benchmark release artifacts from the expert + non-expert tracks.

Outputs (in release/):
  - sakhi_benchmark_expert.csv    (149 rows; parallel EN/HI/MR with ideal answers)
  - sakhi_benchmark_non_expert.csv (231 rows; community-sourced parallel pairs)
  - sakhi_rubrics.json            (per-theme rubric: 15 criteria × 5 axes × weights)
  - sakhi_croissant.json          (Croissant 1.0 metadata pointing to the CSVs)
  - README.md                     (dataset card: provenance, licence, fields)

No model responses are included — the benchmark is separated from benchmark artifacts.
The companion HuggingFace parquet release is built by build_hf_release.py.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import pandas as pd

from scoring.scoring_rubric import RUBRIC_MAP, AXIS_MAP

REPO = Path(__file__).resolve().parent.parent.parent
RELEASE = REPO / "release"
RELEASE.mkdir(exist_ok=True)

AXIS_WEIGHTS = {
    "Accuracy": 0.30,
    "Completeness": 0.25,
    "Context Awareness": 0.20,
    "Communication": 0.15,
    "Terminology Accessibility": 0.10,
}


def build_expert_csv():
    df = pd.read_csv(REPO / "data2" / "sakhi_expert_raw_150.csv")
    keep = ["q_id", "theme", "domain", "question", "question_hi", "question_mr",
            "ideal_answer", "ideal_answer_hi", "ideal_answer_mr", "sources"]
    keep = [c for c in keep if c in df.columns]
    out = df[keep].copy()
    out = out.rename(columns={
        "question": "question_en",
        "ideal_answer": "answer_en",
        "ideal_answer_hi": "answer_hi",
        "ideal_answer_mr": "answer_mr",
    })
    # drop rows without a question
    out = out[out["question_en"].notna() & out["question_en"].str.strip().ne("")]
    path = RELEASE / "sakhi_benchmark_expert.csv"
    out.to_csv(path, index=False)
    return path, len(out)


def build_non_expert_csv():
    df = pd.read_csv(REPO / "data2" / "sakhi_non_expert_raw_230.csv")
    keep = ["theme", "domain", "question", "questions_hindi", "questions_marathi",
            "answer", "answer_hindi", "answer_marathi", "references"]
    keep = [c for c in keep if c in df.columns]
    out = df[keep].copy()
    out = out.rename(columns={
        "question": "question_en",
        "questions_hindi": "question_hi",
        "questions_marathi": "question_mr",
        "answer": "answer_en",
        "answer_hindi": "answer_hi",
        "answer_marathi": "answer_mr",
    })
    out.insert(0, "q_id", [f"ne-{i:04d}" for i in range(len(out))])
    out = out[out["question_en"].notna() & out["question_en"].str.strip().ne("")]
    path = RELEASE / "sakhi_benchmark_non_expert.csv"
    out.to_csv(path, index=False)
    return path, len(out)


def build_rubric_json():
    rubrics = {}
    for theme, rubric_str in RUBRIC_MAP.items():
        criteria = json.loads(rubric_str)
        axes_str = AXIS_MAP.get(theme, "{}")
        axes = json.loads(axes_str) if axes_str else {}
        rubrics[theme] = {
            "criteria": criteria,
            "axes": axes,
            "axis_weights": AXIS_WEIGHTS,
        }
    path = RELEASE / "sakhi_rubrics.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(rubrics, f, ensure_ascii=False, indent=2)
    return path, len(rubrics)


README = """# Sakhi Benchmark — Dataset Release

A parallel multilingual (English / Hindi / Marathi) benchmark for evaluating large language
models on maternal and reproductive health questions as asked by women in rural and
semi-urban India.

The HuggingFace parquet release lives at `simppl/sakhi`; this directory holds the
CSV-flavoured variant for GitHub readers.

## Files

| File | Rows | Description |
|------|------|-------------|
| `sakhi_benchmark_expert.csv` | 149 | Expert-validated Q/A pairs; each English entry has a doctor-curated `answer_en` and professionally-translated Hindi/Marathi equivalents. |
| `sakhi_benchmark_non_expert.csv` | 231 | Community-sourced Q/A pairs covering the same ten maternal-health themes; answers drawn from synthetic generation with medical guideline grounding (see paper for provenance). |
| `sakhi_rubrics.json` | 10 themes | Thematic rubric: for each theme, 15 binary criteria partitioned into 5 weighted axes (Accuracy 0.30, Completeness 0.25, Context Awareness 0.20, Communication 0.15, Terminology Accessibility 0.10). |
| `sakhi_doctor_ratings.csv` | 2,103 | Doctor calibration verdicts: 11 practising Indian doctors (R1-R11; 2 OB/GYN, 9 GP) on 148 expert-arm questions across all rubric criteria. |
| `sakhi_croissant.json` | — | Croissant 1.0 metadata (file URLs, schemas, MD5 checksums, RAI fields). |
| `results/` | — | Aggregated MQS by model x dataset x language x theme x axis under each LLM judge. |

The two benchmark CSVs share this schema:

```
q_id, theme, domain, question_en, question_hi, question_mr,
answer_en, answer_hi, answer_mr[, references|sources]
```

## Themes

Antenatal & Maternal Health Care, Nutrition Diet & Supplementation, Mental Emotional & Social
Well-being, Clinical Procedures & Guidelines, Medication & Vaccination Safety, Reproductive &
Sexual Health (Beyond Pregnancy), Health Systems Access & Provider Support, Infection Prevention
& Hygiene Practices, Symptom Interpretation & Danger Sign Recognition, Risk & Complication
Management.

## Evaluation protocol

The Medical Quality Score (MQS) is the sum of axis-level pass ratios weighted by the axis weights
above. A response's rubric score is returned by an LLM judge using the zero-shot prompt specified
in the accompanying paper (Appendix A). To ensure comparability, judges should use `temperature=0`
and the verbatim rubric criteria listed in `sakhi_rubrics.json`.

## Reproducing the release

The artefacts in this directory are regenerated by:

```
python -m scoring.pipeline.build_release         # CSV variant (this directory)
python -m scoring.pipeline.build_hf_release      # parquet variant (release/hf_dataset/)
```

Both scripts read from the canonical raw CSVs in `data2/` and from `release/results/`.

## Licence

See the repository `LICENSE` file. The benchmark is released under CC BY 4.0 for research and
non-commercial evaluation of AI systems; it is not intended for direct clinical use.
"""


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def build_csv_croissant(p_ex: Path, p_ne: Path, p_ru: Path, p_dr: Path | None) -> Path:
    """Generate sakhi_croissant.json describing the CSV-flavoured release."""
    from scoring.pipeline.build_hf_release import build_croissant as _build  # for shared shape  # noqa: E501

    # We don't reuse build_hf_release.build_croissant directly because it
    # hard-codes parquet ids; instead we duplicate the small bits we need.
    file_objects = [
        {
            "@type": "cr:FileObject", "@id": "expert-csv",
            "name": p_ex.name,
            "description": "Expert arm: 149 doctor-edited Q&A pairs in EN/HI/MR.",
            "encodingFormat": "text/csv", "contentUrl": p_ex.name,
            "md5": md5(p_ex),
        },
        {
            "@type": "cr:FileObject", "@id": "non-expert-csv",
            "name": p_ne.name,
            "description": "Non-expert arm: 231 community-sourced Q&A pairs with doctor-reviewed reference answers, in EN/HI/MR.",
            "encodingFormat": "text/csv", "contentUrl": p_ne.name,
            "md5": md5(p_ne),
        },
        {
            "@type": "cr:FileObject", "@id": "rubrics-json",
            "name": p_ru.name,
            "description": "10 themes x 5 axes x 3 binary criteria rubric used to score model responses.",
            "encodingFormat": "application/json", "contentUrl": p_ru.name,
            "md5": md5(p_ru),
        },
    ]
    if p_dr is not None and p_dr.exists():
        file_objects.append({
            "@type": "cr:FileObject", "@id": "doctor-ratings-csv",
            "name": p_dr.name,
            "description": "2,103 binary rubric verdicts on 148 expert-arm questions, supplied by 11 practising Indian doctors (R1-R11; 2 OB/GYN, 9 General Practitioners).",
            "encodingFormat": "text/csv", "contentUrl": p_dr.name,
            "md5": md5(p_dr),
        })

    def fld(rid, dtype, fid, col):
        return {
            "@type": "cr:Field", "@id": rid, "dataType": dtype,
            "source": {"fileObject": {"@id": fid}, "extract": {"column": col}},
        }

    benchmark_cols = ["q_id", "theme", "domain", "question_en", "question_hi",
                      "question_mr", "answer_en", "answer_hi", "answer_mr"]

    record_sets = [
        {
            "@type": "cr:RecordSet", "@id": "expert-records",
            "name": "Expert arm records",
            "description": "Doctor-edited reference answers parallel across English, Hindi, and Marathi.",
            "field": [fld(f"expert/{c}", "sc:Text", "expert-csv", c) for c in benchmark_cols]
                     + [fld("expert/sources", "sc:Text", "expert-csv", "sources")],
        },
        {
            "@type": "cr:RecordSet", "@id": "non-expert-records",
            "name": "Non-expert arm records",
            "description": "Community-sourced reference answers reviewed by ASHA workers and nonprofit staff.",
            "field": [fld(f"non_expert/{c}", "sc:Text", "non-expert-csv", c) for c in benchmark_cols]
                     + [fld("non_expert/references", "sc:Text", "non-expert-csv", "references")],
        },
    ]
    if p_dr is not None and p_dr.exists():
        record_sets.append({
            "@type": "cr:RecordSet", "@id": "doctor-ratings-records",
            "name": "Doctor calibration verdicts",
            "description": "One row per (reviewer, question, rubric criterion). Reviewer pseudonyms run R1 to R11.",
            "field": [
                fld("doc/doctor_id", "sc:Text", "doctor-ratings-csv", "doctor_id"),
                fld("doc/doctor_role", "sc:Text", "doctor-ratings-csv", "doctor_role"),
                fld("doc/doctor_experience", "sc:Text", "doctor-ratings-csv", "doctor_experience"),
                fld("doc/doctor_ai_exposure", "sc:Text", "doctor-ratings-csv", "doctor_ai_exposure"),
                fld("doc/question_id", "sc:Text", "doctor-ratings-csv", "question_id"),
                fld("doc/question_text", "sc:Text", "doctor-ratings-csv", "question_text"),
                fld("doc/ai_response", "sc:Text", "doctor-ratings-csv", "ai_response"),
                fld("doc/rubric_text", "sc:Text", "doctor-ratings-csv", "rubric_text"),
                fld("doc/axis", "sc:Text", "doctor-ratings-csv", "axis"),
                fld("doc/verdict", "sc:Text", "doctor-ratings-csv", "verdict"),
                fld("doc/theme", "sc:Text", "doctor-ratings-csv", "theme"),
                fld("doc/domain", "sc:Text", "doctor-ratings-csv", "domain"),
                fld("doc/references", "sc:Text", "doctor-ratings-csv", "references"),
            ],
        })

    # Pull the same RAI metadata block from the HF Croissant builder.
    from scoring.pipeline.build_hf_release import build_croissant as _hf_build_croissant  # noqa
    # Easiest: read the HF Croissant just produced and copy its top-level RAI fields.
    hf_path = RELEASE / "hf_dataset" / "croissant.json"
    if hf_path.exists():
        hf = json.loads(hf_path.read_text())
        rai = {k: v for k, v in hf.items() if k.startswith("rai:")}
        top = {k: v for k, v in hf.items()
               if k in ("@context", "@type", "name", "alternateName", "description",
                        "conformsTo", "citeAs", "creator", "keywords", "license",
                        "version", "datePublished", "isLiveDataset")}
    else:
        # Fallback: minimal context (HF builder must run first for full RAI metadata).
        top, rai = {}, {}
        print("WARNING: build_hf_release.py has not been run; CSV Croissant will be missing RAI metadata.")

    croissant = {
        **top,
        "url": "https://github.com/SimPPL/MedicaLLM-Eval",
        **rai,
        "distribution": file_objects,
        "recordSet": record_sets,
    }
    out = RELEASE / "sakhi_croissant.json"
    out.write_text(json.dumps(croissant, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def main():
    p_ex, n_ex = build_expert_csv()
    p_ne, n_ne = build_non_expert_csv()
    p_ru, n_ru = build_rubric_json()
    readme_path = RELEASE / "README.md"
    readme_path.write_text(README, encoding="utf-8")

    p_dr = RELEASE / "sakhi_doctor_ratings.csv"
    p_cr = build_csv_croissant(p_ex, p_ne, p_ru, p_dr if p_dr.exists() else None)

    print(f"Wrote {p_ex} ({n_ex} rows)")
    print(f"Wrote {p_ne} ({n_ne} rows)")
    print(f"Wrote {p_ru} ({n_ru} themes)")
    print(f"Wrote {readme_path}")
    print(f"Wrote {p_cr}")


if __name__ == "__main__":
    main()
