"""Normalize CSV columns so generation and scoring agree on names."""

from __future__ import annotations

import pandas as pd


def ensure_answer_reference_column(df: pd.DataFrame) -> pd.DataFrame:
    """Rubric judge and English scoring expect an `answer` reference column."""
    out = df.copy()
    if "ideal_answer" in out.columns:
        if "answer" not in out.columns:
            out["answer"] = out["ideal_answer"]
        else:
            out["answer"] = out["answer"].fillna(out["ideal_answer"])
    return out


def ensure_multilingual_question_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """Align non-expert column names with multilingual generation (`question_hi` / `question_mr`)."""
    out = df.copy()
    if "question_hi" not in out.columns and "questions_hindi" in out.columns:
        out["question_hi"] = out["questions_hindi"]
    if "question_mr" not in out.columns and "questions_marathi" in out.columns:
        out["question_mr"] = out["questions_marathi"]
    return out


def ensure_multilingual_reference_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """Map expert ideal translations to names expected by ling_semantic_scorer.get_multilingual_ref_cols."""
    out = df.copy()
    if "answer_hi" not in out.columns and "ideal_answer_hi" in out.columns:
        out["answer_hi"] = out["ideal_answer_hi"]
    if "answer_mr" not in out.columns and "ideal_answer_mr" in out.columns:
        out["answer_mr"] = out["ideal_answer_mr"]
    if "answer_hindi" not in out.columns and "answer_hi" in out.columns:
        out["answer_hindi"] = out["answer_hi"]
    if "answer_marathi" not in out.columns and "answer_mr" in out.columns:
        out["answer_marathi"] = out["answer_mr"]
    return out


def prepare_for_english_generation(df: pd.DataFrame) -> pd.DataFrame:
    return ensure_answer_reference_column(df)


def prepare_for_multilingual_generation(df: pd.DataFrame) -> pd.DataFrame:
    out = ensure_multilingual_question_aliases(df)
    out = ensure_multilingual_reference_aliases(out)
    return out
