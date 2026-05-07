"""Repository-root-relative paths for generation and downstream scoring."""

from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


DATA2 = repo_root() / "data2"
OUTPUTS = repo_root() / "outputs"
RUBRIC_SCORES_DIR = OUTPUTS / "rubric_scores"
LINGUISTIC_SEMANTIC_DIR = OUTPUTS / "linguistic_semantic"

# --- data2 inputs (source CSVs) ---
SAKHI_NON_EXPERT_RAW_230 = DATA2 / "sakhi_non_expert_raw_230.csv"
SAKHI_EXPERT_RAW_150 = DATA2 / "sakhi_expert_raw_150.csv"
SAKHI_HINDI_MARATHI_EXPERT_150 = DATA2 / "sakhi_hindi_marathi_expert_150.csv"

# --- generation outputs (under outputs/) ---
LLM_ENGLISH_NON_EXPERT_230 = OUTPUTS / "llm_test_data_english_non_expert_230.csv"
LLM_ENGLISH_EXPERT_150 = OUTPUTS / "llm_test_data_english_expert_150.csv"
LLM_MULTILINGUAL_NON_EXPERT_230 = OUTPUTS / "llm_test_data_multilingual_non_expert_230.csv"
LLM_MULTILINGUAL_EXPERT_150 = OUTPUTS / "llm_test_data_multilingual_expert_150.csv"

# --- rubric judge outputs (filenames must match ling_semantic_scorer.py expectations) ---
RUBRIC_NON_EXPERT_ENGLISH = RUBRIC_SCORES_DIR / "rubric_scores_non_expert_english_gpt.csv"
RUBRIC_EXPERT_ENGLISH = RUBRIC_SCORES_DIR / "rubric_scores_expert_english_gpt.csv"
RUBRIC_NON_EXPERT_MULTILINGUAL = RUBRIC_SCORES_DIR / "rubric_scores_non_expert_multilingual_gpt.csv"
RUBRIC_EXPERT_MULTILINGUAL = RUBRIC_SCORES_DIR / "rubric_scores_expert_multilingual_gpt.csv"
