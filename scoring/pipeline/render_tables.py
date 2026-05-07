"""Render aggregated results CSVs into LaTeX table fragments that the paper \\input{}s.

Writes to overleaf/paper/latex/tables/:
  tab_mqs_overall.tex          model x (expert, non_expert) x (en, hi, mr) MQS
  tab_mqs_crosslingual.tex     pivoted model x lang (dataset as rows) for cross-lingual gaps
  tab_mqs_themes.tex           model x theme (within each dataset) MQS
  tab_cost_mqs.tex             per-model cost vs mean MQS (EN non_expert, primary judge)
  tab_axis_failures.tex        model x axis pass rate (within-dataset EN)
  tab_validity.tex             generation reliability per model x lang x run
  tab_judge_agreement.tex      pairwise MQS correlations + Fleiss kappa on calibration subset

Convention (NeurIPS Datasets & Benchmarks style):
  * Best value per column shown in \\textbf{bold}; second best shown \\underline{underlined}.
  * A \\midrule separates the evaluated panel from "Gemma 3 27B (baseline)", which is
    a comparator model (matched base for MedGemma) rather than a panel entry.
  * Coverage gaps (missing cells) are shown as "--" and tie-breaking for bold/underline
    ignores missing cells.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
from statistics import mean, pstdev

import pandas as pd

REPO = Path(__file__).resolve().parent.parent.parent
RESULTS = REPO / "release" / "results"
TABLES = REPO / "overleaf" / "paper" / "latex" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

MODEL_DISPLAY = {
    "gpt_5_mini":            "GPT-5 Mini",
    "gpt_4o_mini":           "GPT-4o Mini",
    "cohere_command_a":      "Command A",
    "gemini_3_pro":          "Gemini 3 Pro",
    "gemini_3_flash":        "Gemini 3 Flash",
    "gemini_3_1_flash_lite": "Gemini 3.1 Flash-Lite",
    "claude_opus_4_7":       "Claude Opus 4.7",
    "claude_haiku_4_5":      "Claude Haiku 4.5",
    "llama_3_3_70b":         "Llama 3.3 70B",
    "llama_4_maverick":      "Llama 4 Maverick",
    "aya_expanse":           "Aya Expanse 32B",
    "medgemma_27b":          "MedGemma 27B",
    "medgemma_4b":           "MedGemma 4B",
    "gemma_3_27b":           "Gemma 3 27B (baseline)",
}
MODEL_ORDER = list(MODEL_DISPLAY.keys())
BASELINE_MODEL = "gemma_3_27b"  # visually separated from the evaluated panel

LANG_NAMES = {"en": "EN", "hi": "HI", "mr": "MR"}
DATASET_NAMES = {"expert": "Expert (149)", "non_expert": "Non-expert (231)"}

# Short labels for theme table column headers. The full names contain "&", which
# collides with tabular column separators inside \rotatebox{}; using a clean
# single-to-two-word label avoids the LaTeX compile error and keeps the header
# readable when rotated 60 degrees.
THEME_SHORT = {
    "Antenatal & Maternal Health Care":                  "Antenatal",
    "Clinical Procedures & Guidelines":                  "Clinical",
    "Health Systems, Access & Provider Support":         "Health Sys.",
    "Infection Prevention & Hygiene Practices":          "Infection",
    "Medication & Vaccination Safety":                   "Medication",
    "Mental, Emotional & Social Well-being":             "Mental",
    "Nutrition, Diet & Supplementation":                 "Nutrition",
    "Reproductive & Sexual Health (Beyond Pregnancy)":   "Reproductive",
    "Risk & Complication Management":                    "Risk Mgmt.",
    "Symptom Interpretation & Danger Sign Recognition":  "Symptoms",
}


def _pick_mqs_df(judge_alias: str = "gpt_4o_mini") -> pd.DataFrame | None:
    p = RESULTS / f"mqs_per_model__{judge_alias}.csv"
    return pd.read_csv(p) if p.exists() else None


def _fmt_val(v, precision: int = 3) -> str:
    if v is None or pd.isna(v):
        return "--"
    try:
        return f"{float(v):.{precision}f}"
    except Exception:
        return "--"


def _rank_bold_underline(values_by_model: dict[str, float], higher_is_better: bool = True) -> tuple[set[str], set[str]]:
    """Return (best_models, second_models). Ties all get the same tier."""
    valid: list[tuple[str, float]] = []
    for m, v in values_by_model.items():
        if v is None:
            continue
        try:
            if pd.isna(v):
                continue
        except (TypeError, ValueError):
            pass
        try:
            valid.append((m, float(v)))
        except (TypeError, ValueError):
            continue
    if not valid:
        return set(), set()
    valid.sort(key=lambda x: x[1], reverse=higher_is_better)
    best_val = valid[0][1]
    best_set = {m for m, v in valid if v == best_val}
    remaining = [(m, v) for m, v in valid if v != best_val]
    if not remaining:
        return best_set, set()
    second_val = remaining[0][1]
    second_set = {m for m, v in remaining if v == second_val}
    return best_set, second_set


def _format_cell(val: float | None, best: set[str], second: set[str], model: str, precision: int = 3, signed: bool = False) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "--"
    s = f"{val:+.{precision}f}" if signed else f"{val:.{precision}f}"
    if model in best:
        return r"\textbf{" + s + "}"
    if model in second:
        return r"\underline{" + s + "}"
    return s


def _emit_rows(out: list[str], ordered_models: list[str], row_fn) -> None:
    """Emit rows for the evaluated panel, then a midrule, then the baseline row(s)."""
    panel = [m for m in ordered_models if m != BASELINE_MODEL]
    for m in panel:
        line = row_fn(m)
        if line is not None:
            out.append(line)
    baseline_line = row_fn(BASELINE_MODEL) if BASELINE_MODEL in ordered_models else None
    if baseline_line is not None:
        out.append(r"\midrule")
        out.append(baseline_line)


def render_mqs_overall():
    df = _pick_mqs_df()
    path = TABLES / "tab_mqs_overall.tex"
    if df is None or df.empty:
        path.write_text("% no data yet\n")
        return
    # pivot into a dict[model][dataset][lang] = (mean, std)
    table: dict[str, dict[str, dict[str, tuple[float, float]]]] = {}
    for _, r in df.iterrows():
        table.setdefault(r["gen_model"], {}).setdefault(r["dataset"], {})[r["lang"]] = (r["mqs_mean"], r["mqs_std"])

    columns = [("expert", "en"), ("expert", "hi"), ("expert", "mr"),
               ("non_expert", "en"), ("non_expert", "hi"), ("non_expert", "mr")]
    best_per_col: dict[tuple, set[str]] = {}
    second_per_col: dict[tuple, set[str]] = {}
    for col in columns:
        vals_by_model = {m: (table[m].get(col[0], {}).get(col[1])[0] if table[m].get(col[0], {}).get(col[1]) is not None else None)
                         for m in table}
        best, second = _rank_bold_underline(vals_by_model, higher_is_better=True)
        best_per_col[col] = best
        second_per_col[col] = second

    out = [
        r"\begin{table*}[t]",
        r"\centering\small",
        r"\caption{Medical Quality Score (MQS) by model, dataset, and language under the GPT-4o-mini judge. Cells are the mean across three runs and all questions for which the model produced a valid response. \textbf{Bold} marks the best score per column; \underline{underline} marks the second best. A midrule separates the 13-model evaluated panel from the baseline Gemma 3 27B, which is reported only to anchor the medical-finetune comparison against MedGemma 27B.}",
        r"\label{tab:mqs_overall}",
        r"\begin{tabular}{l" + "c" * 6 + "}",
        r"\toprule",
        r"\textbf{Model} & \multicolumn{3}{c}{\textbf{Expert (149)}} & \multicolumn{3}{c}{\textbf{Non-expert (231)}} \\",
        r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}",
        r" & \textbf{EN} & \textbf{HI} & \textbf{MR} & \textbf{EN} & \textbf{HI} & \textbf{MR} \\",
        r"\midrule",
    ]

    def row_fn(m):
        if m not in table:
            return None
        cells = [MODEL_DISPLAY[m]]
        for col in columns:
            v = table[m].get(col[0], {}).get(col[1])
            val = v[0] if v is not None else None
            cells.append(_format_cell(val, best_per_col[col], second_per_col[col], m))
        return " & ".join(cells) + r" \\"

    _emit_rows(out, MODEL_ORDER, row_fn)
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    path.write_text("\n".join(out) + "\n")


def render_mqs_crosslingual():
    df = _pick_mqs_df()
    path = TABLES / "tab_mqs_crosslingual.tex"
    if df is None or df.empty:
        path.write_text("% no data yet\n")
        return
    by_model_lang: dict[str, dict[str, list[float]]] = {}
    for _, r in df.iterrows():
        by_model_lang.setdefault(r["gen_model"], {}).setdefault(r["lang"], []).append(r["mqs_mean"])

    rows_data: dict[str, dict[str, float]] = {}
    for m in MODEL_ORDER:
        if m not in by_model_lang:
            continue
        lang_mean = {lg: (mean(v) if v else None) for lg, v in by_model_lang[m].items()}
        en = lang_mean.get("en")
        hi = lang_mean.get("hi")
        mr = lang_mean.get("mr")
        dhi = (hi - en) if (en is not None and hi is not None) else None
        dmr = (mr - en) if (en is not None and mr is not None) else None
        rows_data[m] = {"en": en, "hi": hi, "mr": mr, "dhi": dhi, "dmr": dmr,
                         "abs_dhi": abs(dhi) if dhi is not None else None,
                         "abs_dmr": abs(dmr) if dmr is not None else None}

    # best = highest for EN/HI/MR; most positive for deltas (largest improvement)
    best_en, second_en = _rank_bold_underline({m: d["en"] for m, d in rows_data.items()}, higher_is_better=True)
    best_hi, second_hi = _rank_bold_underline({m: d["hi"] for m, d in rows_data.items()}, higher_is_better=True)
    best_mr, second_mr = _rank_bold_underline({m: d["mr"] for m, d in rows_data.items()}, higher_is_better=True)
    # For delta columns, largest positive = biggest Indic-over-EN advantage
    best_dhi, second_dhi = _rank_bold_underline({m: d["dhi"] for m, d in rows_data.items()}, higher_is_better=True)
    best_dmr, second_dmr = _rank_bold_underline({m: d["dmr"] for m, d in rows_data.items()}, higher_is_better=True)

    out = [
        r"\begin{table}[t]\centering\small",
        r"\caption{Cross-lingual MQS averaged over both dataset arms. $\Delta$HI and $\Delta$MR show the signed change from English; positive means the model scores higher in Hindi or Marathi than English under the GPT-4o-mini judge. \textbf{Bold} marks the best (highest) score per column; \underline{underline} marks the second best.}",
        r"\label{tab:mqs_crosslingual}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"\textbf{Model} & \textbf{EN} & \textbf{HI} & \textbf{MR} & \textbf{$\Delta$HI} & \textbf{$\Delta$MR} \\",
        r"\midrule",
    ]

    def row_fn(m):
        if m not in rows_data:
            return None
        d = rows_data[m]
        cells = [
            MODEL_DISPLAY[m],
            _format_cell(d["en"], best_en, second_en, m),
            _format_cell(d["hi"], best_hi, second_hi, m),
            _format_cell(d["mr"], best_mr, second_mr, m),
            _format_cell(d["dhi"], best_dhi, second_dhi, m, signed=True),
            _format_cell(d["dmr"], best_dmr, second_dmr, m, signed=True),
        ]
        return " & ".join(cells) + r" \\"

    _emit_rows(out, MODEL_ORDER, row_fn)
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(out) + "\n")


def render_mqs_themes():
    p = RESULTS / "mqs_per_theme__gpt_4o_mini.csv"
    path = TABLES / "tab_mqs_themes.tex"
    if not p.exists():
        path.write_text("% no data yet\n"); return
    df = pd.read_csv(p)
    if df.empty:
        path.write_text("% no data yet\n"); return
    en = df[df["lang"] == "en"].copy()
    agg = en.groupby(["gen_model", "theme"])["mqs_mean"].mean().unstack()
    themes = list(agg.columns)

    # Short header label for each theme; THEME_SHORT avoids "&" collisions
    # with tabular column separators inside \rotatebox{}.
    def short_label(t: str) -> str:
        short = THEME_SHORT.get(t, t.split("&")[0].split("(")[0].strip())
        return r"\rotatebox{60}{" + short + "}"

    best_per_theme: dict[str, set[str]] = {}
    second_per_theme: dict[str, set[str]] = {}
    for t in themes:
        vals = {m: (agg.loc[m, t] if m in agg.index else None) for m in MODEL_ORDER}
        best, second = _rank_bold_underline(vals, higher_is_better=True)
        best_per_theme[t] = best
        second_per_theme[t] = second

    out = [
        r"\begin{table*}[t]\centering\small",
        r"\caption{MQS by maternal-health theme (English, both dataset arms averaged, GPT-4o-mini judge). \textbf{Bold} marks the best score per theme column; \underline{underline} marks the second best. Theme labels are rotated; short names correspond to the ten themes described in Section~\ref{sec:methodology} (e.g., Antenatal = Antenatal \& Maternal Health Care).}",
        r"\label{tab:mqs_themes}",
        r"\begin{tabular}{l" + "c" * len(themes) + "}",
        r"\toprule",
        r"\textbf{Model} & " + " & ".join(short_label(t) for t in themes) + r" \\",
        r"\midrule",
    ]

    def row_fn(m):
        if m not in agg.index:
            return None
        cells = [MODEL_DISPLAY[m]]
        for t in themes:
            v = agg.loc[m, t] if m in agg.index else None
            cells.append(_format_cell(v, best_per_theme[t], second_per_theme[t], m))
        return " & ".join(cells) + r" \\"

    _emit_rows(out, MODEL_ORDER, row_fn)
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    path.write_text("\n".join(out) + "\n")


def render_cost_mqs():
    cost_p = RESULTS / "cost_per_model.csv"
    mqs = _pick_mqs_df()
    path = TABLES / "tab_cost_mqs.tex"
    if not cost_p.exists() or mqs is None or mqs.empty:
        path.write_text("% no data yet\n"); return
    cdf = pd.read_csv(cost_p)
    en_non = mqs[(mqs["lang"] == "en") & (mqs["dataset"] == "non_expert")]
    joined = cdf.merge(en_non[["gen_model", "mqs_mean"]], left_on="model", right_on="gen_model", how="left")
    joined["mqs_per_usd"] = joined["mqs_mean"] / joined["cost_per_response_usd"].replace(0, pd.NA)

    by_model = {r["model"]: r for _, r in joined.iterrows()}
    mqs_vals = {m: (by_model[m].get("mqs_mean") if m in by_model else None) for m in MODEL_ORDER}
    eff_vals = {m: (by_model[m].get("mqs_per_usd") if m in by_model else None) for m in MODEL_ORDER}
    # Cost: lower is better
    cost_vals = {m: (by_model[m].get("cost_per_response_usd") if m in by_model else None) for m in MODEL_ORDER}
    best_mqs, second_mqs = _rank_bold_underline(mqs_vals, higher_is_better=True)
    best_eff, second_eff = _rank_bold_underline(eff_vals, higher_is_better=True)
    # Exclude zero cost (Claude Opus 4.7 via subscription) from "lowest cost" ranking
    cost_vals_nonzero = {m: v for m, v in cost_vals.items() if v is not None and v > 0}
    best_cost, second_cost = _rank_bold_underline(cost_vals_nonzero, higher_is_better=False)

    out = [
        r"\begin{table}[t]\centering\small",
        r"\caption{Cost-performance frontier. Per-response generation cost (USD) paired with mean non-expert EN MQS under the GPT-4o-mini judge. MQS/USD is an efficiency indicator; higher is better. \textbf{Bold} marks the best and \underline{underline} marks the second best in each column. Claude Opus 4.7 is accessed through a local Claude Code subscription at zero marginal cost, so it is excluded from the cost and MQS/USD ranking.}",
        r"\label{tab:cost_mqs}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"\textbf{Model} & \textbf{\$/resp} & \textbf{MQS} & \textbf{MQS/\$} \\",
        r"\midrule",
    ]

    def row_fn(m):
        if m not in by_model:
            return None
        r = by_model[m]
        dollars = r.get("cost_per_response_usd")
        mq = r.get("mqs_mean")
        eff = r.get("mqs_per_usd")
        # Cost cell: precision 5, lower-is-better bold. Zero/None left unbolded.
        if dollars is None or pd.isna(dollars):
            cost_cell = "--"
        elif dollars == 0:
            cost_cell = "0.00000"
        else:
            s = f"{dollars:.5f}"
            if m in best_cost:
                cost_cell = r"\textbf{" + s + "}"
            elif m in second_cost:
                cost_cell = r"\underline{" + s + "}"
            else:
                cost_cell = s
        mqs_cell = _format_cell(mq, best_mqs, second_mqs, m)
        if eff is None or pd.isna(eff):
            eff_cell = "--"
        else:
            s = f"{eff:,.0f}"
            if m in best_eff:
                eff_cell = r"\textbf{" + s + "}"
            elif m in second_eff:
                eff_cell = r"\underline{" + s + "}"
            else:
                eff_cell = s
        return " & ".join([MODEL_DISPLAY[m], cost_cell, mqs_cell, eff_cell]) + r" \\"

    _emit_rows(out, MODEL_ORDER, row_fn)
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(out) + "\n")


def render_axis_failures():
    p = RESULTS / "axis_per_model__gpt_4o_mini.csv"
    path = TABLES / "tab_axis_failures.tex"
    if not p.exists():
        path.write_text("% no data yet\n"); return
    df = pd.read_csv(p)
    if df.empty:
        path.write_text("% no data yet\n"); return
    en_non = df[(df["lang"] == "en") & (df["dataset"] == "non_expert")]
    axes = ["Accuracy", "Completeness", "Context Awareness", "Communication", "Terminology Accessibility"]

    # Build per-axis best/second
    best_per_axis: dict[str, set[str]] = {}
    second_per_axis: dict[str, set[str]] = {}
    for ax in axes:
        vals = {}
        for m in MODEL_ORDER:
            sub = en_non[(en_non["gen_model"] == m) & (en_non["axis"] == ax)]["pass_rate"]
            vals[m] = float(sub.iloc[0]) if not sub.empty else None
        best, second = _rank_bold_underline(vals, higher_is_better=True)
        best_per_axis[ax] = best
        second_per_axis[ax] = second

    out = [
        r"\begin{table*}[t]\centering\small",
        r"\caption{Axis-level pass rates on the non-expert EN arm (GPT-4o-mini judge). Values are the mean fraction of binary rubric criteria satisfied within each axis. \textbf{Terminology Accessibility is the lowest axis for every model} (0.123--0.206), followed by Context Awareness (0.226--0.264), so the dominant failure mode is lexical (untranslated medical shorthand, abbreviations, drug names) rather than broadly cultural. \textbf{Bold} and \underline{underline} mark the best and second-best model per axis column.}",
        r"\label{tab:axis_failures}",
        r"\begin{tabular}{l" + "c" * len(axes) + "}",
        r"\toprule",
        r"\textbf{Model} & " + " & ".join(r"\textbf{" + a + "}" for a in axes) + r" \\",
        r"\midrule",
    ]

    def row_fn(m):
        sub = en_non[en_non["gen_model"] == m]
        if sub.empty:
            return None
        cells = [MODEL_DISPLAY[m]]
        for ax in axes:
            v = sub[sub["axis"] == ax]["pass_rate"]
            val = float(v.iloc[0]) if not v.empty else None
            cells.append(_format_cell(val, best_per_axis[ax], second_per_axis[ax], m))
        return " & ".join(cells) + r" \\"

    _emit_rows(out, MODEL_ORDER, row_fn)
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    path.write_text("\n".join(out) + "\n")


def render_validity():
    p = RESULTS / "valid_response_counts.csv"
    path = TABLES / "tab_validity.tex"
    if not p.exists():
        path.write_text("% no data yet\n"); return
    df = pd.read_csv(p)
    if df.empty:
        path.write_text("% no data yet\n"); return
    agg = df.groupby(["gen_model", "dataset", "lang"])[["ok", "empty", "err", "total"]].sum().reset_index()
    agg["ok_rate"] = agg["ok"] / agg["total"]
    pivot = agg.pivot_table(index="gen_model", columns=["dataset", "lang"], values="ok_rate")
    out = [
        r"\begin{table}[t]\centering\small",
        r"\caption{Valid-response rate per (dataset, language) averaged across the three runs. Lower values reflect empty, truncated, or refused generations, which count as a distinct failure mode from low MQS on a valid response. We retain models with zero valid responses in certain cells (e.g., Gemini 3.1 Flash-Lite) so that coverage gaps are visible rather than hidden.}",
        r"\label{tab:validity}",
        r"\begin{tabular}{l" + "c" * 6 + "}",
        r"\toprule",
        r"\textbf{Model} & \multicolumn{3}{c}{\textbf{Expert (149)}} & \multicolumn{3}{c}{\textbf{Non-expert (231)}} \\",
        r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}",
        r" & \textbf{EN} & \textbf{HI} & \textbf{MR} & \textbf{EN} & \textbf{HI} & \textbf{MR} \\",
        r"\midrule",
    ]
    cols = [("expert", "en"), ("expert", "hi"), ("expert", "mr"),
            ("non_expert", "en"), ("non_expert", "hi"), ("non_expert", "mr")]

    def row_fn(m):
        if m not in pivot.index:
            return None
        cells = [MODEL_DISPLAY[m]]
        for c in cols:
            try:
                v = pivot.loc[m, c]
            except Exception:
                v = None
            cells.append(_fmt_val(v))
        return " & ".join(cells) + r" \\"

    _emit_rows(out, MODEL_ORDER, row_fn)
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(out) + "\n")


def render_judge_agreement():
    path = TABLES / "tab_judge_agreement.tex"
    subset = REPO / "runs" / "calibration_subset.jsonl"
    judges = {"gpt_4o_mini": REPO / "runs" / "judge_cal__gpt_4o_mini.jsonl",
              "claude_opus_4_6": REPO / "runs" / "judge_cal__claude_opus_4_6.jsonl",
              "gpt_5_1": REPO / "runs" / "judge_cal__gpt_5_1.jsonl"}
    missing = [k for k, p in judges.items() if not p.exists()]
    if not subset.exists() or missing:
        path.write_text(
            "% calibration data not complete yet\n"
            "\\begin{table}[t]\\centering\\small\n"
            "\\caption{Three-judge agreement on the 500-response calibration subset. "
            f"Pending judges: {', '.join(missing) if missing else 'none'}.}}\n"
            "\\label{tab:judge_agreement}\n"
            "\\begin{tabular}{lccc}\n"
            "\\toprule\n"
            "\\textbf{Pair} & \\textbf{Pearson $\\rho$} & \\textbf{Mean $\\Delta$MQS} & \\textbf{Fleiss $\\kappa$ (all)} \\\\\n"
            "\\midrule\n"
            "GPT-4o-mini $\\leftrightarrow$ Claude Opus 4.6 & TBD & TBD & \\multirow{3}{*}{TBD} \\\\\n"
            "GPT-4o-mini $\\leftrightarrow$ GPT-5.1           & TBD & TBD & \\\\\n"
            "Claude Opus 4.6 $\\leftrightarrow$ GPT-5.1       & TBD & TBD & \\\\\n"
            "\\bottomrule\n"
            "\\end{tabular}\n"
            "\\end{table}\n"
        )
        return

    def load(path):
        out = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
                if r.get("error") or r.get("mqs") is None:
                    continue
                out[r["uid"]] = float(r["mqs"])
            except Exception:
                pass
        return out

    aliases = ["gpt_4o_mini", "claude_opus_4_6", "gpt_5_1"]
    scored: dict[str, dict[str, float]] = {a: load(judges[a]) for a in aliases}
    shared = set.intersection(*(set(d.keys()) for d in scored.values()))

    def pearson(xs, ys):
        n = len(xs)
        if n < 2:
            return float("nan")
        mx, my = sum(xs) / n, sum(ys) / n
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        dx = (sum((x - mx) ** 2 for x in xs) ** 0.5) or 1e-9
        dy = (sum((y - my) ** 2 for y in ys) ** 0.5) or 1e-9
        return num / (dx * dy)

    pairs = [("gpt_4o_mini", "claude_opus_4_6"), ("gpt_4o_mini", "gpt_5_1"), ("claude_opus_4_6", "gpt_5_1")]
    rows = []
    for a, b in pairs:
        xs = [scored[a][u] for u in shared]
        ys = [scored[b][u] for u in shared]
        rho = pearson(xs, ys)
        delta = mean([x - y for x, y in zip(xs, ys)]) if shared else float("nan")
        rows.append((a, b, rho, delta))

    # Bold highest Pearson (best agreement) and smallest |Delta| (smallest bias)
    rho_vals = [r[2] for r in rows]
    delta_abs = [abs(r[3]) for r in rows]
    best_rho_idx = rho_vals.index(max(rho_vals))
    best_bias_idx = delta_abs.index(min(delta_abs))

    out = [
        r"\begin{table}[t]\centering\small",
        r"\caption{Three-judge agreement on the stratified 500-response calibration subset ($n=" + str(len(shared)) + r"$ rows with all three judges present). Pearson $\rho$ is computed on MQS and \textbf{bold} marks the judge pair with highest rank agreement. Mean $\Delta$MQS is the signed bias; \underline{underline} marks the smallest absolute bias.}",
        r"\label{tab:judge_agreement}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"\textbf{Pair} & \textbf{Pearson $\rho$} & \textbf{Mean $\Delta$MQS} \\",
        r"\midrule",
    ]
    name = {"gpt_4o_mini": "GPT-4o-mini", "claude_opus_4_6": "Claude Opus 4.6", "gpt_5_1": "GPT-5.1"}
    for i, (a, b, rho, delta) in enumerate(rows):
        rho_str = f"{rho:.3f}"
        delta_str = f"{delta:+.3f}"
        if i == best_rho_idx:
            rho_str = r"\textbf{" + rho_str + "}"
        if i == best_bias_idx:
            delta_str = r"\underline{" + delta_str + "}"
        out.append(f"{name[a]} $\\leftrightarrow$ {name[b]} & {rho_str} & {delta_str} \\\\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(out) + "\n")


def main():
    render_mqs_overall()
    render_mqs_crosslingual()
    render_mqs_themes()
    render_cost_mqs()
    render_axis_failures()
    render_validity()
    render_judge_agreement()
    for f in sorted(TABLES.glob("*.tex")):
        print(f.relative_to(REPO))


if __name__ == "__main__":
    main()
