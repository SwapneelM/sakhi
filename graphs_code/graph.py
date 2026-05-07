import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# UI/Theme Setup - Enhanced for high quality
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.weight': 'semibold',
    'axes.labelweight': 'semibold',
    'axes.titleweight': 'bold',
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white'
})

# Constants
BASE_DIR = os.path.dirname(__file__)
FINAL_FILES_DIR = os.path.join(BASE_DIR, "final_files")
IMAGE_DIR = os.path.join(FINAL_FILES_DIR, "images")
MODELS = ["gpt_5_mini","gpt_4o_mini","cohere_command_a","gemini_2_5_pro","gemini_2_5_flash","llama_3_3_70b","llama_4_maverick","aya_expanse","medgemma_4b","medgemma_27b"]
MODEL_NAMES = {'cohere_command_a':'Command A','gemini_2_5_flash':'Gemini 2.5 Flash','gemini_2_5_pro':'Gemini 2.5 Pro','gpt_5_mini':'GPT-5 Mini','gpt_4o_mini':'GPT-4o Mini','llama_3_3_70b':'Llama 3.3 70B','llama_4_maverick':'Llama 4 Maverick','aya_expanse':'Aya Expanse','medgemma_4b':'MedGemma 4B','medgemma_27b':'MedGemma 27B'}
# 10 maximally distinct colors (spread across hue wheel; no blue/cyan/teal/green cluster)
COLORS = {
    "gpt_5_mini": "#C62828",        # Red
    "gpt_4o_mini": "#EF6C00",       # Orange
    "cohere_command_a": "#FFB300",  # Amber
    "gemini_2_5_pro": "#7B1FA2",    # Purple
    "gemini_2_5_flash": "#AD1457",  # Magenta
    "llama_3_3_70b": "#1565C0",     # Blue (only blue)
    "llama_4_maverick": "#EC407A",  # Pink / rose
    "aya_expanse": "#9E9D24",       # Olive / lime
    "medgemma_4b": "#2E7D32",       # Green
    "medgemma_27b": "#5D4037",      # Brown
}
CATS = {"Proprietary": (["gpt_5_mini", "gpt_4o_mini", "cohere_command_a", "gemini_2_5_pro", "gemini_2_5_flash"], "#E74C3C"), "Open-Source Large": (["llama_3_3_70b", "llama_4_maverick"], "#3498DB"), "Open-Source Small": (["aya_expanse", "medgemma_4b", "medgemma_27b"], "#2ECC71")}

def infer_language_from_col(col: str, default: str = "en") -> str:
    """Infer language from column name."""
    cl = col.lower()
    if cl.endswith("_hi") or "_hindi" in cl or "_hindi_" in cl:
        return "hindi"
    if cl.endswith("_mr") or "_marathi" in cl or "_marathi_" in cl:
        return "marathi"
    return default

def infer_rater_type_from_name(name: str) -> str:
    """Infer rater type from filename."""
    name_l = name.lower()
    if "expert" in name_l:
        return "expert"
    return "non_expert"

def load_data():
    """Load all scored CSV files from final_files directory."""
    os.makedirs(IMAGE_DIR, exist_ok=True)
    data = {}
    
    if not os.path.isdir(FINAL_FILES_DIR):
        print(f"ERROR: final_files directory not found: {FINAL_FILES_DIR}")
        return data
    
    csv_files = [f for f in os.listdir(FINAL_FILES_DIR) 
                 if (f.endswith("_scored.csv") or f in ["expert_scored.csv", "non_expert_scored.csv"]) 
                 and not f.startswith(".")]
    
    if not csv_files:
        print(f"WARNING: No scored CSV files found in {FINAL_FILES_DIR}")
        return data
    
    for fname in csv_files:
        fpath = os.path.join(FINAL_FILES_DIR, fname)
        try:
            df = pd.read_csv(fpath, low_memory=False)
            if df.empty:
                print(f"   Warning: {fname} is empty, skipping")
                continue
            
            base = os.path.splitext(fname)[0].replace("_scored", "")
            
            # Check what languages have actual model response columns
            langs_with_models = set()
            
            # Check for English columns (model columns without _hindi/_marathi)
            has_en_models = any(c.startswith(tuple(MODELS)) and 
                               "_hindi" not in c.lower() and "_marathi" not in c.lower() 
                               and not c.endswith("_hi") and not c.endswith("_mr")
                               for c in df.columns)
            if has_en_models:
                langs_with_models.add("en")
            
            # Check for Hindi columns
            has_hi_models = any(c.startswith(tuple(MODELS)) and 
                              ("_hindi" in c.lower() or c.endswith("_hi"))
                              for c in df.columns)
            if has_hi_models:
                langs_with_models.add("hindi")
            
            # Check for Marathi columns
            has_mr_models = any(c.startswith(tuple(MODELS)) and 
                              ("_marathi" in c.lower() or c.endswith("_mr"))
                              for c in df.columns)
            if has_mr_models:
                langs_with_models.add("marathi")
            
            # If no model columns found, assume English
            if not langs_with_models:
                langs_with_models.add("en")
            
            # Create entries only for languages that have model response columns
            for lang in sorted(langs_with_models):
                config_name = f"{base.replace('_', ' ').title()} ({lang.upper()})"
                data[(config_name, lang)] = df
                
        except Exception as e:
            print(f"   Warning: Failed to read {fname}: {e}, skipping")
            continue
    
    print(f"   Loaded {len(data)} dataset(s) from {len(csv_files)} file(s)")
    return data

def nc(df, cols): return [c for c in cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

def plot_metrics_summary(datasets, judge_name):
    """Plot metrics summary for all datasets."""
    if not datasets:
        print("   Warning: No datasets to plot")
        return
    
    num = len(datasets)
    fig, axes = plt.subplots(num, 1, figsize=(14, 7 * num), squeeze=False)
    axes = axes.flatten()
    handles, labels = [], []
    
    for idx, (ax, ((name, lang), df)) in enumerate(zip(axes, datasets.items())):
        rows = []
        for m in MODELS:
            # Filter columns by language
            if lang == "en":
                mqs_cols = [c for c in df.columns if c.startswith(m) and "judge_mqs_run" in c 
                           and "_hindi" not in c and "_marathi" not in c and not c.endswith("_hi") and not c.endswith("_mr")]
                # Semantic columns use pattern: model1_semantic_avg, model2_semantic_avg, model3_semantic_avg
                sem_cols = [c for c in df.columns if c.startswith(m) and ("1_semantic_avg" in c or "2_semantic_avg" in c or "3_semantic_avg" in c)
                           and "_hindi" not in c and "_marathi" not in c and not c.endswith("_hi") and not c.endswith("_mr")]
                # Linguistic columns use pattern: model1_linguistic_avg, model2_linguistic_avg, model3_linguistic_avg
                lin_cols = [c for c in df.columns if c.startswith(m) and ("1_linguistic_avg" in c or "2_linguistic_avg" in c or "3_linguistic_avg" in c)
                           and "_hindi" not in c and "_marathi" not in c and not c.endswith("_hi") and not c.endswith("_mr")]
            else:
                lang_suffix = "_hindi" if lang == "hindi" else "_marathi"
                lang_code = "_hi" if lang == "hindi" else "_mr"
                mqs_cols = [c for c in df.columns if c.startswith(m) and lang_suffix in c and "judge_mqs_run" in c]
                # For Hindi/Marathi: model_hindi1_hi_semantic_avg, model_marathi1_mr_semantic_avg, etc.
                sem_cols = [c for c in df.columns if c.startswith(m) and lang_suffix in c and lang_code in c and c.endswith("_semantic_avg")]
                lin_cols = [c for c in df.columns if c.startswith(m) and lang_suffix in c and lang_code in c and c.endswith("_linguistic_avg")]
            
            mqs = nc(df, mqs_cols)
            sem = nc(df, sem_cols)
            lin = nc(df, lin_cols)
            
            if mqs:
                r = {"model": MODEL_NAMES.get(m, m), "key": m, "MQS": df[mqs].mean().mean(), "MQS_std": df[mqs].mean().std()}
                if sem: 
                    s = df[sem].mean()
                    r.update({"Semantic": s.mean(), "Semantic_std": s.std()})
                if lin: 
                    l = df[lin].mean()
                    r.update({"Linguistic": l.mean(), "Linguistic_std": l.std()})
                rows.append(r)
        
        summ = pd.DataFrame(rows)
        if summ.empty:
            ax.text(0.5, 0.5, "No Data", ha='center', fontsize=12, weight='semibold')
            ax.set_title(f"{name} ({lang.upper()})", weight='bold')
            continue
        
        summ["color"] = summ["key"].map(COLORS)
        x, w = np.arange(len(summ)), 0.25
        
        # Metric colors matching reference: MQS (light blue), Semantic (red), Linguistic (teal)
        metric_colors = {
            "MQS": "#6B7FDE",      # Light blue
            "Semantic": "#E85D4E",  # Red
            "Linguistic": "#00BFA5" # Teal
        }
        
        for i, (met, col) in enumerate(zip(["MQS", "Semantic", "Linguistic"], 
                                           [metric_colors["MQS"], metric_colors["Semantic"], metric_colors["Linguistic"]])):
            if met in summ:
                v, s = summ[met].fillna(0), summ.get(f"{met}_std", 0)
                bars = ax.bar(x + i * w, v, w, color=col, label=met if idx == 0 else "", 
                             edgecolor='black', linewidth=0.8, alpha=0.95)
                if idx == 0 and met not in labels:
                    handles.append(bars)
                    labels.append(met)
                # Add value labels above bars (horizontal, not rotated)
                for b, val in zip(bars, v):
                    if val > 0:
                        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, 
                               f"{val:.3f}", ha="center", va="bottom", fontsize=9, 
                               weight='bold')
        
        # Add alternating background bands with subtle colors (matching reference)
        for i, (ridx, row) in enumerate(summ.iterrows()):
            # Use subtle alternating colors: light beige, light blue, light green
            band_colors = ['#F5F5DC', '#E6F3FF', '#E6FFE6']  # Beige, light blue, light green
            ax.axvspan(i - 0.45, i + 0.95, facecolor=band_colors[i % len(band_colors)], 
                      alpha=0.3, zorder=0)
        
        ax.set_xticks(x + w)
        ax.set_xticklabels(summ["model"], rotation=45, ha="right", fontsize=11, weight='semibold')
        ax.set_ylabel('Score', fontsize=12, weight='semibold')
        ax.set_title(f"{name} ({lang.upper()})", fontsize=14, weight='bold', pad=15)
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
        ax.grid(axis="y", alpha=0.3, linewidth=0.8, linestyle='--', which='major')
        ax.grid(axis="y", alpha=0.15, linewidth=0.5, linestyle=':', which='minor')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)
        ax.set_facecolor('white')
    
    if handles:
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), 
                  ncol=3, fontsize=12, frameon=False, handlelength=1.5, handletextpad=0.5)
    
    fig.patch.set_facecolor('white')
    plt.tight_layout(rect=[0, 0, 1, 0.96], pad=3.0, h_pad=3.0, w_pad=2.0)
    fn = f"metrics_summary_{judge_name.lower().replace('-', '_')}"
    plt.savefig(os.path.join(IMAGE_DIR, f"{fn}.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(IMAGE_DIR, f"{fn}.pdf"), bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: {fn}.png")

# Correlation plot: one professional figure per dataset, saved in a subfolder
CORRELATION_OUTPUT_DIR = "correlation"  # subfolder under IMAGE_DIR
CORRELATION_FIGSIZE = (9, 6)  # balanced aspect, not stretched
CORRELATION_X_TICK_STEP = 50   # Response length tick every 50 chars
CORRELATION_Y_TICK_STEP = 0.02 # MQS tick every 0.02
CORRELATION_AXIS_PADDING_X = 0.08  # fraction of x range to add as margin
CORRELATION_AXIS_PADDING_Y = 0.08  # fraction of y range to add as margin (min 0.02 for y)


def _safe_filename(name: str, lang: str) -> str:
    """Build a safe filename from config name and language."""
    safe = re.sub(r"[^\w\s-]", "", name).strip().replace(" ", "_").lower()
    if not safe:
        safe = "config"
    return f"{safe}_{lang.lower()}"


def plot_correlation(datasets, judge_name):
    """Plot correlation between response length and MQS: one professional figure per dataset."""
    if not datasets:
        return
    
    out_dir = os.path.join(IMAGE_DIR, CORRELATION_OUTPUT_DIR)
    os.makedirs(out_dir, exist_ok=True)
    
    for (name, lang), df in datasets.items():
        parsed = []
        for m in MODELS:
            if lang == "en":
                r = [c for c in df.columns if c.startswith(m) and c.endswith("_run1")
                     and "judge" not in c and "_hindi" not in c and "_marathi" not in c]
                mqs_runs = [c for c in df.columns if c.startswith(m) and "judge_mqs_run" in c
                            and "_hindi" not in c and "_marathi" not in c]
            else:
                lang_suffix = "_hindi" if lang == "hindi" else "_marathi"
                r = [c for c in df.columns if c.startswith(m) and lang_suffix in c
                     and c.endswith("_run1") and "judge" not in c]
                mqs_runs = [c for c in df.columns if c.startswith(m) and lang_suffix in c
                            and "judge_mqs_run" in c]
            
            if r and mqs_runs:
                l_val = df[r].astype(str).map(len).mean().mean()
                s_val = df[mqs_runs].mean(axis=1).mean()
                if pd.notna(l_val) and np.isfinite(s_val):
                    parsed.append({"len": l_val, "score": s_val, "name": MODEL_NAMES.get(m, m),
                                  "col": COLORS[m], "model": m})
        
        if not parsed or len(parsed) < 2:
            continue
        
        pdf = pd.DataFrame(parsed)
        x_min, x_max = pdf["len"].min(), pdf["len"].max()
        y_min, y_max = pdf["score"].min(), pdf["score"].max()
        x_range = (x_max - x_min) or 100
        y_range = (y_max - y_min) or 0.1
        
        # Axis limits with good spread and padding
        pad_x = max(x_range * CORRELATION_AXIS_PADDING_X, 25)
        pad_y = max(y_range * CORRELATION_AXIS_PADDING_Y, 0.02)
        x_lo = max(0, x_min - pad_x)
        x_hi = x_max + pad_x
        y_lo = max(0, y_min - pad_y)
        y_hi = min(1.0, y_max + pad_y)
        # Round to nice tick boundaries
        x_lo = (x_lo // CORRELATION_X_TICK_STEP) * CORRELATION_X_TICK_STEP
        x_hi = ((x_hi // CORRELATION_X_TICK_STEP) + 1) * CORRELATION_X_TICK_STEP
        y_lo = round(y_lo / CORRELATION_Y_TICK_STEP) * CORRELATION_Y_TICK_STEP
        y_hi = round(y_hi / CORRELATION_Y_TICK_STEP) * CORRELATION_Y_TICK_STEP
        y_lo = max(0, y_lo)
        y_hi = min(1.0, y_hi)
        
        fig, ax = plt.subplots(figsize=CORRELATION_FIGSIZE, facecolor="white")
        ax.set_facecolor("white")
        
        # Scatter: solid circles, thin dark outline
        for _, p in pdf.iterrows():
            ax.scatter(p["len"], p["score"], c=p["col"], s=80, edgecolors="black", linewidths=1.2,
                      zorder=3, label=p["name"])
        
        # Trend line
        z = np.polyfit(pdf["len"], pdf["score"], 1)
        t = np.poly1d(z)
        x_line = np.linspace(x_lo, x_hi, 100)
        ax.plot(x_line, t(x_line), color="#7F8C8D", linestyle="--", linewidth=2, zorder=2)
        
        # Axis setup: good spread, clear labels
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.xaxis.set_major_locator(plt.MultipleLocator(CORRELATION_X_TICK_STEP))
        ax.yaxis.set_major_locator(plt.MultipleLocator(CORRELATION_Y_TICK_STEP))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}"))
        ax.set_xlabel("Response Length (chars)", fontsize=12, weight="semibold", color="#333")
        ax.set_ylabel("MQS Score", fontsize=12, weight="semibold", color="#333")
        ax.tick_params(axis="both", labelsize=11, colors="#333")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.2)
        ax.spines["bottom"].set_linewidth(1.2)
        ax.spines["left"].set_color("#333")
        ax.spines["bottom"].set_color("#333")
        # No grid for clean look like reference
        ax.grid(False)
        
        # Title and subtitle (match reference: main title top left, subtitle below)
        short_judge = "GPT" if "gpt" in judge_name.lower() else judge_name.replace("-", " ").split()[0]
        fig.suptitle(f"Length vs MQS Correlation - {short_judge}",
                     fontsize=16, weight="bold", color="#333", x=0.02, ha="left", y=0.98)
        ax.set_title(f"{judge_name} Judge ({lang.upper()})", fontsize=13, weight="bold",
                     color="#2C3E50", pad=8, loc="left")
        
        # Legend top-right
        ax.legend(loc="upper right", fontsize=10, frameon=True, fancybox=False,
                  edgecolor=(0, 0, 0, 0.2), framealpha=1, facecolor="white")
        
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        safe_name = _safe_filename(name, lang)
        fn = f"correlation_{safe_name}.png"
        path = os.path.join(out_dir, fn)
        plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"   ✓ Saved: {CORRELATION_OUTPUT_DIR}/{fn}")

def plot_safety_analysis(datasets, judge_name):
    """Plot safety analysis (minimum MQS scores)."""
    if not datasets:
        return
    
    num = len(datasets)
    fig, axes = plt.subplots(num, 1, figsize=(12, 6 * num), squeeze=False)
    axes = axes.flatten()
    
    for idx, (ax, ((name, lang), df)) in enumerate(zip(axes, datasets.items())):
        rows, worst = [], []
        for m in MODELS:
            # Filter by language
            if lang == "en":
                mqs = [c for c in df if c.startswith(m) and "judge_mqs_run" in c
                      and "_hindi" not in c and "_marathi" not in c]
            else:
                lang_suffix = "_hindi" if lang == "hindi" else "_marathi"
                mqs = [c for c in df if c.startswith(m) and lang_suffix in c 
                      and "judge_mqs_run" in c]
            
            if not mqs:
                continue
            
            avg = df[mqs].mean(axis=1)
            mn = avg.min()
            mn_idx = avg[avg == mn].index.tolist()
            key = MODEL_NAMES.get(m, m)
            rows.append({"model": key, "key": m, "min": mn})
            worst.append({"model": key, "min": mn, "occ": len(mn_idx), "tot": len(avg.dropna())})
        
        if not rows:
            ax.text(0.5, 0.5, "No Data", ha='center', fontsize=12, weight='semibold')
            ax.set_title(f"{name} ({lang.upper()})", weight='bold')
            continue
        
        dd = pd.DataFrame(rows).sort_values("min")
        dd["col"] = dd.key.map(COLORS)
        bars = ax.barh(dd.model, dd["min"], color=dd.col, edgecolor="black", 
                      linewidth=1.0, alpha=0.9)
        
        for i, (m, v) in enumerate(zip(dd.model, dd["min"])):
            w = next(x for x in worst if x["model"] == m)
            ax.text(v + 0.015, i, f"{v:.3f} ({w['occ']/w['tot']*100:.1f}%)", 
                   va="center", fontsize=10, weight='semibold')
        
        ax.set_xlabel('Minimum MQS Score', fontsize=12, weight='semibold')
        ax.set_title(f"{name} ({lang.upper()})", fontsize=13, weight='bold', pad=12)
        ax.axvline(0.3, color="red", ls="--", linewidth=2.5, alpha=0.8, label='Critical Threshold')
        ax.xaxis.set_major_locator(plt.MultipleLocator(0.1))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(0.05))
        ax.grid(axis="x", alpha=0.3, linewidth=0.8, linestyle='--', which='major')
        ax.grid(axis="x", alpha=0.15, linewidth=0.5, linestyle=':', which='minor')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)
        ax.set_facecolor('white')
        if idx == 0:
            ax.legend(fontsize=10, frameon=True, fancybox=True, shadow=True)
    
    fig.patch.set_facecolor('white')
    plt.tight_layout(pad=2.5)
    fn = f"safety_{judge_name.lower().replace('-', '_')}"
    plt.savefig(os.path.join(IMAGE_DIR, f"{fn}.png"), dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(IMAGE_DIR, f"{fn}.pdf"), bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   ✓ Saved: {fn}.png")

def analyze_semantic_consistency(datasets, judge_name):
    """Analyze semantic consistency across runs."""
    results = []
    for (name, lang), df in datasets.items():
        for m in MODELS:
            # Filter by language
            if lang == "en":
                r = [c for c in df if c.startswith(m) and ("_run1" in c or "_run2" in c or "_run3" in c) 
                     and "judge" not in c and "_hindi" not in c and "_marathi" not in c]
                mqs_cols = [x for x in df if x.startswith(m) and "judge_mqs_run" in x
                           and "_hindi" not in x and "_marathi" not in x]
            else:
                lang_suffix = "_hindi" if lang == "hindi" else "_marathi"
                r = [c for c in df if c.startswith(m) and lang_suffix in c 
                     and ("_run1" in c or "_run2" in c or "_run3" in c) and "judge" not in c]
                mqs_cols = [x for x in df if x.startswith(m) and lang_suffix in x 
                           and "judge_mqs_run" in x]
            
            if len(r) >= 3:
                X = df[r].dropna()
                if len(X):
                    try:
                        v = TfidfVectorizer().fit_transform(X.values.flatten().astype(str))
                        n = len(X)
                        s = (np.diag(cosine_similarity(v[:n], v[n:2*n])) + 
                             np.diag(cosine_similarity(v[:n], v[2*n:])) + 
                             np.diag(cosine_similarity(v[n:2*n], v[2*n:]))) / 3
                        mqs = df[mqs_cols].mean(axis=1).values if mqs_cols else [np.nan] * len(X)
                        for val, m_val in zip(s, mqs):
                            results.append({"config": name, "lang": lang, "model": m, 
                                          "avg_sim": val, "mqs_mean": m_val})
                    except Exception as e:
                        print(f"   Warning: Semantic consistency failed for {m} in {name}: {e}")
                        continue
    
    if results:
        pd.DataFrame(results).to_csv(os.path.join(IMAGE_DIR, f"semantic_consistency_{judge_name.lower()}.csv"), index=False)
        print(f"   ✓ Saved: semantic_consistency_{judge_name.lower()}.csv")

def main():
    """Main function to generate all graphs."""
    try:
        print(">>> Loading data from final_files...")
        datasets = load_data()
        
        if not datasets:
            print(f"ERROR: No data found in {FINAL_FILES_DIR}")
            print("   Please run ling_semantic_scorer.py first to generate scored files.")
            return
        
        print(f"\n>>> Generating graphs for {len(datasets)} dataset(s)...")
        
        # Group by judge type (all use GPT-4o-mini judge)
        judge_name = "GPT-4o-mini"
        
        print("\n>>> Plotting metrics summary...")
        plot_metrics_summary(datasets, judge_name)
        
        print("\n>>> Plotting correlation analysis...")
        plot_correlation(datasets, judge_name)
        
        print("\n>>> Plotting safety analysis...")
        plot_safety_analysis(datasets, judge_name)
        
        print("\n>>> Analyzing semantic consistency...")
        analyze_semantic_consistency(datasets, judge_name)
        
        print(f"\n>>> Success! All graphs written to: {IMAGE_DIR}")
        
    except Exception as e:
        print(f"\nERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__": main()