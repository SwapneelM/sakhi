"""
Generate 4-5 candidate process-diagram images for the Sakhi review-pipeline figure
via OpenRouter's image-generation models (Gemini Imagen family).

Saves each candidate to overleaf/paper/pictures/candidates/fig_pipeline_imagen_<n>.png
along with a JSON manifest of the prompt used and any text the model returned.

The current pipeline diagram (matplotlib boxes) has been criticised for:
- no stakeholder icons (ASHA workers, mothers, AI/mobile, doctors)
- text overflowing the box borders
- generic block-and-arrow look that does not communicate the "language is embedded
  in culture" framing the senior coauthor asked for.

This script asks an image model for a high-density, paper-publishable pipeline
illustration with explicit icons, labelled boxes, and arrows.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "overleaf" / "paper" / "pictures" / "candidates"
OUT_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(Path.home() / ".env.openrouter")
API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not API_KEY:
    sys.exit("OPENROUTER_API_KEY missing - source ~/.env.openrouter first")

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

BASE_PROMPT = """A clean, publication-quality scientific process diagram for a NeurIPS
paper, in the style of figure 1 of the PARTNR paper or the Artificial Hivemind
header figure: muted colours, flat-vector clipart icons, soft drop-shadows,
sans-serif labels, generous whitespace.

Title at the top: "Sakhi Benchmark: Three-Channel Community Review Pipeline"

LEFT COLUMN — KNOWLEDGE SOURCES (one stacked group):
  - Icon: stack of three document books labelled "WHO ANC", "India NHM",
    "ANM Manual" (each on its own document icon).
  - Group caption: "Source corpus".

ARROW from left group to MIDDLE COLUMN.

MIDDLE COLUMN — GENERATION (single rounded box, robot icon top-left of box):
  - Title inside box (bold): "AI generation".
  - Sub-text inside box: "Aya Expanse drafter -> MedGemma validator".
  - Sub-text below: "845 candidate Q&A pairs (EN / HI / MR)".
  - Make sure text fits inside the box with comfortable padding.

ARROW from middle box to a horizontal row of THREE channel cards.

RIGHT COLUMN — THREE COMMUNITY-REVIEW CHANNELS (three rounded cards, side by side):
  - Card 1, icon: stethoscope + female-doctor avatar.
    Title: "Doctor channel".
    Sub-text: "Clinical validation, edits".
    Footer chip: "n=11 doctors, 9 GP / 2 OB-GYN".
  - Card 2, icon: smartphone + woman-with-headscarf community-health-worker avatar.
    Title: "ASHA-worker channel".
    Sub-text: "Sociolinguistic grounding".
    Footer chip: "Marathi & Hindi rural India".
  - Card 3, icon: speech-bubble + group-of-mothers avatar.
    Title: "Nonprofit-staff channel".
    Sub-text: "Cultural & family-dynamics review".
    Footer chip: "Maternal community context".

ARROWS from each of the three cards converge into ONE outcome panel below.

OUTCOME PANEL (wide rounded box at the bottom, two split halves):
  Left half title: "Expert track".
    Sub-text: "149 doctor-edited reference responses".
  Right half title: "Non-expert track".
    Sub-text: "231 community-sourced, reviewed by nonprofit frontline staff (NOT doctors)".

To the right of the outcome panel, a small detached card:
  Icon: clipboard with check marks.
  Title: "Doctor calibration set".
  Sub-text: "148 questions, 169 verdicts, 21 doubly rated".

Visual rules:
  - All TEXT MUST fit inside its box with at least 12 px padding; do not let text
    cross the border of any box.
  - Use a calm palette of dark slate, soft teal, soft coral, warm beige; no neon.
  - Use thin, dark-grey arrows with small triangular arrowheads.
  - Use flat 2D clipart icons, not photographs and not 3D rendered scenes.
  - White background, no decorative gradients or starbursts.
  - Aspect ratio 16:9, suitable for a two-column NeurIPS figure.
  - Render all text crisply at high resolution; do not blur or stylise letters.
  - Avoid lorem-ipsum or fake placeholder strings; only use the labels above.
"""

VARIANTS = [
    ("v6_minimal_line_fixed",
     "Minimal line-art style with one accent colour (teal). Black thin outlines, "
     "icons drawn as simple line glyphs, no fills. Use the EXACT non-expert track "
     "label as written in the prompt; do NOT paraphrase."),
    ("v7_warm_clipart_fixed",
     "Warm-coloured flat clipart, friendly avatars. Use the EXACT non-expert track "
     "label as written in the prompt; do NOT paraphrase."),
]

MODEL = "google/gemini-3-pro-image-preview"
TIMEOUT_S = 240


def request_image(prompt_text: str, variant_name: str) -> dict:
    payload = {
        "model": MODEL,
        "modalities": ["image", "text"],
        "messages": [
            {"role": "user", "content": prompt_text},
        ],
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/SwapneelM/MedicaLLM-Eval",
        "X-Title": "Sakhi pipeline figure candidate generation",
    }
    r = requests.post(ENDPOINT, headers=headers, json=payload, timeout=TIMEOUT_S)
    if r.status_code != 200:
        return {"error": f"HTTP {r.status_code}: {r.text[:500]}"}
    return r.json()


def extract_image_bytes(resp: dict) -> bytes | None:
    """Pull image bytes out of an OpenRouter response. Format varies by model."""
    try:
        choice = resp["choices"][0]["message"]
    except Exception:
        return None
    # OpenRouter image responses can come back in multiple shapes:
    images = choice.get("images") or []
    for img in images:
        url = img.get("image_url", {}).get("url") if isinstance(img, dict) else None
        if not url:
            continue
        if url.startswith("data:image"):
            b64 = url.split(",", 1)[1]
            return base64.b64decode(b64)
        if url.startswith("http"):
            try:
                rr = requests.get(url, timeout=60)
                if rr.status_code == 200:
                    return rr.content
            except Exception:
                pass
    content = choice.get("content")
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") in {"image", "image_url"}:
                url = part.get("image_url", {}).get("url") if isinstance(part.get("image_url"), dict) else part.get("url")
                if url and url.startswith("data:image"):
                    return base64.b64decode(url.split(",", 1)[1])
    return None


def main():
    manifest = {
        "model": MODEL,
        "base_prompt": BASE_PROMPT,
        "variants": [],
    }
    for tag, style_note in VARIANTS:
        full_prompt = BASE_PROMPT.strip() + "\n\nStyle for this candidate: " + style_note
        out_png = OUT_DIR / f"fig_pipeline_imagen_{tag}.png"
        print(f"[{tag}] requesting from {MODEL} ...", flush=True)
        t0 = time.time()
        resp = request_image(full_prompt, tag)
        dt = time.time() - t0
        entry = {
            "tag": tag,
            "style_note": style_note,
            "elapsed_s": round(dt, 1),
            "out_path": str(out_png.relative_to(ROOT)),
        }
        if "error" in resp:
            entry["error"] = resp["error"]
            print(f"  ERROR: {resp['error']}", flush=True)
        else:
            img_bytes = extract_image_bytes(resp)
            if img_bytes is None:
                entry["error"] = "no image bytes in response"
                snippet = json.dumps(resp)[:600]
                entry["raw_snippet"] = snippet
                print(f"  no image bytes; raw response snippet: {snippet}", flush=True)
            else:
                out_png.write_bytes(img_bytes)
                entry["bytes"] = len(img_bytes)
                print(f"  wrote {out_png.name} ({len(img_bytes)} bytes, {dt:.1f}s)", flush=True)
        manifest["variants"].append(entry)
    manifest_path = OUT_DIR / "pipeline_imagen_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nmanifest -> {manifest_path}")


if __name__ == "__main__":
    main()
