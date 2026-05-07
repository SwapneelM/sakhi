import os
import re
import sys
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

_GEN_DIR = Path(__file__).resolve().parent
if str(_GEN_DIR) not in sys.path:
    sys.path.insert(0, str(_GEN_DIR))

from csv_prep import prepare_for_multilingual_generation
from paths import (
    LLM_ENGLISH_EXPERT_150,
    LLM_ENGLISH_NON_EXPERT_230,
    LLM_MULTILINGUAL_EXPERT_150,
    LLM_MULTILINGUAL_NON_EXPERT_230,
    OUTPUTS,
)
from time import time, sleep
from concurrent.futures import ThreadPoolExecutor, as_completed
import dspy
import json

SAKHI_PROMPT_MULTILINGUAL = """You are a knowledgeable and caring assistant trained to support pregnancy-related health. Your task is to provide accurate, empathetic, and reliable answers to user questions specifically about pregnancy, prenatal care, and postnatal well-being.

CRITICAL LANGUAGE REQUIREMENT: You MUST answer ONLY in {language}. DO NOT use English. Your entire response must be in {language} script and language. This is absolutely mandatory.

TONE: Use a warm, supportive, and informative tone. Avoid medical jargon; explain any technical term in plain, easy-to-understand language in {language}.

SCOPE: Respond to questions related to women's health, including pregnancy, prenatal care, postnatal well-being, reproductive health and contraception, sexual health and wellness, menstrual health and disorders, common gynecological conditions, nutrition, mental health, and general wellness unique to women.

For very short or unclear questions, assume a pregnancy-related intent and restate the implied question clearly before answering in {language}.

ANSWER FORMAT: Provide a single-paragraph answer that is clear, concise, and around 60-80 words, maximum 100 words. Do NOT use bullet points or lists. Always recommend consulting a doctor for serious symptoms, diagnoses, or uncertainties.

QUESTION: {question}

Remember: Your answer MUST be entirely in {language}. Provide only the answer text without any tags or formatting."""

class HealthAssistant(dspy.Signature):
    question = dspy.InputField()
    answer = dspy.OutputField()

def extract_clean_answer(text):
    if pd.isna(text) or not text:
        return ""
    text = str(text).strip()
    text = re.sub(r"<[^>]+>", "", text)
    return text.strip()

def is_valid_answer(x):
    if pd.isna(x) or x is None:
        return False
    text = extract_clean_answer(x)
    if not text or len(text) < 5:
        return False
    return bool(re.search(r'[A-Za-z0-9\u0900-\u097F\u0980-\u09FF]', text))

def format_time(s):
    if s < 60:
        return f"{int(s)}s"
    if s < 3600:
        return f"{s/60:.1f}m"
    return f"{s/3600:.1f}h"

class CohereLM(dspy.LM):
    def __init__(self, model_name: str, api_key: str):
        import cohere
        super().__init__(model=model_name, temperature=0.7, max_tokens=300)
        self.model_name = model_name
        self.client = cohere.ClientV2(api_key)
        self.provider = "cohere"
        self.history = []
        self.temp = 0.7
        self.max_tok = 300
    def __call__(self, prompt=None, messages=None, **kwargs):
        if messages is None:
            messages = [{"role": "user", "content": prompt}]
        for attempt in range(3):
            try:
                response = self.client.chat(model=self.model_name, messages=messages, temperature=self.temp, max_tokens=self.max_tok)
                content = response.message.content[0].text
                if content and content.strip():
                    content = content.strip()
                    content = re.sub(r'```json\s*', '', content)
                    content = re.sub(r'```\s*', '', content)
                    content = content.strip()
                    try:
                        parsed = json.loads(content)
                        if isinstance(parsed, dict) and 'answer' in parsed:
                            return [parsed['answer']]
                    except json.JSONDecodeError:
                        pass
                    return [content]
                if attempt < 2:
                    sleep(2)
            except Exception as e:
                if attempt < 2:
                    sleep(2)
                else:
                    error_msg = str(e)[:200]
                    print(f"        Cohere API Error: {error_msg}")
                    return [""]
        return [""]

class OpenRouterLM(dspy.LM):
    def __init__(self, model_name: str, api_key: str):
        from openai import OpenAI
        if "gpt-5" in model_name.lower() or "gpt5" in model_name.lower():
            super().__init__(model=model_name, temperature=1.0, max_tokens=16000)
        else:
            super().__init__(model=model_name, temperature=0.7, max_tokens=300)
        self.model_name = model_name
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key, timeout=120)
        self.provider = "openrouter"
        self.history = []
        self.temp = 1.0 if ("gpt-5" in model_name.lower() or "gpt5" in model_name.lower()) else 0.7
        self.max_tok = 16000 if ("gpt-5" in model_name.lower() or "gpt5" in model_name.lower()) else 300
    def __call__(self, prompt=None, messages=None, **kwargs):
        if messages is None:
            messages = [{"role": "user", "content": prompt}]
        extra_body = {}
        if "gemini" in self.model_name.lower():
            extra_body = {"reasoning": {"effort": "low", "exclude": True}}
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(model=self.model_name, messages=messages, max_tokens=self.max_tok, temperature=self.temp, timeout=120, extra_body=extra_body)
                content = response.choices[0].message.content
                if content and content.strip():
                    content = content.strip()
                    content = re.sub(r'```json\s*', '', content)
                    content = re.sub(r'```\s*', '', content)
                    content = content.strip()
                    try:
                        parsed = json.loads(content)
                        if isinstance(parsed, dict) and 'answer' in parsed:
                            return [parsed['answer']]
                    except json.JSONDecodeError:
                        pass
                    return [content]
                if attempt < 2:
                    sleep(2)
            except Exception as e:
                if attempt < 2:
                    sleep(2)
                else:
                    error_msg = str(e)[:200]
                    print(f"        API Error: {error_msg}")
                    return [""]
        return [""]

class HuggingFaceLM(dspy.LM):
    def __init__(self, model_name: str, api_key: str, endpoint_url: str):
        import requests
        super().__init__(model=model_name, temperature=0.7, max_tokens=300)
        self.model_name = model_name
        self.api_key = api_key
        self.endpoint_url = endpoint_url
        self.session = requests.Session()
        self.provider = "huggingface"
        self.history = []
        self.warmed_up = False
    def warmup(self):
        if self.warmed_up:
            return
        print("      🔥 Warming up endpoint...")
        warmup_prompt = "Hello, how are you?"
        try:
            response = self.session.post(self.endpoint_url, headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}, json={"inputs": warmup_prompt, "parameters": {"max_new_tokens": 50, "temperature": 0.7}}, timeout=300)
            response.raise_for_status()
            self.warmed_up = True
            print("      ✓ Endpoint warmed up successfully")
        except Exception as e:
            print(f"      ⚠️ Warmup failed: {str(e)[:100]}")
    def __call__(self, prompt=None, messages=None, **kwargs):
        if messages is None:
            prompt_text = prompt
        else:
            prompt_text = messages[-1]["content"] if messages else prompt
        for attempt in range(5):
            try:
                timeout = 300 if attempt == 0 else 180 + (attempt * 30)
                response = self.session.post(self.endpoint_url, headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}, json={"inputs": prompt_text, "parameters": {"max_new_tokens": 300, "temperature": 0.7, "repetition_penalty": 1.2, "return_full_text": False, "num_return_sequences": 1, "do_sample": True}}, timeout=timeout)
                response.raise_for_status()
                result = response.json()
                generated_text = ""
                if isinstance(result, list) and len(result) > 0:
                    generated_text = result[0].get("generated_text", "")
                elif isinstance(result, dict) and "generated_text" in result:
                    generated_text = result.get("generated_text", "")
                if generated_text:
                    generated_text = re.sub(r"<[^>]+>", "", generated_text)
                    if "USER QUESTION:" in generated_text or "QUESTION:" in generated_text:
                        parts = re.split(r"USER QUESTION:|QUESTION:", generated_text)
                        generated_text = parts[0].strip()
                    generated_text = re.sub(r'```\w*', '', generated_text)
                    generated_text = re.sub(r'\s+', ' ', generated_text)
                    lines = generated_text.split('\n')
                    cleaned_lines = []
                    for line in lines:
                        line = line.strip()
                        if line and not line.startswith("I can only answer questions"):
                            cleaned_lines.append(line)
                    generated_text = ' '.join(cleaned_lines).strip()
                    return [generated_text]
                if attempt < 4:
                    wait_time = 2 ** attempt
                    sleep(wait_time)
            except Exception as e:
                error_msg = str(e)[:200]
                if attempt < 4:
                    wait_time = 2 ** attempt
                    print(f"        HF API Error (attempt {attempt + 1}/5): {error_msg}, retrying in {wait_time}s...")
                    sleep(wait_time)
                else:
                    print(f"        HF API Error (final attempt): {error_msg}")
                    return [""]
        return [""]

def process_single_row(idx, question, lm, prompt_template, language):
    try:
        full_prompt = prompt_template.format(question=question, language=language)
        response = lm(prompt=full_prompt)
        clean_resp = extract_clean_answer(response[0] if response else "")
        return idx, clean_resp, None
    except Exception as e:
        error_msg = str(e)[:200]
        return idx, "", error_msg

def generate_multilingual_responses(csv_path: str, output_path: str, language: str, question_col: str, max_workers: int = 10, progress_interval: int = 50):
    load_dotenv()
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    cohere_api_key = os.getenv("COHERE_API_KEY")
    
    if not openrouter_api_key:
        print("⚠️ OPENROUTER_API_KEY not found in environment")
        return None
    
    print(f"\n🔍 Generating responses in {language}")
    print(f"  OPENROUTER_API_KEY: {'✓ Found' if openrouter_api_key else '✗ Missing'}")
    print(f"  COHERE_API_KEY: {'✓ Found' if cohere_api_key else '✗ Missing'}\n")
    
    models = {
        "cohere_command_a": ("cohere/command-a", "openrouter"),
        "gemini_2_5_flash": ("google/gemini-2.5-flash", "openrouter"),
        "gemini_2_5_pro": ("google/gemini-2.5-pro", "openrouter"),
        "gpt_5_mini": ("openai/gpt-5-mini", "openrouter"),
        "gpt_4o_mini": ("openai/gpt-4o-mini", "openrouter"),
        "llama_3_3_70b": ("meta-llama/llama-3.3-70b-instruct", "openrouter"),
        "llama_4_maverick": ("meta-llama/llama-4-maverick", "openrouter"),
        "aya_expanse": ("c4ai-aya-expanse-32b", "cohere")
    }
    
    df = prepare_for_multilingual_generation(pd.read_csv(csv_path))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    if os.path.exists(output_path):
        df_out = pd.read_csv(output_path)
        print(f"✓ Loaded existing output file: {len(df_out)} rows")
        for col in df.columns:
            if col not in df_out.columns:
                df_out[col] = df[col]
    else:
        df_out = df.copy()
        print(f"✓ Created new output from input: {len(df_out)} rows")
    
    if len(df_out) != len(df):
        print(f"⚠️ WARNING: Row count mismatch! Recreating from input...")
        df_out = df.copy()
    
    lang_suffix = "hindi" if language == "Hindi" else "marathi"
    
    for model_short, (model_full, provider) in models.items():
        print(f"\n{'='*60}")
        print(f"Starting model: {model_short} ({model_full}) via {provider} - Language: {language}")
        print(f"{'='*60}")
        
        if provider == "cohere" and not cohere_api_key:
            print(f"⚠️ Skipping {model_short}: COHERE_API_KEY not found")
            continue
        
        for run in range(1, 4):
            col_name = f"{model_short}_{lang_suffix}_run{run}"
            print(f"\n  → Run {run}/3: {col_name}")
            
            if col_name not in df_out.columns:
                df_out[col_name] = pd.Series(dtype='object')
            else:
                if df_out[col_name].dtype != 'object':
                    df_out[col_name] = df_out[col_name].astype('object')
            
            valid_mask = df_out[col_name].apply(is_valid_answer)
            if valid_mask.all():
                print(f"    ✓ All responses already valid, skipping...")
                continue
            
            has_question = df[question_col].notna() & (df[question_col].astype(str).str.strip() != '')
            rows_to_process = df_out[has_question & ~valid_mask].index.tolist()
            
            if not rows_to_process:
                print(f"    ✓ All responses already valid")
                continue
            
            print(f"    → Processing {len(rows_to_process)} rows with {max_workers} workers")
            
            try:
                if provider == "cohere":
                    lm = CohereLM(model_full, cohere_api_key)
                else:
                    lm = OpenRouterLM(model_full, openrouter_api_key)
            except Exception as e:
                print(f"    ✗ Failed to initialize: {e}")
                continue
            
            start_time = time()
            completed = 0
            failed = 0
            
            try:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {executor.submit(process_single_row, idx, df.loc[idx, question_col], lm, SAKHI_PROMPT_MULTILINGUAL, language): idx for idx in rows_to_process}
                    
                    for future in as_completed(futures):
                        idx, clean_resp, error = future.result()
                        if error:
                            print(f"      ✗ Row {idx}: {error}")
                            failed += 1
                        elif not clean_resp:
                            print(f"      ✗ Row {idx}: Empty response")
                            failed += 1
                        else:
                            completed += 1
                        
                        df_out.loc[idx, col_name] = clean_resp
                        
                        if (completed + failed) % progress_interval == 0:
                            df_out.to_csv(output_path, index=False)
                            elapsed = time() - start_time
                            remaining = len(rows_to_process) - completed - failed
                            rate = (completed + failed) / elapsed if elapsed > 0 else 0
                            eta = remaining / rate if rate > 0 else 0
                            print(f"      Progress: {completed}/{len(rows_to_process)} completed, {failed} failed, ETA: {format_time(eta)}")
            
            except KeyboardInterrupt:
                print(f"\n    ⚠️ Interrupted by user, saving progress...")
                df_out.to_csv(output_path, index=False)
                raise
            
            df_out.to_csv(output_path, index=False)
            elapsed = time() - start_time
            print(f"    ✓ Run {run} complete: {completed} success, {failed} failed in {format_time(elapsed)}")
        
        df_out.to_csv(output_path, index=False)
        print(f"\n✓ Completed all runs for {model_short} in {language}")
    
    return df_out

def generate_huggingface_multilingual_responses(csv_path: str, output_path: str, language: str, question_col: str, max_workers: int = 3, progress_interval: int = 25):
    load_dotenv()
    api_key = os.getenv("HUGGINGFACE_API_KEY")
    
    if not api_key:
        print("⚠️ HUGGINGFACE_API_KEY not found in environment")
        return None
    
    print(f"\n🔍 Generating HuggingFace responses in {language}")
    print(f"  HUGGINGFACE_API_KEY: {'✓ Found' if api_key else '✗ Missing'}")
    print(f"  MEDGEMMA_4B_ENDPOINT_URL: {os.getenv('MEDGEMMA_4B_ENDPOINT_URL', '✗ Missing')}")
    print(f"  MEDGEMMA_27B_ENDPOINT_URL: {os.getenv('MEDGEMMA_27B_ENDPOINT_URL', '✗ Missing')}\n")
    
    models = {
        "medgemma_4b": ("medgemma-4b-it-kcz", os.getenv("MEDGEMMA_4B_ENDPOINT_URL")),
        "medgemma_27b": ("medgemma-27b-text-it-zum", os.getenv("MEDGEMMA_27B_ENDPOINT_URL"))
    }
    
    df = prepare_for_multilingual_generation(pd.read_csv(csv_path))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    if os.path.exists(output_path):
        df_out = pd.read_csv(output_path)
        print(f"✓ Loaded existing output file: {len(df_out)} rows")
        for col in df.columns:
            if col not in df_out.columns:
                df_out[col] = df[col]
    else:
        df_out = df.copy()
        print(f"✓ Created new output from input: {len(df_out)} rows")
    
    if len(df_out) != len(df):
        print(f"⚠️ WARNING: Row count mismatch! Recreating from input...")
        df_out = df.copy()
    
    lang_suffix = "hindi" if language == "Hindi" else "marathi"
    
    for model_short, (model_name, endpoint_url) in models.items():
        print(f"\n{'='*60}")
        print(f"Starting model: {model_short} ({model_name}) - Language: {language}")
        print(f"{'='*60}")
        
        if not endpoint_url:
            print(f"⚠️ Skipping {model_short}: No endpoint URL found")
            continue
        
        print(f"  ℹ️  Using sequential processing (1 worker) to avoid cold starts")
        
        for run in range(1, 4):
            col_name = f"{model_short}_{lang_suffix}_run{run}"
            print(f"\n  → Run {run}/3: {col_name}")
            
            if col_name not in df_out.columns:
                df_out[col_name] = pd.Series(dtype='object')
            else:
                if df_out[col_name].dtype != 'object':
                    df_out[col_name] = df_out[col_name].astype('object')
            
            valid_mask = df_out[col_name].apply(is_valid_answer)
            if valid_mask.all():
                print(f"    ✓ All responses already valid, skipping...")
                continue
            
            has_question = df[question_col].notna() & (df[question_col].astype(str).str.strip() != '')
            rows_to_process = df_out[has_question & ~valid_mask].index.tolist()
            
            if not rows_to_process:
                print(f"    ✓ All responses already valid")
                continue
            
            est_time_per_req = 20
            estimated_total = (len(rows_to_process) * est_time_per_req)
            print(f"    → Processing {len(rows_to_process)} rows sequentially")
            print(f"    ⏱️  Estimated time: {format_time(estimated_total)} (assuming ~{est_time_per_req}s per request)")
            
            try:
                lm = HuggingFaceLM(model_name, api_key, endpoint_url)
                lm.warmup()
            except Exception as e:
                print(f"    ✗ Failed to initialize: {e}")
                continue
            
            start_time = time()
            completed = 0
            failed = 0
            
            try:
                for i, idx in enumerate(rows_to_process):
                    question = df.loc[idx, question_col]
                    idx_result, clean_resp, error = process_single_row(idx, question, lm, SAKHI_PROMPT_MULTILINGUAL, language)
                    
                    if error:
                        print(f"      ✗ Row {idx}: {error}")
                        failed += 1
                    elif not clean_resp:
                        print(f"      ✗ Row {idx}: Empty response")
                        failed += 1
                    else:
                        completed += 1
                    
                    df_out.loc[idx, col_name] = clean_resp
                    
                    if (completed + failed) % progress_interval == 0:
                        df_out.to_csv(output_path, index=False)
                        elapsed = time() - start_time
                        remaining = len(rows_to_process) - completed - failed
                        rate = (completed + failed) / elapsed if elapsed > 0 else 0
                        eta = remaining / rate if rate > 0 else 0
                        print(f"      📊 Progress: {completed}/{len(rows_to_process)} completed, {failed} failed, {rate:.2f} req/s, ETA: {format_time(eta)}")
            
            except KeyboardInterrupt:
                print(f"\n    ⚠️ Interrupted by user, saving progress...")
                df_out.to_csv(output_path, index=False)
                raise
            
            df_out.to_csv(output_path, index=False)
            elapsed = time() - start_time
            avg_rate = completed / elapsed if elapsed > 0 else 0
            print(f"    ✓ Run {run} complete: {completed} success, {failed} failed in {format_time(elapsed)} ({avg_rate:.2f} req/s)")
        
        df_out.to_csv(output_path, index=False)
        print(f"\n✓ Completed all runs for {model_short} in {language}")
    
    return df_out

def _run_multilingual_for_pair(english_csv: Path, multilingual_csv: Path, label: str) -> None:
    if not english_csv.exists():
        print(f"⚠️ Skipping {label}: missing English generations at {english_csv} (run generating_llm_responses.py first)")
        return
    inp, out = str(english_csv), str(multilingual_csv)
    print("\n" + "=" * 80)
    print(f"STARTING HINDI GENERATION — {label}")
    print("=" * 80)
    generate_multilingual_responses(inp, out, language="Hindi", question_col="question_hi", max_workers=10, progress_interval=50)
    generate_huggingface_multilingual_responses(inp, out, language="Hindi", question_col="question_hi", max_workers=10, progress_interval=25)
    print("\n" + "=" * 80)
    print(f"STARTING MARATHI GENERATION — {label}")
    print("=" * 80)
    generate_multilingual_responses(inp, out, language="Marathi", question_col="question_mr", max_workers=10, progress_interval=50)
    generate_huggingface_multilingual_responses(inp, out, language="Marathi", question_col="question_mr", max_workers=10, progress_interval=25)


if __name__ == "__main__":
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    # Extends English CSVs (same row order) with Hindi/Marathi model columns.
    _run_multilingual_for_pair(LLM_ENGLISH_NON_EXPERT_230, LLM_MULTILINGUAL_NON_EXPERT_230, "230-row non-expert")
    _run_multilingual_for_pair(LLM_ENGLISH_EXPERT_150, LLM_MULTILINGUAL_EXPERT_150, "150-row expert")
    print("\n" + "=" * 80)
    print("ALL MULTILINGUAL GENERATION COMPLETE!")
    print("=" * 80)