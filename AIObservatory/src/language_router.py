import pandas as pd
import os
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

# Seed the detector to ensure deterministic language detection
DetectorFactory.seed = 0

import re

# Words that strongly indicate a technical prompt that should be routed to English 
# even if langdetect thinks it is another language (e.g. "sql error" -> id, "it support" -> it)
TECH_WORDS = {
    "sql", "query", "error", "issue", "problem", "update", "create", "delete", 
    "remove", "add", "token", "app", "user", "run", "test", "data", "db", 
    "table", "list", "use", "using", "get", "set", "make", "id", "support", 
    "server", "python", "java", "code", "script", "api", "rest", "select", 
    "from", "where", "join", "insert", "log", "file", "network", "system"
}

from functools import lru_cache

@lru_cache(maxsize=100000)
def cached_detect(text):
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"

def detect_language(text):
    if pd.isna(text) or not str(text).strip():
        return "unknown"
        
    s_text = str(text).strip()
    
    # 0. Fast Unicode Filter
    # If the text has a high ratio of non-ASCII characters (>15%) and >5 total unicode chars,
    # it is overwhelmingly likely to be Chinese, Japanese, Korean, Arabic, Russian, etc.
    # We instantly route it out to save time and prevent langdetect from being confused by English signatures.
    text_len = len(s_text)
    if text_len > 0:
        unicode_count = sum(1 for c in s_text if ord(c) > 127)
        if unicode_count > 5 and (unicode_count / text_len) > 0.15:
            return "unknown" # Instantly routes to the non-English pipeline
            
    words = set(re.findall(r'[a-záéíóúñäöüß]+', s_text.lower()))
    
    # 1. Primary language detection (cached for speedup on duplicate prompts)
    lang = cached_detect(s_text)
        
    # 2. Rescue misclassified short tech prompts
    # If langdetect thinks it's non-English, but it contains clear technical keywords,
    # override to English. We ONLY do this for short prompts (less than 150 chars)
    # because langdetect is highly accurate on long texts, and long foreign emails 
    # might accidentally contain a tech word like "from" in their English signature.
    if lang != "en" and lang != "unknown":
        if len(s_text) < 150 and words.intersection(TECH_WORDS):
            return "en"
            
    return lang

def route_languages():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    
    # Check for text file first, then fallback to csv
    input_file = os.path.join(data_dir, "prompt_sample.txt")
    if not os.path.exists(input_file):
        input_file = os.path.join(data_dir, "prompt_sample.csv")
        if not os.path.exists(input_file):
            print(f"File not found: prompt_sample.txt or prompt_sample.csv in {data_dir}")
            return
        
    print(f"Loading {input_file}...")
    
    # Read raw lines to avoid pandas splitting on commas inside the text
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
            
        # If the first line is a CSV header like 'prompt_text', skip it
        if lines and "prompt" in lines[0].lower() and len(lines[0].split()) == 1:
            lines = lines[1:]
            
        df = pd.DataFrame({"prompt_text": lines})
    except Exception as e:
        print(f"Error reading {input_file}: {e}")
        return

    print("Detecting languages (this may take several minutes)...")
    languages = []
    total = len(df)
    for i, text in enumerate(df["prompt_text"]):
        if i % 10000 == 0 and i > 0:
            print(f"  -> Routed {i:,} / {total:,} prompts ({(i/total)*100:.1f}%)")
        languages.append(detect_language(text))
    
    df["language"] = languages
    print(f"  -> Routed {total:,} / {total:,} prompts (100.0%)")
    
    # Isolate English
    english_df = df[df["language"] == "en"].copy()
    non_english_df = df[df["language"] != "en"].copy()
    
    english_file = os.path.join(data_dir, "english_prompts.txt")
    non_english_file = os.path.join(data_dir, "non_english_prompts.txt")
    
    # Save as raw text to avoid CSV quoting/header issues
    with open(english_file, "w", encoding="utf-8") as f:
        for prompt in english_df["prompt_text"]:
            f.write(f"{prompt}\n")
            
    with open(non_english_file, "w", encoding="utf-8") as f:
        for idx, row in non_english_df.iterrows():
            f.write(f"[{row['language']}] {row['prompt_text']}\n")
    
    print(f"Routing complete!")
    print(f"- English prompts saved to: {english_file} ({len(english_df)} rows)")
    print(f"- Non-English prompts saved to: {non_english_file} ({len(non_english_df)} rows)")

if __name__ == "__main__":
    route_languages()
