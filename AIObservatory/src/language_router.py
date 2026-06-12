import pandas as pd
import os
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

# Seed the detector to ensure deterministic language detection
DetectorFactory.seed = 0

import re

ENGLISH_WORDS = {
    "the", "be", "to", "of", "and", "that", "have", "it", "for", "not", "on", 
    "with", "as", "you", "at", "this", "but", "his", "by", "from", "they", "we", 
    "say", "her", "she", "or", "an", "will", "my", "one", "all", "would", "there", "their", 
    "what", "is", "are", "am", "was", "were", "been", "has", "had", "does", "did", "can", 
    "could", "should", "shall", "may", "might", "must", "how", "why", "when", "where", "who", 
    "which", "vs", "please", "your", "yours", "our", "ours", "us", 
    "yes", "if", "then", "else", "make", "get", "set", "use", "using", "list", "table", "db", 
    "sql", "query", "error", "issue", "problem", "update", "create", "delete", "remove", "add"
}

FOREIGN_WORDS = {
    "que", "de", "la", "el", "en", "y", "los", "se", "del", "las", "un", "por", "con", 
    "una", "su", "para", "es", "como", "más", "pero", "al", "lo", "esto", "o", "si", 
    "está", "dice", "cliente", "informacion", "español", "inglés", "traduci",
    "und", "die", "der", "das", "ist", "zu", "für", "auf", "mit", "sich", "des", "eine"
}

def detect_language(text):
    if pd.isna(text) or not str(text).strip():
        return "unknown"
        
    s_text = str(text).strip()
    # Strip punctuation and get pure alphabetic words (including accents)
    words = set(re.findall(r'[a-záéíóúñäöüß]+', s_text.lower()))
    
    # 1. Immediate rejection of known foreign stop words
    if words.intersection(FOREIGN_WORDS):
        return "non_english"
        
    # 2. Strong override for obvious English tech/structural words
    if words.intersection(ENGLISH_WORDS):
        return "en"
        
    # 3. Fallback to langdetect
    try:
        lang = detect(s_text)
        # Fix the "id" (Indonesian) or "it" (Italian) bug for short tech acronyms
        if lang in ['id', 'it'] and len(words) <= 3:
            return "en"
        return lang
    except LangDetectException:
        return "unknown"

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
