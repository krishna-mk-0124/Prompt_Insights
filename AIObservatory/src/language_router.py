import pandas as pd
import os
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

# Seed the detector to ensure deterministic language detection
DetectorFactory.seed = 0

def detect_language(text):
    if pd.isna(text) or not str(text).strip():
        return "unknown"
    try:
        return detect(str(text))
    except LangDetectException:
        return "unknown"

def route_languages():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    input_file = os.path.join(data_dir, "prompt_sample.csv")
    
    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
        return
        
    print("Loading prompt_sample.csv...")
    # Attempt to read headers, fallback if no headers
    try:
        df = pd.read_csv(input_file)
        # If it's a single column without header, pandas treats the first row as header.
        # Check if the column name looks like a prompt instead of 'prompt_text'
        if len(df.columns) == 1 and len(str(df.columns[0]).split()) > 3:
             df = pd.read_csv(input_file, header=None, names=["prompt_text"])
        elif len(df.columns) == 1:
             df.columns = ["prompt_text"]
    except Exception as e:
        print(f"Error reading {input_file}: {e}")
        return

    print("Detecting languages (this may take a moment)...")
    df["language"] = df["prompt_text"].apply(detect_language)
    
    # Isolate English
    english_df = df[df["language"] == "en"].copy()
    non_english_df = df[df["language"] != "en"].copy()
    
    english_file = os.path.join(data_dir, "english_prompts.csv")
    non_english_file = os.path.join(data_dir, "non_english_prompts.csv")
    
    # Save the output (drop the language column from the English output to maintain schema)
    english_df[["prompt_text"]].to_csv(english_file, index=False)
    # Keep the language column in the non-English output so we know what they are
    non_english_df.to_csv(non_english_file, index=False)
    
    print(f"Routing complete!")
    print(f"- English prompts saved to: {english_file} ({len(english_df)} rows)")
    print(f"- Non-English prompts saved to: {non_english_file} ({len(non_english_df)} rows)")

if __name__ == "__main__":
    route_languages()
