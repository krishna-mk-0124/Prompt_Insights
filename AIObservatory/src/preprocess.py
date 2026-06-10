import re
import json
import os

# Load expanded multilingual dictionary
# Maps foreign corporate/finance terms to English
dict_path = os.path.join(os.path.dirname(__file__), "corporate_dictionary.json")
try:
    with open(dict_path, "r", encoding="utf-8") as f:
        KEYWORD_MAPPING = json.load(f)
except FileNotFoundError:
    print(f"Warning: {dict_path} not found. Keyword normalization will be skipped.")
    KEYWORD_MAPPING = {}

# Regex to strip conversational fluff
STOP_PHRASES = [
    r"(?i)\bplease advise\b",
    r"(?i)\bthanks in advance\b",
    r"(?i)\bthanks\b",
    r"(?i)\burgent\b",
    r"(?i)\basap\b",
    r"(?i)\bhelp\b",
    r"(?i)\bpls\b"
]

def preprocess_prompt(text: str) -> str:
    if not isinstance(text, str):
        return ""
        
    cleaned = text
    
    # 1. Strip conversational fluff (stop-phrases)
    for phrase in STOP_PHRASES:
        cleaned = re.sub(phrase, "", cleaned)
        
    # 2. Mask dynamic entities (amounts, card digits, dates, etc.)
    # Mask currency amounts like $450.00, 100 USD, etc.
    cleaned = re.sub(r"(\$|€|£|¥)?\d+(,\d{3})*(\.\d{2})?\s?(usd|eur|gbp|jpy|cny)?", "<AMOUNT>", cleaned, flags=re.IGNORECASE)
    # Mask exactly 4 digits (usually card endings)
    cleaned = re.sub(r"\b\d{4}\b", "<CARD_DIGITS>", cleaned)
    
    # 3. Keyword Normalization
    # Note: A real implementation might use a tokenizer, but simple string replacement works for our proof-of-concept
    # We pad with spaces for Spanish to avoid substring issues, but for CJK characters spacing is different
    # so we do a direct replace.
    cleaned_lower = cleaned.lower()
    
    for foreign_word, english_word in KEYWORD_MAPPING.items():
        # Simple replace. For a full production system, word boundaries (\b) should be used for alphabetic languages.
        cleaned_lower = cleaned_lower.replace(foreign_word, english_word)
        
    # Clean up excess whitespace
    cleaned = re.sub(r"\s+", " ", cleaned_lower).strip()
    
    # Strip punctuation noise that might be left over from removing "Please advise."
    cleaned = re.sub(r"^[^\w\s]+|[^\w\s]+$", "", cleaned)
    
    return cleaned.strip()
