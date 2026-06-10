import re

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
    
    # Clean up excess whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    
    # Strip punctuation noise that might be left over from removing "Please advise."
    cleaned = re.sub(r"^[^\w\s]+|[^\w\s]+$", "", cleaned)
    
    return cleaned.strip()
