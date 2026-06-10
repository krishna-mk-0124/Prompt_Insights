import re

# A poor man's multilingual keyword normalization dictionary
# Maps common corporate terms in Spanish, Japanese, Chinese, etc. to English
KEYWORD_MAPPING = {
    # Spanish
    "contraseña": "password",
    "tarjeta": "card",
    "corporativa": "corporate",
    "informe": "report",
    "transacciones": "transaction",
    "comerciante": "merchant",
    "tipo": "rate",
    "cambio": "exchange",
    "liquidaciones": "settlements",
    "internacionales": "international",
    "beneficios": "benefits",
    "disputa": "dispute",
    "cargo": "charge",
    "cuenta": "account",
    "pautas": "guidelines",
    "cumplimiento": "compliance",
    "reembolso": "refund",
    "sala": "lounge",
    "límites": "limits",
    "riesgo": "risk",
    "panel": "dashboard",
    "error": "error",
    "acceder": "access",
    
    # Chinese (Simplified)
    "密码": "password",
    "卡": "card",
    "公司": "corporate",
    "报告": "report",
    "交易": "transaction",
    "商户": "merchant",
    "汇率": "exchange rate",
    "结算": "settlements",
    "国际": "international",
    "福利": "benefits",
    "争议": "dispute",
    "扣款": "charge",
    "账户": "account",
    "指南": "guidelines",
    "合规": "compliance",
    "退款": "refund",
    "休息室": "lounge",
    "限制": "limits",
    "风险": "risk",
    "仪表板": "dashboard",
    "错误": "error",
    "访问": "access",

    # Japanese
    "パスワード": "password",
    "カード": "card",
    "コーポレート": "corporate",
    "レポート": "report",
    "取引": "transaction",
    "加盟店": "merchant",
    "為替レート": "exchange rate",
    "決済": "settlements",
    "国際": "international",
    "特典": "benefits",
    "異議": "dispute",
    "請求": "charge",
    "アカウント": "account",
    "ガイドライン": "guidelines",
    "コンプライアンス": "compliance",
    "返金": "refund",
    "ラウンジ": "lounge",
    "制限": "limits",
    "リスク": "risk",
    "ダッシュボード": "dashboard",
    "エラー": "error",
    "アクセス": "access"
}

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
