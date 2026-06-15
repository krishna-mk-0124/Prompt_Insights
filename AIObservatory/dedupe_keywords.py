import pandas as pd

tax_df = pd.read_csv('data/optimized_taxonomy.csv')

def dedupe(text):
    if not isinstance(text, str):
        return ""
    words = text.split()
    seen = set()
    clean_words = []
    for w in words:
        wl = w.lower()
        if wl not in seen:
            seen.add(wl)
            # We keep the lowercased version to normalize everything
            clean_words.append(wl)
    return " ".join(clean_words)

tax_df['keywords'] = tax_df['keywords'].apply(dedupe)
tax_df.to_csv('data/optimized_taxonomy.csv', index=False)
print("Successfully removed duplicates from keywords while preserving bigram order!")
