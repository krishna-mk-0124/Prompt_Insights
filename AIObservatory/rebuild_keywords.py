import pandas as pd
import re

TAX_FILE = 'data/optimized_taxonomy.csv'

def clean_text(text):
    # Remove special characters and lowercase
    text = re.sub(r'[^a-zA-Z\s]', ' ', str(text).lower())
    # Remove common stopwords
    stopwords = {'and', 'or', 'the', 'a', 'an', 'in', 'on', 'for', 'to', 'of', 'with', 'by', 'at', 'about'}
    words = [w for w in text.split() if w not in stopwords]
    # Remove duplicates while preserving order
    return list(dict.fromkeys(words))

def rebuild_pure_keywords():
    df = pd.read_csv(TAX_FILE)
    
    pure_keywords_list = []
    
    for index, row in df.iterrows():
        cat_words = clean_text(row['category_name'])
        subcat_words = clean_text(row['subcategory_name'])
        
        # Combine words from category and subcategory to create ultra-pure anchors
        combined_words = cat_words + subcat_words
        
        # Remove duplicates
        pure_anchors = list(dict.fromkeys(combined_words))
        
        # Join into string
        pure_keywords_list.append(" ".join(pure_anchors))
        
    df['keywords'] = pure_keywords_list
    
    df.to_csv(TAX_FILE, index=False)
    print("Successfully purged corrupted keywords and rebuilt ultra-pure semantic anchors for all 236 subcategories.")

if __name__ == "__main__":
    rebuild_pure_keywords()
