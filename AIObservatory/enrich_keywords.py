import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import SGDClassifier
import re

print("Loading mapped dataset...")
df = pd.read_csv('data/fully_categorized_dataset.csv')
tax_df = pd.read_csv('data/optimized_taxonomy.csv')

# Only use officially categorized prompts
mapped_df = df[df["category_id"] != -1].copy()

print(f"Training SGD Classifier on {len(mapped_df)} mapped prompts...")
# OPTIMIZATION: Dropped max_features to 10k and forced float32 to cut RAM usage by 70%
vectorizer = TfidfVectorizer(max_features=10000, stop_words=list(ENGLISH_STOP_WORDS), ngram_range=(1, 2), sublinear_tf=True, dtype=np.float32)
X = vectorizer.fit_transform(mapped_df["raw_text"].fillna(""))
y = mapped_df["subcategory_id"].values

# OPTIMIZATION: n_jobs=1 prevents multiprocessing memory duplication
clf = SGDClassifier(loss='log_loss', penalty='elasticnet', l1_ratio=0.15, max_iter=1000, random_state=42, n_jobs=1)
clf.fit(X, y)

feature_names = np.array(vectorizer.get_feature_names_out())
print("Extracting top 15 mathematically predictive keywords for each subcategory...")

tax_df["keywords"] = tax_df["keywords"].fillna("")

for i, class_label in enumerate(clf.classes_):
    # Get the coefficients for this class
    coef = clf.coef_[i]
    # Sort and get top 20
    top_indices = coef.argsort()[-20:][::-1]
    top_features = feature_names[top_indices]
    
    # Filter out anything too short or non-alphabetical
    clean_features = []
    for feat in top_features:
        feat = feat.strip()
        if len(feat) > 3 and re.match(r'^[a-z ]+$', feat):
            clean_features.append(feat)
            if len(clean_features) == 15:
                break
                
    # Append to taxonomy
    if class_label in tax_df['subcategory_id'].values:
        idx = tax_df[tax_df['subcategory_id'] == class_label].index[0]
        existing_keywords = tax_df.at[idx, 'keywords']
        new_keywords_str = " ".join(clean_features)
        
        # Combine without duplicating
        combined = set(existing_keywords.split()) | set(new_keywords_str.split())
        tax_df.at[idx, 'keywords'] = " ".join(sorted(list(combined)))

tax_df.to_csv('data/optimized_taxonomy.csv', index=False)
print("Successfully injected ML-learned keywords into optimized_taxonomy.csv!")
