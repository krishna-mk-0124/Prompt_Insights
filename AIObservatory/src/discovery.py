import os
import sys
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import MiniBatchKMeans
from .preprocess import preprocess_prompt

def clean_and_truncate(text):
    text = str(text)
    words = text.split()
    if len(words) > 50:
        return " ".join(words[:50])
    return text

def extract_top_keywords(tfidf_matrix, vectorizer, cluster_labels, num_clusters, top_n=2):
    feature_names = np.array(vectorizer.get_feature_names_out())
    cluster_names = {}
    for i in range(num_clusters):
        cluster_indices = np.where(cluster_labels == i)[0]
        if len(cluster_indices) == 0:
            cluster_names[i] = "empty_cluster"
            continue
        cluster_tfidf = tfidf_matrix[cluster_indices]
        mean_tfidf = np.asarray(cluster_tfidf.mean(axis=0)).flatten()
        top_indices = mean_tfidf.argsort()[-top_n:][::-1]
        cluster_names[i] = "_".join(feature_names[top_indices])
    return cluster_names

def run_discovery():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    input_file = os.path.join(data_dir, "english_prompts.txt")
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Run language_router.py first.")
        sys.exit(1)
        
    print(f"Loading texts from {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
        
    df = pd.DataFrame({"raw_text": lines})
    print(f"Loaded {len(df)} English prompts. Beginning Auto-Hierarchical mapping...")
    
    print("\n[Phase 1/4] Pre-processing & Truncating (Max 50 words)")
    df["truncated_prompt"] = df["raw_text"].apply(clean_and_truncate)
    
    processed_texts = []
    total = len(df)
    for i, text in enumerate(df["truncated_prompt"]):
        if i % 25000 == 0 and i > 0:
            print(f"  -> Processed {i:,} / {total:,} prompts ({(i/total)*100:.1f}%)")
        processed_texts.append(preprocess_prompt(text))
    df["processed_text"] = processed_texts
    
    print("\n[Phase 2/4] Vectorization (TF-IDF)")
    vectorizer = TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(df["processed_text"])
    
    print("\n[Phase 3/4] Dimensionality Reduction (SVD)")
    svd = TruncatedSVD(n_components=150, random_state=42)
    X_reduced = svd.fit_transform(tfidf_matrix)
    
    print("\n[Phase 4/4] Level 1 Clustering: 10 Main Categories")
    kmeans_main = MiniBatchKMeans(n_clusters=10, random_state=42, batch_size=10000, n_init='auto')
    main_labels = kmeans_main.fit_predict(X_reduced)
    df["category_id"] = main_labels
    
    main_category_names = extract_top_keywords(tfidf_matrix, vectorizer, main_labels, 10, top_n=2)
    df["auto_category_name"] = df["category_id"].map(main_category_names)
    
    print("\n[Phase 4/4] Level 2 Clustering: 20 Subcategories per Main Category")
    taxonomy_mapping = []
    df["subcategory_id"] = -1
    df["auto_subcategory_name"] = ""
    
    for cat_id in range(10):
        cat_name = main_category_names[cat_id]
        print(f"  -> Slicing Main Category {cat_id} ({cat_name}) into 20 subcategories...")
        cat_mask = df["category_id"] == cat_id
        if not cat_mask.any(): continue
        
        cat_indices = np.where(cat_mask)[0]
        cat_X_reduced = X_reduced[cat_indices]
        cat_tfidf = tfidf_matrix[cat_indices]
        
        n_clusters_sub = min(20, len(cat_indices))
        kmeans_sub = MiniBatchKMeans(n_clusters=n_clusters_sub, random_state=42, batch_size=min(10000, len(cat_indices)), n_init='auto')
        sub_labels = kmeans_sub.fit_predict(cat_X_reduced)
        
        sub_names = extract_top_keywords(cat_tfidf, vectorizer, sub_labels, n_clusters_sub, top_n=2)
        
        for local_sub_id in range(n_clusters_sub):
            global_sub_id = (cat_id * 20) + local_sub_id
            
            sub_mask = sub_labels == local_sub_id
            global_indices = cat_indices[sub_mask]
            
            df.loc[global_indices, "subcategory_id"] = global_sub_id
            df.loc[global_indices, "auto_subcategory_name"] = sub_names[local_sub_id]
            
            taxonomy_mapping.append({
                "category_id": cat_id,
                "auto_category_name": cat_name,
                "subcategory_id": global_sub_id,
                "auto_subcategory_name": sub_names[local_sub_id]
            })
            
    print("\n[Exporting Automations]")
    taxonomy_df = pd.DataFrame(taxonomy_mapping)
    tax_path = os.path.join(data_dir, "taxonomy_mapping.csv")
    taxonomy_df.to_csv(tax_path, index=False)
    
    full_path = os.path.join(data_dir, "fully_automated_dataset.csv")
    df[["raw_text", "truncated_prompt", "category_id", "auto_category_name", "subcategory_id", "auto_subcategory_name"]].to_csv(full_path, index=False)
    
    print(f"\nDone! Automatically generated {len(taxonomy_df)} taxonomy combinations.")
    print(f"1. Taxonomy Mapping saved to {tax_path}")
    print(f"2. Fully Labeled Dataset saved to {full_path}")

if __name__ == "__main__":
    run_discovery()
