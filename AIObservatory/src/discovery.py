import os
import sys
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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

def run_hybrid_discovery():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    input_file = os.path.join(data_dir, "english_prompts.txt")
    tax_file = os.path.join(data_dir, "taxonomy.csv")
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Run language_router.py first.")
        sys.exit(1)
    if not os.path.exists(tax_file):
        print(f"Error: {tax_file} not found.")
        sys.exit(1)
        
    print(f"Loading official taxonomy from {tax_file}...")
    tax_df = pd.read_csv(tax_file)
    tax_df["combined_desc"] = tax_df["category_name"] + " " + tax_df["subcategory_name"]
    
    # Preprocess taxonomy definitions
    tax_df["processed_desc"] = tax_df["combined_desc"].apply(preprocess_prompt)
    
    print(f"Loading texts from {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
        
    df = pd.DataFrame({"raw_text": lines})
    print(f"Loaded {len(df)} English prompts. Beginning Hybrid Mapping...")
    
    print("\n[Phase 1/4] Pre-processing User Prompts (Max 50 words)")
    df["truncated_prompt"] = df["raw_text"].apply(clean_and_truncate)
    
    processed_texts = []
    total = len(df)
    for i, text in enumerate(df["truncated_prompt"]):
        if i % 25000 == 0 and i > 0:
            print(f"  -> Processed {i:,} / {total:,} prompts ({(i/total)*100:.1f}%)")
        processed_texts.append(preprocess_prompt(text))
    df["processed_text"] = processed_texts
    
    print("\n[Phase 2/4] Zero-Shot Vectorization (TF-IDF)")
    # We fit TFIDF on the corpus containing BOTH the taxonomy and the user prompts
    vectorizer = TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=(1, 2))
    
    corpus = tax_df["processed_desc"].tolist() + df["processed_text"].tolist()
    tfidf_all = vectorizer.fit_transform(corpus)
    
    X_tax = tfidf_all[:len(tax_df)]
    X_prompts = tfidf_all[len(tax_df):]
    
    print("\n[Phase 3/4] Mathematical Routing (Cosine Similarity)")
    # Calculate similarity between all prompts and all 200 taxonomies
    print("  -> Computing cosine distances against the 200 official taxonomies...")
    sim_matrix = cosine_similarity(X_prompts, X_tax)
    
    max_sims = sim_matrix.max(axis=1)
    best_tax_idx = sim_matrix.argmax(axis=1)
    
    # Assign Official Categories
    df["category_id"] = -1
    df["category_name"] = ""
    df["subcategory_id"] = -1
    df["subcategory_name"] = ""
    
    # If the text shares virtually no words with ANY taxonomy (sim < 0.05), route to Other
    THRESHOLD = 0.05
    official_mask = max_sims >= THRESHOLD
    other_mask = max_sims < THRESHOLD
    
    official_count = official_mask.sum()
    other_count = other_mask.sum()
    print(f"  -> {official_count:,} prompts successfully matched an official taxonomy!")
    print(f"  -> {other_count:,} prompts had zero similarity and were routed to 'Other/Miscellaneous'.")
    
    # Map Official Prompts
    official_indices = np.where(official_mask)[0]
    matched_tax_rows = tax_df.iloc[best_tax_idx[official_indices]]
    
    df.loc[official_indices, "category_id"] = matched_tax_rows["category_id"].values
    df.loc[official_indices, "category_name"] = matched_tax_rows["category_name"].values
    df.loc[official_indices, "subcategory_id"] = matched_tax_rows["subcategory_id"].values
    df.loc[official_indices, "subcategory_name"] = matched_tax_rows["subcategory_name"].values
    
    hybrid_taxonomy_mapping = tax_df[["category_id", "category_name", "subcategory_id", "subcategory_name"]].to_dict('records')
    
    print("\n[Phase 4/4] Auto-Discovery for 'Other/Miscellaneous' Fallback Bucket")
    if other_count > 0:
        max_cat_id = tax_df["category_id"].max()
        max_sub_id = tax_df["subcategory_id"].max()
        
        other_cat_id = max_cat_id + 1
        
        df.loc[other_mask, "category_id"] = other_cat_id
        df.loc[other_mask, "category_name"] = "Other/Miscellaneous"
        
        other_indices = np.where(other_mask)[0]
        X_other = X_prompts[other_indices]
        
        # Reduce dimensionality to find geometric clusters for 'Other'
        svd = TruncatedSVD(n_components=min(150, other_count - 1), random_state=42)
        X_reduced_other = svd.fit_transform(X_other)
        
        n_clusters_other = min(20, other_count)
        print(f"  -> Slicing 'Other' into {n_clusters_other} auto-generated subcategories...")
        
        kmeans = MiniBatchKMeans(n_clusters=n_clusters_other, random_state=42, batch_size=min(10000, other_count), n_init='auto')
        other_labels = kmeans.fit_predict(X_reduced_other)
        
        other_names = extract_top_keywords(X_other, vectorizer, other_labels, n_clusters_other, top_n=2)
        
        for local_id in range(n_clusters_other):
            global_sub_id = max_sub_id + 1 + local_id
            sub_name = other_names[local_id]
            
            sub_mask = other_labels == local_id
            global_indices = other_indices[sub_mask]
            
            df.loc[global_indices, "subcategory_id"] = global_sub_id
            df.loc[global_indices, "subcategory_name"] = sub_name
            
            hybrid_taxonomy_mapping.append({
                "category_id": other_cat_id,
                "category_name": "Other/Miscellaneous",
                "subcategory_id": global_sub_id,
                "subcategory_name": sub_name
            })
            
    print("\n[Exporting Final Hybrid Automations]")
    taxonomy_df = pd.DataFrame(hybrid_taxonomy_mapping)
    tax_path = os.path.join(data_dir, "hybrid_taxonomy_mapping.csv")
    taxonomy_df.to_csv(tax_path, index=False)
    
    full_path = os.path.join(data_dir, "fully_categorized_dataset.csv")
    df[["raw_text", "truncated_prompt", "category_id", "category_name", "subcategory_id", "subcategory_name"]].to_csv(full_path, index=False)
    
    print(f"\nDone! Automatically matched / generated {len(taxonomy_df)} taxonomy combinations.")
    print(f"1. Hybrid Taxonomy Dictionary saved to {tax_path}")
    print(f"2. Fully Mapped Training Dataset saved to {full_path}")

if __name__ == "__main__":
    run_hybrid_discovery()
