import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import HDBSCAN

def clean_and_truncate(text):
    if pd.isna(text) or not isinstance(text, str):
        return None
    words = text.split()
    if len(words) < 8:
        return None
    if len(words) > 500:
        return " ".join(words[:250] + words[-250:])
    return text

def run_discovery():
    data_path = os.path.join("data", "prompt_sample")
    output_path = os.path.join("data", "discovered_topics.csv")
    
    print(f"Reading from {data_path}...")
    if not os.path.exists(data_path):
        print(f"File {data_path} not found. Ensure the file exists before running.")
        return

    with open(data_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(lines)} raw lines.")
    
    # Clean and truncate
    processed_lines = [clean_and_truncate(line) for line in lines]
    df = pd.DataFrame({"raw_text": lines, "processed_text": processed_lines})
    df = df.dropna(subset=["processed_text"]).reset_index(drop=True)
    
    print(f"Kept {len(df)} lines after cleaning.")
    
    if len(df) == 0:
        print("No data left after cleaning.")
        return

    # Feature extraction
    print("Extracting features...")
    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        sublinear_tf=True,
        max_features=20000
    )
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        sublinear_tf=True,
        max_features=20000
    )
    
    vectorizer = FeatureUnion([
        ("word", word_vectorizer),
        ("char", char_vectorizer)
    ])
    
    pipeline = Pipeline([
        ("features", vectorizer),
        ("svd", TruncatedSVD(n_components=30, random_state=42))
    ])
    
    X_reduced = pipeline.fit_transform(df["processed_text"])
    
    print("Clustering...")
    hdbscan = HDBSCAN(min_cluster_size=15, min_samples=5)
    labels = hdbscan.fit_predict(X_reduced)
    df["cluster"] = labels
    
    # Output clusters, extract top keywords
    print("Extracting top keywords per cluster...")
    cluster_results = []
    
    # We will use a separate TfidfVectorizer just on words to find keywords per cluster
    keyword_vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)
    
    for cluster_id in sorted(df["cluster"].unique()):
        if cluster_id == -1:
            continue # Skip noise
            
        cluster_data = df[df["cluster"] == cluster_id]
        combined_text = " ".join(cluster_data["processed_text"].tolist())
        
        # We need at least one document for vectorizer, but since it's combined, it's 1 document
        # Wait, tfidf across all clusters combined is better to penalize common words
        pass

    # Better approach for keywords: TF-IDF per cluster treating each cluster as a document
    # Group documents by cluster
    docs_per_cluster = df[df["cluster"] != -1].groupby("cluster")["processed_text"].apply(lambda x: " ".join(x)).reset_index()
    if not docs_per_cluster.empty:
        tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
        X_cluster_tfidf = tfidf.fit_transform(docs_per_cluster["processed_text"])
        feature_names = np.array(tfidf.get_feature_names_out())
        
        for i, row in docs_per_cluster.iterrows():
            cluster_id = row["cluster"]
            # Get top 10 keywords
            tfidf_scores = X_cluster_tfidf[i].toarray().flatten()
            top_10_idx = tfidf_scores.argsort()[-10:][::-1]
            top_10_keywords = [feature_names[idx] for idx in top_10_idx]
            
            sample_fragment = df[df["cluster"] == cluster_id]["processed_text"].iloc[0][:150] + "..."
            
            cluster_results.append({
                "cluster_id": cluster_id,
                "size": len(df[df["cluster"] == cluster_id]),
                "top_10_keywords": ", ".join(top_10_keywords),
                "sample_fragment": sample_fragment
            })
            
    results_df = pd.DataFrame(cluster_results)
    os.makedirs("data", exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"Done! Found {len(results_df)} dense clusters. Results saved to {output_path}")

if __name__ == "__main__":
    run_discovery()
