import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances

DATA_FILE = 'data/fully_categorized_dataset.csv'
GOLDEN_FILE = 'data/golden_set_review.csv'
TAX_FILE = 'data/optimized_taxonomy.csv'

def main():
    if not os.path.exists(DATA_FILE):
        print(f"Error: Could not find {DATA_FILE}. Please run discovery.py first.")
        return

    print("Loading mapped dataset...")
    df = pd.read_csv(DATA_FILE)
    
    # Remove 'Other' for validation purposes
    mapped_df = df[df['category_name'] != 'Other/Miscellaneous'].copy()
    print(f"Total mapped prompts to evaluate: {len(mapped_df):,}")
    
    if len(mapped_df) == 0:
        print("No mapped prompts found.")
        return

    # 1. Golden Set Extraction (Stratified Sampling)
    print("\n--- Phase 1: Generating Golden Set for Human Review ---")
    if not os.path.exists(GOLDEN_FILE):
        # Sample up to 5 prompts from each subcategory
        sampled = mapped_df.groupby('subcategory_name').apply(lambda x: x.sample(min(len(x), 5), random_state=42)).reset_index(drop=True)
        # Cap at 500 total
        if len(sampled) > 500:
            sampled = sampled.sample(500, random_state=42)
        
        # We only need prompt, category, subcategory, and a blank column for human to mark 1/0
        golden_df = sampled[['prompt', 'category_name', 'subcategory_name']].copy()
        golden_df['is_correct_mapping (1=Yes, 0=No)'] = ""
        
        golden_df.to_csv(GOLDEN_FILE, index=False)
        print(f"Created {GOLDEN_FILE} with {len(golden_df)} prompts.")
        print("ACTION REQUIRED: Please open this file, manually review the mappings, and mark 1 or 0 in the last column.")
    else:
        golden_df = pd.read_csv(GOLDEN_FILE)
        if golden_df['is_correct_mapping (1=Yes, 0=No)'].isnull().all():
            print(f"Found {GOLDEN_FILE}, but it has not been manually graded yet.")
        else:
            graded = golden_df.dropna(subset=['is_correct_mapping (1=Yes, 0=No)'])
            accuracy = graded['is_correct_mapping (1=Yes, 0=No)'].astype(int).mean()
            print(f"Golden Set Accuracy based on human review: {accuracy*100:.2f}% (over {len(graded)} reviewed prompts)")

    # 2. Mathematical Purity Check (Intra-Cluster Distance)
    print("\n--- Phase 2: Mathematical Purity Check ---")
    print("Computing TF-IDF to check for Semantic Drift...")
    vec = TfidfVectorizer(max_features=5000, stop_words='english')
    # Use a random sample of 20k to speed up math
    sample_for_math = mapped_df.sample(min(len(mapped_df), 20000), random_state=42)
    sample_for_math['text'] = sample_for_math['prompt'].astype(str)
    
    X = vec.fit_transform(sample_for_math['text'])
    
    # Calculate average distance from centroid for each subcategory
    from sklearn.neighbors import NearestCentroid
    clf = NearestCentroid()
    clf.fit(X, sample_for_math['subcategory_name'])
    
    centroids = clf.centroids_
    classes = clf.classes_
    
    # We warn if a category has a very high intra-cluster distance (meaning it's heavily diluted)
    distances = []
    for i, c in enumerate(classes):
        idx = (sample_for_math['subcategory_name'] == c).values
        if idx.sum() < 5:
            continue
        X_c = X[idx]
        centroid = centroids[i].reshape(1, -1)
        dist = cosine_distances(X_c, centroid).mean()
        distances.append((c, dist, idx.sum()))
    
    distances.sort(key=lambda x: x[1], reverse=True)
    print("\nTop 5 Most Mathematically Diluted Categories (High Distance = Bad):")
    for c, dist, count in distances[:5]:
        print(f"  - {c} (Distance: {dist:.3f}, Prompts in sample: {count})")
        if dist > 0.85:
            print("      ⚠️ WARNING: Semantic Drift detected in this category!")

    print("\nEvaluation complete.")

if __name__ == "__main__":
    main()
