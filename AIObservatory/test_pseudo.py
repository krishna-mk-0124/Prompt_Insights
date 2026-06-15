import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestCentroid
from sklearn.metrics.pairwise import cosine_similarity

prompts = [
    "java error stacktrace",     # mapped to 0
    "null pointer exception",    # mapped to 0
    "scala indexing fast",       # mapped to 1 (Data)
    "bigquery sql data",         # mapped to 1
    
    "groovy exception",          # Unmapped (Other)
    "indexing scala scanned",    # Unmapped (Other)
]
labels = [0, 0, 1, 1, -1, -1]

vec = TfidfVectorizer()
X = vec.fit_transform(prompts)

X_mapped = X[:4]
y_mapped = labels[:4]
X_unmapped = X[4:]

clf = NearestCentroid()
clf.fit(X_mapped, y_mapped)

# Extract centroids and compare using cosine similarity
# centroids_ is a dense array, shape (n_classes, n_features)
sims = cosine_similarity(X_unmapped, clf.centroids_)
max_sims = sims.max(axis=1)
best_idx = sims.argmax(axis=1)

for i, unmap_idx in enumerate(range(4, 6)):
    print(f"Prompt: {prompts[unmap_idx]}")
    print(f"Matched Class: {clf.classes_[best_idx[i]]} with Sim: {max_sims[i]:.3f}")
