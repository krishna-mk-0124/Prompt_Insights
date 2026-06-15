import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

tax_docs = [
    "stacktrace exception null pointer segfault core dump badrequest conferma poid cdn fail issue"
]

prompt = "groovy pendulum pydantic poid cdn conferma badrequest filled operator wide weighted error completely something else here to make it 50 words " * 3

class_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X_tax = class_vectorizer.fit_transform(tax_docs)
X_prompts_class = class_vectorizer.transform([prompt])

sim_matrix = cosine_similarity(X_prompts_class, X_tax)
print(f"Cosine Similarity: {sim_matrix[0][0]}")
