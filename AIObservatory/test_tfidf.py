import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

tax_docs = [
    "stacktrace exception null pointer segfault core dump badrequest conferma poid cdn fail issue",
    "java python c++ react angular node express api graphql git terraform software code script deploy",
    "sql table column etl nosql mongodb hadoop bigquery dax data row csv db wide weighted operator naming maps brackets",
]

prompts = [
    "groovy pendulum pydantic poid cdn conferma badrequest filled operator wide weighted error fail completely something else here to make it 50 words " * 3, # The spam log
    "how do I use the plus operator in java?", # A random prompt with "operator" and "java"
    "my cat is eating food and I love data very much but I don't know what to do about my car", # A random prompt with "data"
    "give me a python script to parse a csv", # A legit prompt
]

vectorizer = TfidfVectorizer(stop_words='english')
X_tax = vectorizer.fit_transform(tax_docs)
X_prompts = vectorizer.transform(prompts)

sim_matrix = cosine_similarity(X_prompts, X_tax)
max_sims = sim_matrix.max(axis=1)
best_tax_idx = sim_matrix.argmax(axis=1)

for i, p in enumerate(prompts):
    print(f"Prompt: {p[:30]}... -> Tax {best_tax_idx[i]} (Sim: {max_sims[i]:.3f})")
