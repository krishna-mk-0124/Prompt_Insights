import os
import time
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
import joblib

def run_training():
    data_path = os.path.join("data", "labeled_sample.csv")
    model_dir = "models"
    model_path = os.path.join(model_dir, "intent_pipeline.pkl")
    
    print(f"Looking for labeled data at {data_path}...")
    
    # Wait or assume existence
    max_wait_time = 10 # Only waiting 10s for demo purposes if not found immediately
    waited = 0
    while not os.path.exists(data_path):
        print(f"File not found. Waiting... ({waited}/{max_wait_time}s)")
        time.sleep(2)
        waited += 2
        if waited >= max_wait_time:
            print(f"Data not found after {max_wait_time}s. Please ensure {data_path} is created. Exiting.")
            return

    print("Data found! Loading...")
    df = pd.read_csv(data_path)
    
    if "truncated_prompt" not in df.columns or "subcategory_id" not in df.columns:
        print("Required columns ('truncated_prompt', 'subcategory_id') are missing from the data.")
        return

    from .preprocess import preprocess_prompt
    print("Applying preprocessing (noise reduction & normalization) to training data...")
    
    processed_x = []
    total = len(df)
    for i, text in enumerate(df["truncated_prompt"].fillna("")):
        if i % 10000 == 0 and i > 0:
            print(f"  -> Preprocessed {i:,} / {total:,} training prompts ({(i/total)*100:.1f}%)")
        processed_x.append(preprocess_prompt(text))
    print(f"  -> Preprocessed {total:,} / {total:,} training prompts (100.0%)")
    
    X = pd.Series(processed_x)
    y = df["subcategory_id"]
    
    print(f"Loaded {len(df)} labeled samples. Building pipeline...")
    
    # TfidfVectorizer: word analyzer, ngram_range=(1,3), max_features=80000, sublinear_tf=True
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 3),
        max_features=80000,
        sublinear_tf=True
    )
    
    # CalibratedClassifierCV wrapping LinearSVC(C=0.5, class_weight='balanced')
    base_estimator = LinearSVC(C=0.5, class_weight="balanced", dual=False)
    clf = CalibratedClassifierCV(base_estimator, cv=3)
    
    pipeline = Pipeline([
        ("tfidf", vectorizer),
        ("classifier", clf)
    ])
    
    print("Training calibrated pipeline...")
    pipeline.fit(X, y)
    
    print("Training complete. Serializing model...")
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(pipeline, model_path, compress=3)
    
    print(f"Pipeline successfully saved to {model_path}.")

if __name__ == "__main__":
    run_training()
