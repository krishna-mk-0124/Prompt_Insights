# Prompt Insights: Architecture & Core Principles

This document provides a comprehensive, Confluence-style overview of the Prompt Insights machine learning pipeline. The core objective of this project is to accurately categorize 1.2 million historical user prompts into 236 highly specific, industry-standard subcategories **without relying on expensive, third-party Deep Learning APIs (like OpenAI or Claude).**

To achieve our strict 99% accuracy target on ultra-low-resource hardware (1 CPU core, <3GB RAM), the architecture utilizes a multi-phase cascade of heavily optimized Scikit-Learn algorithms, custom Unicode heuristic filters, and a self-healing Machine Learning keyword enrichment loop.

---

## Phase 1: Data Ingestion & Language Filtering

**Script:** `src/language_router.py`

Before mathematical categorization begins, the pipeline must strip out non-English prompts, as the taxonomy is designed for English jargon. 

### Mechanisms & Algorithms:
1.  **Fast Unicode Filter Heuristic:** 
    Running full language detection on 1.2 million rows is extremely slow. We bypass this by first calculating the ratio of non-ASCII characters (`ord(c) > 127`). If a prompt has >5 Unicode characters and the ratio exceeds 15%, it is overwhelmingly likely to be Chinese, Japanese, Korean, Arabic, or Russian. It is instantly routed out of the pipeline.
2.  **LRU Cached `langdetect`:**
    For the remaining prompts, we use the `langdetect` library. Because employee prompts often contain exact duplicates (e.g., "how to reset password"), we wrap the detector in an `@lru_cache(maxsize=100000)` to achieve an massive speedup by never recalculating the same string twice.
3.  **The Tech Rescue Override (Threshold <150 chars):**
    `langdetect` notoriously misclassifies short English code snippets (e.g., `select * from users`) as foreign languages like Italian or Romanian because they lack standard English grammar. If a prompt is under 150 characters, we cross-reference it against a hardcoded set of `TECH_WORDS` (`sql`, `query`, `error`, `python`). If a match is found, we forcefully override the algorithm and route it back into the English pipeline.

**Example:**
*   **Prompt:** `select * from server_logs`
*   **Action:** `langdetect` guesses Italian. The Tech Rescue override sees the word "server" and length <150, overriding it to `en`.

---

## Phase 2: The Zero-Shot Router (Cosine Similarity)

**Script:** `src/discovery.py`

This is the primary categorization engine. It attempts to map the English prompts to the 236 subcategories defined in `optimized_taxonomy.csv`.

### Mechanisms & Algorithms:
1.  **Dual TF-IDF Vectorization:**
    Instead of deep learning embeddings, we convert the prompts into massive mathematical sparse matrices using `TfidfVectorizer`. We run two in parallel:
    *   **Word N-Grams:** Looks at 1-word to 2-word combinations. (e.g., `"docker container"`). `max_features=50000`.
    *   **Character N-Grams (`analyzer='char_wb'`):** Looks at 3 to 5 letter chunks. This makes the algorithm virtually immune to typos. `max_features=40000`.
2.  **Cosine Similarity:**
    We vectorize both the Prompts and the Taxonomy Keywords, and calculate the Cosine Similarity dot-product. The prompt is assigned to the category with the highest mathematical score.
3.  **The Strict Overlap Mask (The Quality Gate):**
    A high cosine similarity score alone is dangerous (e.g., if a prompt is only 2 words long, a single matching word yields a falsely high 50% score). We use a `CountVectorizer` as an "Overlap Mask". A prompt is ONLY allowed to pass Phase 2 if it physically shares **at least 2 overlapping words** with the taxonomy category.

**Example:**
*   **Prompt:** `"I need to build a dockerized app for my sql database"`
*   **Action:** Phase 2 calculates the Word/Char TF-IDF score. The highest similarity is the "Databases" category. The Overlap Mask verifies that the prompt contains `sql` and `database` (2 words), which match the taxonomy. The prompt is officially categorized.

---

## Phase 3: The ML Rescue Sweep

**Script:** `src/discovery.py`

Prompts that fail the Strict Overlap Mask in Phase 2 are dumped into an "Other/Miscellaneous" bucket. Phase 3 acts as a supervised Machine Learning safety net to rescue these outliers.

### Mechanisms & Algorithms:
1.  **Dynamic Training Data:**
    The prompts that *successfully* passed Phase 2 are used as the "Ground Truth" training data.
2.  **SGD Classifier with ElasticNet Regularization:**
    We train a Logistic Regression model via Stochastic Gradient Descent (`SGDClassifier`). 
    *   **Penalty:** We use `penalty='elasticnet'` with `l1_ratio=0.15`. This is critical. ElasticNet mathematically shrinks the weights of "noise" words (like 'the', 'and', 'please') to exactly `0.0`, ensuring the model only pays attention to dense technical jargon.
    *   **One-vs-Rest:** Because there are 236 categories, the algorithm actually trains 236 separate binary models in C/Cython.
3.  **The Rescue Prediction:**
    Once trained on the Phase 2 hits, the SGD model looks at the "Other" outliers and predicts where they belong based on hidden word correlations that weren't in the official taxonomy.

**Example:**
*   **Outlier Prompt:** `"deploy k8s pod"`
*   **Action:** This fails Phase 2 because `k8s` is missing from the taxonomy keywords. However, the SGD Classifier learned during training that prompts containing `deploy` and `pod` heavily correlate with the "Cloud Infrastructure" category. It mathematically overrides the outlier and rescues it into Cloud Infrastructure.

---

## Phase 4: The Keyword Enrichment Loop

**Script:** `enrich_keywords.py`

If the SGD Classifier (Phase 3) is so smart at finding hidden correlations (like `k8s`), we should permanently add those discovered words back into `optimized_taxonomy.csv` so Phase 2 can catch them instantly next time!

### Mechanisms & Algorithms:
1.  **Extracting Feature Weights:**
    We feed the officially categorized dataset back into a fresh `SGDClassifier`. After training, we extract `clf.coef_` (the mathematical weight of every single word in the dictionary for every category).
2.  **Top 15 Extraction:**
    We sort the weights and extract the 15 heaviest, most undeniable keywords for each of the 236 categories.
3.  **Deduplication & Injection (`dedupe_keywords.py`):**
    We convert the existing taxonomy keywords and the newly discovered ML keywords into Python `sets`. This automatically deletes duplicate words. We then overwrite `optimized_taxonomy.csv` with the newly enriched data.

**Example:**
*   **Action:** The SGD Classifier notices that the word `coroutine` has an exceptionally high weight of `+5.82` for "Core Programming Languages". It extracts it and seamlessly injects it into the CSV file. Next month, any prompt containing `coroutine` will instantly pass Phase 2!

---

## Server Deployment & Resource Constraints

To run this pipeline on ultra-low-resource enterprise hardware (1 CPU core, <3GB RAM) without triggering an Out-of-Memory (OOM) crash, we implemented a Master Switch at the top of the codebase.

**The Global Resource Configuration Block:**
Found at the top of `enrich_keywords.py` and `src/discovery.py`:
```python
# ==========================================
# RESOURCE CONFIGURATION (Edit for Server)
# ==========================================
CPU_CORES = 1             # Throttles C-level multithreading to a single core
USE_FLOAT32 = True        # Compresses 64-bit sparse matrices, cutting RAM by 50%
MAX_FEATURES_LIMIT = 10000 # Hard-caps the dictionary size to prevent memory bloat
# ==========================================
```

**Silent Background Logging:**
Because `SGDClassifier` locks the Python GIL while crunching the massive matrices in C, standard python `print()` loops freeze. We utilize a daemonized `threading.Thread` to wake up every 60 seconds and print a clean `[Live Status] Fitting... 1.0 minutes elapsed` log, ensuring you can monitor server progress without clogging the terminal with `verbose=1` Cython dumps.
