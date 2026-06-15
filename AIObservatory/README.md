# Enterprise Zero-Shot Prompt Categorizer

**Developer:** Achut Mahadev Kadam (krishna0124@gmail.com)

An ultra-secure, highly scalable, and mathematically deterministic **Intelligence-Based Machine Learning pipeline** for categorizing hundreds of thousands (or millions) of user prompts without relying on external Deep Learning models, LLMs, or expensive cloud GPUs.

## 🔒 The Enterprise Security & Cost Advantage

In highly regulated enterprise environments, sending proprietary company data or customer prompts to external Large Language Models (like OpenAI, Anthropic, etc.) is an absolute non-starter due to strict data privacy and leakage concerns. Furthermore, running local open-source LLMs requires massive infrastructure investments, complex security scrutiny, and extremely high operational costs.

This repository completely bypasses those limitations by utilizing **advanced Linear Algebra, NLP techniques, and Linear SVMs** to achieve highly reliable categorization:
- **Zero Data Leakage:** The entire pipeline runs 100% locally on standard CPUs. It can be run on an air-gapped machine. No data ever leaves the enterprise.
- **Zero API Costs:** No third-party API calls, eliminating usage fees entirely.
- **Zero GPU Requirements:** Operates efficiently on standard hardware using highly optimized `scikit-learn` algorithms.
- **Zero Hallucination:** Because categorization is based strictly on mathematical optimization and geometric angles, the system is highly deterministic and cannot "hallucinate" false responses.

---

## 🧠 Intelligence-Based ML vs. Naive Keyword Search

It is critical to understand that this pipeline is **not** a simple `CTRL+F` or binary keyword search engine (e.g., `IF "SQL" IN PROMPT THEN CATEGORY A`). Naive keyword searches are rigid, brittle, and incapable of understanding semantic relationships, resulting in massive accuracy degradation when users employ varied vocabulary.

Instead, this implementation is a **Self-Healing, Intelligence-Based Machine Learning Model**. It handles accuracy dynamically by transforming raw text into a multidimensional topological space. Rather than looking for boolean matches, the model calculates mathematical proximity, allowing it to dynamically identify patterns, weigh the importance of specific terms, and "learn" hidden contextual correlations (e.g., learning that the word "bucket" mathematically correlates to the concept of "Cloud Storage" even if the word "cloud" is completely absent).

---

## ⚙️ Core Algorithms & Mathematical Methods (In-Depth)

This pipeline achieves state-of-the-art accuracy by chaining together several sophisticated machine learning algorithms.

### 1. TF-IDF Vectorization (Term Frequency-Inverse Document Frequency)
Before any ML can occur, English sentences must be translated into numbers. The pipeline uses TF-IDF to map both user prompts and official enterprise taxonomies into a massive, 25,000-dimensional geometric space.
* **Research-Level Explanation:** TF-IDF mathematically penalizes the Term Frequency ($TF$) of a word by multiplying it by its Inverse Document Frequency ($IDF$). This dampens the "noise" of common English syntax while exponentially amplifying the mathematical weight of rare, highly specific identifiers.
* **The Example:** In the prompt *"how do I configure my kubernetes cluster"*, naive keyword search treats all 7 words equally. TF-IDF recognizes that "how", "do", and "I" appear millions of times across the dataset, assigning them near-zero mathematical weights. However, "kubernetes" and "cluster" are rare, giving them massive vector coefficients. The prompt is transformed from a sentence into a highly precise topological coordinate dictated almost entirely by its core technical subjects.

### 2. Zero-Shot Cosine Similarity Routing
Once the prompts exist as coordinates in 25,000-dimensional space, the algorithm must figure out which enterprise category they belong to without ever having been trained on historical data (Zero-Shot).
* **Research-Level Explanation:** The algorithm computes the normalized dot product (Cosine Similarity) between the vector of the user's prompt and the vectors of the 236 official enterprise subcategories. Instead of calculating Euclidean distance (which is skewed by prompt length), it calculates the literal geometric *angle* between the vectors.
* **The Example:** The official taxonomy for *"Database Administration"* sits at a specific multidimensional coordinate. A user types *"Help fixing my SQL select query logic"*. The Cosine algorithm calculates the physical angle between the user's vector and the taxonomy's vector. If the angle is incredibly tight (high similarity score) and passes the **Strict Overlap Mask** (requiring at least 2 overlapping conceptual N-Grams), the prompt is officially categorized. 

### 3. Stochastic Gradient Descent (SGD) Classifier & ElasticNet Penalty
Prompts that fail the Zero-Shot routing (due to messy vocabulary or spelling errors) are isolated into a fallback bucket. To rescue them, Phase 3 automatically trains an `SGDClassifier` using a linear Support Vector Machine (SVM) hinge loss.
* **Research-Level Explanation:** The SGD Classifier acts as an ML Rescue Sweep. It trains itself dynamically on the hundreds of thousands of prompts that *were* successfully categorized in Phase 2. Using an **ElasticNet penalty** (a hybrid of L1 Lasso and L2 Ridge regularization), the classifier aggressively zero-outs useless correlations (L1) while retaining groups of highly correlated features (L2).
* **The Example:** The Zero-Shot router categorized 10,000 prompts into the "Cloud Storage" category because they contained the word "S3". During this process, the SGD Classifier notices statistically that 80% of those prompts *also* contained the word "bucket". It mathematically links "bucket" to "Cloud Storage". Later, when it sweeps the fallback bucket, it finds a prompt that just says *"My bucket is full"*. Because of the learned ElasticNet correlation, the model intelligently rescues this prompt and drops it into "Cloud Storage", achieving semantic categorization without explicit keywords!

### 4. The Self-Healing Keyword Enrichment Loop
To ensure the pipeline gets smarter over time without human intervention, it employs a recursive enrichment algorithm.
* **Research-Level Explanation:** After the SGD Classifier finishes its epochs, the pipeline extracts the internal mathematical weight matrix (`clf.coef_`). It analyzes the coefficients for all 236 subcategories and rips out the top 15 mathematically heaviest N-Grams for each.
* **The Example:** The model discovers that the word "dataframe" has an unbelievably high coefficient weight for the "Python Engineering" category. The enrichment loop automatically extracts "dataframe", dedupes it, and permanently injects it back into the `optimized_taxonomy.csv` file. The next day, the Zero-Shot router will instantly recognize "dataframe" as a core taxonomic concept, permanently increasing the baseline accuracy of the entire pipeline.

---

## 🚀 Enterprise Pipeline Quickstart

For detailed flowcharts and architectural diagrams of the internal scripts, see [`ARCHITECTURE.md`](./ARCHITECTURE.md).

### 1. Data Ingestion & Language Filtering
Ensure your daily batch file (e.g., `cleaned_prompts_YYYY-MM-DD.txt`) is placed in the `data/` directory. Run the language router to strip out foreign languages and noisy unicode using our custom heuristic filters.
```bash
python src/language_router.py
```

### 2. The Machine Learning Engine (Zero-Shot & SGD)
Run the core Machine Learning cascade. This will automatically execute the TF-IDF Vectorization, Cosine Similarity matching, and the SGD Classifier Rescue Sweep.
```bash
python src/discovery.py
```
*Outputs: `data/fully_categorized_dataset.csv`*

### 3. Data Aggregation & Database Export
Instantly group the 300,000+ predictions down into 236 aggregated subcategory counts using Pandas, and securely bulk-insert the exact daily metrics directly into your Postgres Database for BI dashboarding.
```bash
python export_to_db.py "cleaned_prompts_2026-06-15.txt"
```

### 4. Keyword Enrichment (Run Periodically)
At the end of the week, run the self-healing enrichment loop to dynamically extract the heaviest ML weights and permanently upgrade your taxonomy.
```bash
python src/enrich_keywords.py
```
