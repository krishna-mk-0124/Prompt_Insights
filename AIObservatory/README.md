# Enterprise Zero-Shot Prompt Categorizer

**Developer:** Achut Mahadev Kadam (krishna0124@gmail.com)

An ultra-secure, highly scalable, and mathematically deterministic pipeline for categorizing hundreds of thousands (or millions) of user prompts without relying on external Deep Learning models, LLMs, or expensive cloud GPUs.

## 🔒 The Enterprise Security & Cost Advantage

In highly regulated enterprise environments, sending proprietary company data or customer prompts to external Large Language Models (like OpenAI, Anthropic, etc.) is an absolute non-starter due to strict data privacy and leakage concerns. 

Furthermore, running local open-source LLMs requires significant infrastructure investments (expensive GPUs), complex security scrutiny, and extremely high operational costs.

This repository completely bypasses those limitations by utilizing **pure mathematics and linear algebra (Zero-Shot NLP techniques)** to achieve highly reliable categorization:
- **Zero Data Leakage:** The entire pipeline runs 100% locally on standard CPUs. It can be run on an air-gapped machine. No data ever leaves the enterprise.
- **Zero API Costs:** No third-party API calls, eliminating usage fees entirely.
- **Zero GPU Requirements:** Operates efficiently on standard hardware using highly optimized `scikit-learn` algorithms.
- **Zero Hallucination:** Because categorization is based strictly on mathematical Cosine Similarity between vector angles, the system is 100% deterministic and cannot "guess" or "hallucinate" false responses.

## ⚙️ How it Works (The Math)

This pipeline achieves state-of-the-art zero-shot categorization by combining predefined enterprise taxonomies with a dynamic fallback clustering algorithm.

1. **TF-IDF Vectorization:** It strips English sentences of useless "stop words" and converts the core keywords (and 2-word N-Grams) of both the user prompts and the official enterprise taxonomies into high-dimensional mathematical vectors.
2. **Cosine Similarity Routing:** It calculates the literal geometric angle between a user's prompt vector and the taxonomy vectors. If a prompt mathematically overlaps with an official enterprise category, it is permanently locked into that category.
3. **K-Means Auto-Discovery (The Safety Net):** If a prompt discusses a topic completely unrelated to the official enterprise taxonomy (a near-zero similarity score), the pipeline rejects it and isolates it into a fallback bucket. It then uses Truncated SVD and MiniBatch K-Means clustering to geometrically group these unknown prompts and auto-generate new subcategories based on their most frequent N-Grams.

## 📊 Pipeline Flowchart

```mermaid
graph TD
    A[Raw Prompt Data 500k+ rows] --> B[language_router.py]
    B --> |Identify Language| C(English Prompts)
    
    C --> D[discovery.py: Hybrid Zero-Shot Engine]
    E[taxonomy.csv: Official Categories] --> D
    
    D --> |Phase 1: TF-IDF Vectorization & Cosine Similarity| F{Is Max Similarity >= 0.05?}
    
    F -->|Yes: Mathematical Match| G[Assign to Official Taxonomy]
    F -->|No: Unrelated Topic| H[Route to 'Other/Miscellaneous' Bucket]
    
    H --> |Phase 2: Fallback Auto-Discovery| I[Truncated SVD Dimensionality Reduction]
    I --> J[MiniBatch KMeans Clustering]
    J --> K[Auto-Generate Subcategories via N-Grams]
    
    G --> L((Fully Categorized Dataset))
    K --> L
```

## 🚀 Quickstart

### 1. Prepare your Data
Place your raw text prompts (e.g., `prompt_sample.txt`) in the `data/` directory.
Ensure your official enterprise categories are defined in `data/taxonomy.csv`.

### 2. Filter Languages
```bash
python src/language_router.py
```
This isolates English prompts into `english_prompts.txt` and logs its progress dynamically.

### 3. Run the Categorizer
```bash
python src/discovery.py
```
This will mathematically map all prompts against the taxonomies. It outputs:
- `fully_categorized_dataset.csv`: Your millions of prompts securely assigned to categories.
- `hybrid_taxonomy_mapping.csv`: Your official enterprise taxonomies + any auto-discovered fallback subcategories.
