import os
import sys
import pandas as pd
import numpy as np
import math
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
    tax_file = os.path.join(data_dir, "optimized_taxonomy.csv")
    
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
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    
    # Block common noise tokens that hijack the unsupervised clusters
    custom_noise = [
        'st', 'oo', 'hi', 'hello', 'hey', 'test', 'com', 'www', 'http', 'https', 'nd', 'rd', 'th', 'pls', 'please',
        'strategy', 'optimization', 'guidance', 'support', 'clarification', 'review', 'management', 'planning', 'improvement', 'architecture', 'design',
        'using', 'run', 'need', 'new', 'make', 'better', 'want', 'just', 'add', 'total', 'et', 'does', 'know', 'let', 'like', 'looks', 'based', 'level', 'high', 'share', 'id', 'ore', 'doing', 'hope', 'think', 'don', 'provide', 'questions', 'ask', 'check', 'good', 'morning', 'getting', 'include', 'sure', 'yes', 'work', 'al', 'dl', 'che', 'copy', 'paste', 'tell',
        'dont', 'remove', 'working', 'file', 'files', 'use', 'cases', 'que', 'el', 'la', 'en', 'thank', 'understand', 'able', 'different', 'cm', 'dw', 'excel', 'sheet', 'information', 'additional', 'days', 'day', 'say', 'way', 'dl', 'il', 'create', 'folder', 'uk', 'india', 'added', 'details', 'report', 'status', 'updated', 'update', 'plan', 'ot', 'ey', 'ok', 'lets', 'field', 'months', 'send', 'set', 'se', 'vs', 'naa', 'prj', 'au', 'variable', 'odl', 'agent', 'match', 'jan', 'mar', 'pm', 'quick', 'clever', 'regarding', 'clic', 'case', 'customer', 'engagement', 'works', 'non',
        'change', 'list', 'question', 'rate', 'count', 'key', 'company', 'sentence', 'instead', 'process', 'sent', 'today', 'local', 'members', 'para', 'los', 'experience', 'following', 'note', 'job', 'right', 'left', 'things', 'come', 'did', 'look', 'forward', 'lo', 'es', 've', 'request', 'kindly', 'teams', 'got', 'ol', 'py', 'line', 'feel', 'free', 'week', 'bit', 'sound', 'llc', 'ca', 'di', 'le', 'issues', 'identified', 'april', 'march', 'input', 'output', 'coming', 'tool', 'search', 'view', 'tomorrow', 'canada', 'united', 'discuss', 'color', 'blue', 'tools', 'die', 'und', 'em', 'session',
        'actually', 'means', 'clear', 'ar', 'active', 'used', 'values', 'section', 'open', 'word', 'por', 'una', 'base', 'years', 'll', 'mail', 'attached', 'saying', 'possible', 'size', 'template', 'place', 'instructions', 'thing', 'rewrite', 'issue', 'description', 'updates', 'okay', 'yeah', 'feedback', 'reflect', 'happy', 'friday', 'generate', 'sum', 'going', 'je', 'est', 'apr', 'feb', 'needs', 'numbers', 'number', 'suggest', 'similar', 'answer', 'answers', 'start', 'compare', 'bullets', 'multiple', 'asking', 'method', 'uploaded', 'lead', 'favor', 'factual', 'studies',
        'point', 'wait', 'min', 'example', 'people', 'little', 'tech', 'del', 'si', 'present', 'current', 'state', 'im', 'wanted', 'connect', 'currently', 'related', 'product', 'reason', 'ready', 'person', 'showing', 'variables', 'sense', 'hours', 'na', 'reply', 'great', 'weekend', 'really', 'appreciate', 'codes', 'bucket', 'dec', 'given', 'available', 'target', 'gb', 'oy', 'tr', 'convert', 'previous', 'mi', 'colleague', 'colleagues', 'launch', 'visual', 'background', 'options', 'june', 'july', 'maybe', 'phase', 'old', 'mentioned',
        'context', 'fields', 'kind', 'sentences', 'tier', 'bt', 'ip', 'mention', 'looking', 'try', 'rating', 'far', 'called', 'las', 'como', 'replace', 'said', 'lot', 'decision', 'records', 'task', 'approach', 'aligned', 'ensure', 'discussion', 'confirm', 'chat', 'coffee', 'lines', 'ich', 'complete', 'progress', 'specific', 'discussed', 'changed', 'sr', 'esta', 'final', 'version', 'default', 'apply', 'pdf', 'viewer', 'asked', 'comment', 'sounds', 'mac', 'city', 'names', 'header', 'sorry', 'delay', 'exact', 'layout', 'taking', 'exercise', 'confirmation',
        'write', 'submit', 'date', 'om', 'hai', 'second', 'completed', 'da', 'doesn', 'dates', 'ones', 'gave', 'average', 'idea', 'este', 'su', 'problem', 'break', 'received', 'talk', 'chatgpt', 'home', 'monday', 'identify', 'past', 'split', 'original', 'category', 'giving', 'rows', 'doc', 'te', 'pero', 'improve', 'box', 'focus', 'areas', 'les', 'tu', 'latest', 'needed', 'reference', 'provided', 'continue', 'frame', 'conversation', 'starters', 'tab', 'window', 'aug', 'oct', 'screen', 'shot', 'calls', 'happens', 'believe', 'fy', 'seeing', 'captions',
        'tables', 'space', 'type', 'missing', 'title', 'slide', 'ya', 'quiero', 'respond', 'doesnt', 'starting', 'exactly', 'having', 'whats', 'trying', 'car', 'lower', 'office', 'blank', 'ist', 'ho', 'role', 'evidence', 'created', 'exec', 'build', 'scenario', 'room', 'overall', 'vibes', 'results', 'pany', 'eer', 'concise', 'filter', 'ap', 'customers', 'bloc', 'round', 'thinking', 'align', 'scope', 'necesito', 'respuesta', 'record', 'insert', 'draw', 'um', 'os', 'max', 'avg', 'declare', 'interval', 'emails', 'abp', 'limited', 'classification', 'insights', 'graphs',
        'detailed', 'ppt', 'column', 'users', 'documents', 'examples', 'period', 'sending', 'bio', 'directly', 'sl', 'feels', 'white', 'reports', 'meant', 'says', 'hotels', 'client', 'thats', 'mind', 'rules', 'stage', 'shorten', 'friendly', 'comments', 'greatest', 'opportunity', 'requirement', 'meet', 'profile', 'profiles', 'times', 'sign', 'transactions', 'activity', 'implement', 'beginning', 'didn', 'reword', 'benefit', 'reduce', 'cell', 'recent', 'save', 'button', 'parent', 'child', 'meeting', 'low', 'medium', 'mo', 'ct', 'yr', 'projects', 'mindset', 'pages', 'btn', 'happened', 'store', 'element', 'describes',
        'headline', 'crisp', 'simpler', 'ja', 'items', 'adding', 'sort', 'map', 'responses', 'meaning', 'happen', 'upload', 'single', 'hola', 'esto', 'liner', 'soon', 'future', 'approved', 'properly', 'weeks', 'monthly', 'drop', 'removed', 'axp', 'aa', 'isn', 'accordingly', 'guess', 'differences', 'solve', 'du', 'pour', 'pa', 'strong', 'refresh', 'making', 'changes', 'sub', 'dim', 'cuenta', 'ser', 'self', 'site', 'reach', 'nov', 'sep', 'rtf', 'cht', 'thursday', 'wednesday', 'coll', 'direct', 'usage', 'io', 'vertx', 'cg', 'label', 'intro', 'walked',
        'elaborate', 'age', 'metrics', 'existing', 'wrong', 'shorter', 'live', 'happening', 'team', 'rephrase', 'sharing', 'main', 'mas', 'solo', 'nice', 'love', 'products', 'ingles', 'bien', 'ir', 'setting', 'mins', 'positive', 'stay', 'statements', 'features', 'english', 'history', 'cust', 'solution', 'cfrcpi', 'turn', 'suggestions', 'hold', 'revise', 'click', 'fecha', 'prepare', 'basic', 'talking', 'news', 'ad', 'hoc', 'txt', 'str', 'cd', 'wont', 'cc', 'ed', 'ni', 'london', 'expand', 'acronyms', 'review', 'check', 'make', 'sure', 'reviewing', 'update', 'status', 'tell', 'help', 'know', 'needs',
        'combine', 'including', 'channel', 'makes', 'analyze', 'platform', 'separate', 'won', 'easier', 'scheduled', 'till', 'print', 'segment', 'band', 'markets', 'double', 'outside', 'iy', 'ee', 'explanation', 'modify', 'setup', 'mm', 'percentage', 'application', 'couple', 'sa', 'jun', 'jul', 'export', 'written', 'des', 'das', 'understanding', 'empathetic', 'book', 'dev', 'cust_xref_id', 'net', 'agree', 'alignment', 'cross', 'benefits', 'experiences', 'relevant', 'roles', 'reviewed', 'trim', 'common', 'timeline', 'yesterday', 'rs', 'combined', 'theme', 'requests', 'moving', 'quickly', 'screenshot', 'validating', 'push', 'red', 'proceed',
        'early', 'avoid', 'alternate', 'implemented', 'subtitle', 'interesting', 'axis', 'speak', 'pattern', 'narrative', 'equipo', 'correo', 'party', 'ui', 'hour', 'mapping', 'challenge', 'included', 'queries', 'tabs', 'extra', 'redo', 'bad', 'success', 'weekly', 'creating', 'processing', 'ma', 'ha', 'unique', 'validate', 'desc', 'ke', 'ki', 'ce', 'une', 'fully', 'cap', 'applicable', 'checks', 'engineer', 'maximum', 'rates', 'personal', 'track', 'fast', 'equal', 'greater', 'important', 'fah', 'stuff', 'moved', 'gto',
        'ent', 'verticles', 'reaching', 'tuesday', 'clean', 'style', 'characters', 'stop', 'leaders', 'worked', 'character', 'dataset', 'nos', 'hacer', 'away', 'counts', 'email', 'chain', 'editable', 'me', 'failed', 'org', 'comes', 'normal', 'contact', 'pending', 'bu', 'ids', 'columns', 'ucc', 'initial', 'categories', 'order', 'migration', 'ft', 'sobre', 'todo', 'recommendation', 'delta', 'submitted', 'previously', 'assumptions', 'cfr', 'gap', 'guardian', 'gonna', 'demo', 'fit', 'purpose', 'innovative', 'pcm', 'rule', 'kpa', 'defensible', 'redactar',
        'brief', 'highlight', 'overview', 'acct_', 'unrecognized', 'exclusion', 'mismatch', 'analyse', 'didnt', 'gracias', 'son', 'refer', 'helpful', 'tracker', 'thought', 'closed', 'sections', 'area', 'dd', 'time', 'risks', 'charts', 'generic', 'tktg', 'told', 'fun', 'icons', 'world', 'earlier', 'polish', 'exist', 'duplicates', 'pro', 'usa', 'bring', 'skills', 'download', 'quote', 'concerns', 'aware', 'basis', 'daily', 'apart', 'initiative', 'rasc', 'started', 'register', 'higher', 'driven', 'photo', 'circle', 'arrow', 'arrows', 'dash', 'british',
        'downloadable', 'ways', 'casual', 'calculation', 'later', 'cover', 'went', 'cards', 'porque', 'ideas', 'repo', 'matching', 'mst', 'partners', 'drive', 'adjust', 'highlighted', 'assigned', 'prior', 'availability', 'wise', 'random', 'middle', 'straight', 'lol', 'deep', 'summary', 'rank', 'servers', 'cms', 'january', 'mr', 'flag', 'complex', 'differently', 'cleaner', 'pie', 'dashes', 'floor', 'cool', 'birthday', 'appear', 'appropriate', 'humor', 'dot', 'ety', 'challenges', 'pti', 'puedes', 'ayudar', 'supported', 'ics', 'plus', 'sur',
        'dy', 'wanting', 'seperate', 'wordy', 'gl', 'shows', 'suggested', 'duplicate', 'remember', 'hard', 'submission', 'opportunities', 'md', 'perfect', 'confused', 'comma', 'sy', 'lt', 'attach', 'funny', 'places', 'catch', 'ores', 'super', 'signature', 'mismatches', 'potential', 'clarify', 'ub', 'repeating', 'der', 'wir', 'primary', 'industry', 'timelines', 'hear', 'hay', 'algo', 'helps', 'telling', 'tenure', 'renewal', 'necessary', 'spacing', 'airport', 'quotes', 'require', 'ic', 'var', 'ipc', 'planned', 'basically', 'boss', 'chnage'
    ]
    extended_stop_words = list(ENGLISH_STOP_WORDS) + custom_noise
    
    # We fit TFIDF on the corpus containing BOTH the taxonomy and the user prompts
    vectorizer = TfidfVectorizer(max_features=10000, stop_words=extended_stop_words, ngram_range=(1, 2))
    
    corpus = tax_df["processed_desc"].tolist() + df["processed_text"].tolist()
    tfidf_all = vectorizer.fit_transform(corpus)
    
    X_tax = tfidf_all[:len(tax_df)]
    X_prompts = tfidf_all[len(tax_df):]
    
    print("\n[Phase 3/4] Mathematical Routing (Cosine Similarity)")
    print("  -> Computing cosine distances against the official taxonomies...")
    sim_matrix = cosine_similarity(X_prompts, X_tax)
    
    max_sims = sim_matrix.max(axis=1)
    best_tax_idx = sim_matrix.argmax(axis=1)
    
    # Assign Official Categories
    df["category_id"] = -1
    df["category_name"] = ""
    df["subcategory_id"] = -1
    df["subcategory_name"] = ""
    
    # If the text shares virtually no words with ANY taxonomy, route to Other
    THRESHOLD = 0.08
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
    
    print("\n[Phase 4/4] Recursive Auto-Discovery for 'Other/Miscellaneous' Fallback Bucket")
    if other_count > 0:
        MAX_CLUSTER_SIZE = 10000
        
        max_cat_id = tax_df["category_id"].max()
        max_sub_id = tax_df["subcategory_id"].max()
        
        other_cat_id = max_cat_id + 1
        
        df.loc[other_mask, "category_id"] = other_cat_id
        df.loc[other_mask, "category_name"] = "Other/Miscellaneous"
        
        other_indices = np.where(other_mask)[0]
        X_other = X_prompts[other_indices]
        
        # Base Dimensionality Reduction
        svd = TruncatedSVD(n_components=min(150, other_count - 1), random_state=42)
        X_reduced_other = svd.fit_transform(X_other)
        
        n_clusters_other = min(20, other_count)
        print(f"  -> Step 1: Initial grouping into {n_clusters_other} baseline clusters...")
        
        kmeans = MiniBatchKMeans(n_clusters=n_clusters_other, random_state=42, batch_size=min(10000, other_count), n_init='auto')
        base_other_labels = kmeans.fit_predict(X_reduced_other)
        
        current_global_sub_id = max_sub_id + 1
        
        for local_id in range(n_clusters_other):
            sub_mask = base_other_labels == local_id
            cluster_size = sub_mask.sum()
            if cluster_size == 0:
                continue
                
            cluster_X = X_other[sub_mask]
            cluster_global_indices = other_indices[sub_mask]
            
            if cluster_size > MAX_CLUSTER_SIZE:
                # Shatter the mega-cluster
                shatter_count = min(math.ceil(cluster_size / MAX_CLUSTER_SIZE), cluster_size)
                print(f"    -> Shattering Mega-Cluster {local_id} ({cluster_size:,} prompts) into {shatter_count} micro-clusters...")
                
                # Reduce dims specifically for this dense mega-cluster
                svd_micro = TruncatedSVD(n_components=min(150, cluster_size - 1), random_state=42)
                X_reduced_micro = svd_micro.fit_transform(cluster_X)
                
                kmeans_micro = MiniBatchKMeans(n_clusters=shatter_count, random_state=42, batch_size=min(10000, cluster_size), n_init='auto')
                micro_labels = kmeans_micro.fit_predict(X_reduced_micro)
                micro_names = extract_top_keywords(cluster_X, vectorizer, micro_labels, shatter_count, top_n=2)
                
                for m_id in range(shatter_count):
                    m_mask = micro_labels == m_id
                    if m_mask.sum() == 0:
                        continue
                        
                    m_global_indices = cluster_global_indices[m_mask]
                    
                    df.loc[m_global_indices, "subcategory_id"] = current_global_sub_id
                    df.loc[m_global_indices, "subcategory_name"] = micro_names[m_id]
                    
                    hybrid_taxonomy_mapping.append({
                        "category_id": other_cat_id,
                        "category_name": "Other/Miscellaneous",
                        "subcategory_id": current_global_sub_id,
                        "subcategory_name": micro_names[m_id]
                    })
                    current_global_sub_id += 1
            else:
                # Normal cluster
                base_names = extract_top_keywords(cluster_X, vectorizer, np.zeros(cluster_size, dtype=int), 1, top_n=2)
                name = base_names[0]
                
                df.loc[cluster_global_indices, "subcategory_id"] = current_global_sub_id
                df.loc[cluster_global_indices, "subcategory_name"] = name
                
                hybrid_taxonomy_mapping.append({
                    "category_id": other_cat_id,
                    "category_name": "Other/Miscellaneous",
                    "subcategory_id": current_global_sub_id,
                    "subcategory_name": name
                })
                current_global_sub_id += 1
                
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
