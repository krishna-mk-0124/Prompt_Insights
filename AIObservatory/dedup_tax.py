import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

df = pd.read_csv('data/optimized_taxonomy.csv')
# Keep track of which rows are core
df['is_core'] = df.index < 59

def deduplicate_subcategories(group_df):
    group_df = group_df.reset_index(drop=True)
    
    # 1. Initialize keywords if not exists
    if 'keywords' not in group_df.columns:
        group_df['keywords'] = group_df['subcategory_name']
        
        # For core rows, generate a clean 2-3 word display name
        for idx, row in group_df.iterrows():
            if row['is_core']:
                words = str(row['subcategory_name']).split()
                clean_name = " ".join(words[:min(3, len(words))])
                group_df.at[idx, 'subcategory_name'] = clean_name

    working_rows = group_df.to_dict('records')
    
    while True:
        # Re-vectorize current working rows
        vectorizer = TfidfVectorizer(stop_words='english')
        try:
            tfidf = vectorizer.fit_transform([r['keywords'] for r in working_rows])
        except ValueError:
            break
            
        sim_matrix = cosine_similarity(tfidf)
        
        # 2. Phase 1: High-Confidence Anchor Merging (> 0.20)
        merged_in_phase_1 = False
        for i, row_i in enumerate(working_rows):
            if not row_i['is_core']: continue
            for j, row_j in enumerate(working_rows):
                if row_j['is_core'] or i == j: continue
                if sim_matrix[i, j] > 0.20:
                    # Merge auto row (j) into core row (i)
                    merged_keywords = set(str(row_i['keywords']).split() + str(row_j['keywords']).split())
                    working_rows[i]['keywords'] = " ".join(list(merged_keywords))
                    working_rows.pop(j)
                    merged_in_phase_1 = True
                    break
            if merged_in_phase_1: break
            
        if merged_in_phase_1:
            continue # Restart loop to re-calculate vectors after merging
            
        # 3. Phase 2: Iterative Compression (Max 20 Subcategories)
        if len(working_rows) <= 20:
            break # Successfully compressed to limit!
            
        # Find the absolute MOST similar pair where at least one is NOT core
        max_sim = -1.0
        best_pair = None
        for i in range(len(working_rows)):
            for j in range(i+1, len(working_rows)):
                # Rule: NEVER merge core + core
                if working_rows[i]['is_core'] and working_rows[j]['is_core']:
                    continue
                if sim_matrix[i, j] > max_sim:
                    max_sim = sim_matrix[i, j]
                    best_pair = (i, j)
                    
        if best_pair is None:
            break # No valid merge pairs left (highly unlikely)
            
        # Merge best_pair[1] into best_pair[0]
        i, j = best_pair
        # Ensure 'i' is the core row if one of them is core
        if working_rows[j]['is_core']:
            i, j = j, i
            
        merged_keywords = set(str(working_rows[i]['keywords']).split() + str(working_rows[j]['keywords']).split())
        working_rows[i]['keywords'] = " ".join(list(merged_keywords))
        
        # Keep the shortest/cleanest display name if both are auto
        if not working_rows[i]['is_core']:
            len_i = len(str(working_rows[i]['subcategory_name']).split())
            len_j = len(str(working_rows[j]['subcategory_name']).split())
            if len_j < len_i:
                working_rows[i]['subcategory_name'] = working_rows[j]['subcategory_name']
                
        working_rows.pop(j)
        
    return pd.DataFrame(working_rows)

final_dfs = []
for cat, group in df.groupby('category_name'):
    final_dfs.append(deduplicate_subcategories(group))
    
final_df = pd.concat(final_dfs, ignore_index=True)

# Sort so core rows appear first, then by category
final_df = final_df.sort_values(by=['is_core', 'category_id'], ascending=[False, True])

# Re-assign subcategory IDs to be sequential
final_df['subcategory_id'] = range(len(final_df))
final_df = final_df.drop(columns=['is_core'])

print(f"Original rows: {len(df)}")
print(f"Deduplicated rows: {len(final_df)}")
final_df.to_csv('data/optimized_taxonomy.csv', index=False)
print("Saved deduplicated taxonomy to data/optimized_taxonomy.csv")
