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

    vectorizer = TfidfVectorizer(stop_words='english')
    try:
        # Vectorize against the keywords, not the display name!
        tfidf = vectorizer.fit_transform(group_df['keywords'])
    except ValueError:
        return group_df
        
    sim_matrix = cosine_similarity(tfidf)
    
    core_indices = group_df[group_df['is_core']].index.tolist()
    auto_indices = group_df[~group_df['is_core']].index.tolist()
    
    # If no core rows exist in this category, we just use the auto rows as anchors
    if not core_indices:
        core_indices = [auto_indices[0]] if auto_indices else []
        auto_indices = auto_indices[1:] if len(auto_indices) > 1 else []
        
    final_rows = []
    merged_auto_indices = set()
    
    # Anchor Clustering: Match auto rows directly to core anchors
    for core_idx in core_indices:
        core_row = group_df.iloc[core_idx]
        current_keywords = set(str(core_row['keywords']).split())
        
        for auto_idx in auto_indices:
            if auto_idx in merged_auto_indices:
                continue
                
            # If the auto row matches this specific core row strongly
            if sim_matrix[core_idx, auto_idx] > 0.20:
                auto_words = str(group_df.iloc[auto_idx]['keywords']).split()
                current_keywords.update(auto_words)
                merged_auto_indices.add(auto_idx)
                
        final_rows.append({
            'category_id': core_row['category_id'],
            'category_name': core_row['category_name'],
            'subcategory_id': core_row['subcategory_id'],
            'subcategory_name': core_row['subcategory_name'],
            'keywords': " ".join(list(current_keywords)),
            'is_core': True
        })
        
    # Any auto rows that didn't match a core anchor survive as their own subcategories
    for auto_idx in auto_indices:
        if auto_idx not in merged_auto_indices:
            auto_row = group_df.iloc[auto_idx]
            final_rows.append({
                'category_id': auto_row['category_id'],
                'category_name': auto_row['category_name'],
                'subcategory_id': auto_row['subcategory_id'],
                'subcategory_name': auto_row['subcategory_name'],
                'keywords': auto_row['keywords'],
                'is_core': False
            })
            
    return pd.DataFrame(final_rows)

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
