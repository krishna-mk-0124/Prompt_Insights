import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

df = pd.read_csv('data/optimized_taxonomy.csv')
# Keep track of which rows are core
df['is_core'] = df.index < 59

def deduplicate_subcategories(group_df):
    group_df = group_df.reset_index(drop=True)
    vectorizer = TfidfVectorizer(stop_words='english')
    try:
        tfidf = vectorizer.fit_transform(group_df['subcategory_name'])
    except ValueError:
        return group_df
        
    sim_matrix = cosine_similarity(tfidf)
    
    G = nx.Graph()
    for i in range(len(group_df)):
        G.add_node(i)
        
    for i in range(len(group_df)):
        for j in range(i+1, len(group_df)):
            if sim_matrix[i, j] > 0.15:
                G.add_edge(i, j)
                
    components = list(nx.connected_components(G))
    
    merged_rows = []
    for comp in components:
        comp_indices = list(comp)
        comp_rows = group_df.iloc[comp_indices]
        
        # Merge all unique words from all subcategories in this cluster
        all_words = " ".join(comp_rows['subcategory_name']).split()
        unique_words = list(dict.fromkeys(all_words)) # preserves order
        merged_name = " ".join(unique_words)
        
        # Prefer a core row if one exists in the cluster
        core_rows = comp_rows[comp_rows['is_core']]
        if not core_rows.empty:
            anchor_row = core_rows.iloc[0]
        else:
            anchor_row = comp_rows.iloc[0]
            
        merged_rows.append({
            'category_id': anchor_row['category_id'],
            'category_name': anchor_row['category_name'],
            'subcategory_id': anchor_row['subcategory_id'],
            'subcategory_name': merged_name,
            'is_core': anchor_row['is_core']
        })
        
    return pd.DataFrame(merged_rows)

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
