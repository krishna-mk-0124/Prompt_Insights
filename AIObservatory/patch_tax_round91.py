import pandas as pd

df = pd.read_csv('data/optimized_taxonomy.csv')

def patch_keywords(row):
    name = str(row['subcategory_name']).lower()
    kw = str(row['keywords'])
    
    # Injecting the newly discovered massive log tokens from Round 91
    if 'data analytics' in name and 'algorithm' in name:
        kw += " indexing scala scanned relate comparision ips tis breif coloumn grow covering sharp"
    elif 'strategic planning' in name:
        kw += " oa listening unknown wnat fro choices minimal factors street stuck older advice"
    elif 'business communications' in name:
        kw += " formats contractions talkative"
    elif 'project management' in name:
        kw += " filled lists receiving"
    elif 'development practices' in name and 'java' in name:
        kw += " component roll prep"
    elif 'cloud infrastructure' in name:
        kw += " ips"
    
    return kw

df['keywords'] = df.apply(patch_keywords, axis=1)
df.to_csv('data/optimized_taxonomy.csv', index=False)
