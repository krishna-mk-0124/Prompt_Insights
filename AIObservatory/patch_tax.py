import pandas as pd

df = pd.read_csv('data/optimized_taxonomy.csv')

def patch_keywords(row):
    name = str(row['subcategory_name']).lower()
    kw = str(row['keywords'])
    
    # Route the massive unmapped clusters into their proper logical buckets
    if 'error triage' in name and 'stacktrace' in name:
        kw += " badrequest conferma poid cdn fail issue"
    elif 'data analytics' in name and 'algorithm' in name:
        kw += " wide weighted operator naming maps brackets"
    elif 'human resources' in name:
        kw += " candidates park wife spot efforts"
    elif 'project management' in name:
        kw += " rush teach ongoing phases playbook"
    elif 'banking operations' in name or 'financial strategy' in name:
        kw += " aml privileged fins stock"
    elif 'data analytics' in name and 'bigquery' in name:
        kw += " bq"
    elif 'development practices' in name and 'frontend' in name:
        kw += " grid browser sugarcoat"
    elif 'cloud infrastructure' in name:
        kw += " env directory eea"
    elif 'business communications' in name and 'email' in name:
        kw += " conclusion valuable"
    
    return kw

df['keywords'] = df.apply(patch_keywords, axis=1)
df.to_csv('data/optimized_taxonomy.csv', index=False)
