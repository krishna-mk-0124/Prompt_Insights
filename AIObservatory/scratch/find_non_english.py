import pandas as pd
import json

df = pd.read_csv('data/optimized_taxonomy.csv')

keywords = ['spanish', 'german', 'french', 'español', 'm√°s', 'informaci', 'qu√©', '„äæ', 'tradu', 'banco']

def has_keyword(name):
    name = str(name).lower()
    for k in keywords:
        if k in name:
            return True
    return False

res = df[df['subcategory_name'].apply(has_keyword)]
with open('scratch/non_english.json', 'w', encoding='utf-8') as f:
    json.dump(res[['subcategory_id', 'subcategory_name']].to_dict('records'), f, ensure_ascii=False, indent=2)
