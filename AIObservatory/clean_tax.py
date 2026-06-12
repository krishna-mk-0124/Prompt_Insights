import pandas as pd
df = pd.read_csv('data/optimized_taxonomy.csv')
print('Total rows:', len(df))

clean_df = df.iloc[:540]
removed_df = df.iloc[540:]

print("--- REVIEW OF REMOVED SUBCATEGORIES ---")
for _, row in removed_df.iterrows():
    print(f"Removed: Category={row['category_name']}, Subcategory={row['subcategory_name']}")

clean_df.to_csv('data/optimized_taxonomy.csv', index=False)
print("Taxonomy cleaned!")
