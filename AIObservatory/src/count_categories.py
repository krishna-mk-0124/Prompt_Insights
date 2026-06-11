import pandas as pd
import os

def main():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    input_file = os.path.join(data_dir, "fully_categorized_dataset.csv")
    output_file = os.path.join(data_dir, "category_counts.csv")
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Ensure discovery.py has run successfully.")
        return
        
    print(f"Loading {input_file}...")
    df = pd.read_csv(input_file)
    
    print("Aggregating counts...")
    # Group by category and subcategory
    counts = df.groupby(
        ["category_id", "category_name", "subcategory_id", "subcategory_name"]
    ).size().reset_index(name="Count")
    
    # Sort for easy reading: first by Category ID, then by descending Count
    counts = counts.sort_values(by=["category_id", "Count"], ascending=[True, False])
    
    # Add a column for Total Count per Category for easier reading
    category_totals = df.groupby(["category_id"])["category_id"].count().reset_index(name="Total count for category")
    counts = pd.merge(counts, category_totals, on="category_id", how="left")
    
    counts.to_csv(output_file, index=False)
    print(f"Successfully wrote {len(counts)} aggregated rows to: {output_file}")

if __name__ == "__main__":
    main()
