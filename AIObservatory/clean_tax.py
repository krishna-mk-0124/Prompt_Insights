import pandas as pd
import re

TAX_FILE = 'data/optimized_taxonomy.csv'

def clean_frankenstein_subcategories():
    df = pd.read_csv(TAX_FILE)
    
    # List of the blindly appended heuristic suffixes we need to strip
    suffixes = [
        " Development Practices",
        " Error Triage",
        " Data Analytics",
        " Cloud Infrastructure",
        " Security Compliance",
        " Business Communications",
        " Financial Strategy",
        " Project Management",
        " Human Resources",
        " Banking Operations",
        " Statistical Analysis",
        " Strategic Planning"
    ]
    
    # Create a regex pattern to match any of these suffixes at the end of the string
    pattern = re.compile(r'(' + '|'.join(suffixes) + r')$')
    
    # Strip the suffix
    df['subcategory_name'] = df['subcategory_name'].apply(lambda x: pattern.sub('', x).strip())
    
    df.to_csv(TAX_FILE, index=False)
    print("Successfully stripped heuristic semantic suffixes from all 236 subcategories.")

if __name__ == "__main__":
    clean_frankenstein_subcategories()
