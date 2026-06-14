import pandas as pd

TAX_FILE = 'data/optimized_taxonomy.csv'

def inject_dictionary():
    df = pd.read_csv(TAX_FILE)
    
    # Curated, non-overlapping dictionaries for key technical subcategories
    tech_dict = {
        "Java Python C++": "import def class public private syntax compile method object variable function loop array",
        "Frontend React Angular": "html css javascript ui dom jsx hook component state prop npm window document",
        "Backend Node Express": "server middleware endpoint route backend npm django flask spring",
        "Api Rest Graphql": "api rest graphql json payload endpoint get post put delete request response",
        "Git Version Control": "git commit push pull branch merge repository clone pr repository conflict",
        "Stacktrace Exception Null": "traceback exception null pointer segfault error fail stacktrace traceback line file",
        "Sql Select Update": "sql select from where join group by insert update delete query table database",
        "Big Data Hadoop": "hadoop spark mapreduce hive kafka streaming big data pipeline",
        "Kubernetes Docker Container": "kubernetes docker container pod cluster k8s image registry orchestration",
        "Ci Cd Jenkins": "ci cd jenkins gitlab pipeline deployment build action runner",
        "Terraform Ansible Infrastructure": "terraform ansible infrastructure as code yaml playbook configuration"
    }
    
    # Iterate and append
    updated_count = 0
    for idx, row in df.iterrows():
        subcat = row['subcategory_name']
        if subcat in tech_dict:
            current_keywords = str(row['keywords'])
            new_keywords = tech_dict[subcat]
            
            # Combine and deduplicate
            combined = current_keywords + " " + new_keywords
            deduped = " ".join(list(dict.fromkeys(combined.split())))
            
            df.at[idx, 'keywords'] = deduped
            updated_count += 1
            
    df.to_csv(TAX_FILE, index=False)
    print(f"Successfully injected pure technical dictionaries into {updated_count} subcategories.")

if __name__ == "__main__":
    inject_dictionary()
