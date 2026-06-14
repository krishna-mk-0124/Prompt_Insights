import pandas as pd
import re

df = pd.read_csv('data/optimized_taxonomy.csv')

# Manual dictionary for the ugliest names
name_fixes = {
    'Sow Statement Of': 'Statement of Work (SOW)',
    'Error Getting Debug': 'Debugging & Error Logs',
    'Bank Neutral': 'Legal & Tribunal Proceedings',
    'Csv Path File': 'CSV Data Loading',
    'Amex Software Error': 'Software Error Reports',
    'Concurrent Chat Sessions': 'Live Chat Concurrency',
    'Tsr Shareholder Trends': 'Shareholder Trends (TSR)',
    'Jump Guides': 'Knowledge Base & Jump Guides',
    'Monitor Build': 'Build Monitoring & DevOps',
    'Eod End Of': 'End of Day (EOD) Operations',
    'Amex Award Alternatives': 'Employee Awards & Recognition',
    'Code Changes Required': 'Code Review & Changes',
    'Jupyter Notebook Club': 'Jupyter Notebooks',
    'Script Execution Run': 'Script Execution Logs',
    'Python Pandas Numpy': 'Python Data Science (Pandas/Numpy)',
    'Diagram Link Visio': 'Diagrams & Flowcharts',
    'Big Data Hadoop': 'Big Data (Hadoop/Spark)',
    'Pricing Gl Settings': 'Pricing & GL Settings',
    'Statistical Problems': 'Statistical Analysis',
    'Chart Graph Dashboard': 'Dashboards & Visualization',
    'Amazon Brands': 'Amazon & Brand Marketing',
    'Brainstorm Ideate Concept': 'Concept Ideation',
    'Webex Conferencing': 'WebEx & Conferencing',
    'Telesales Operations': 'Telesales & Operations',
    'Pilot Project Proposals': 'Pilot Projects',
    'Video Creative Assets': 'Video & Creative Assets',
    'Compliance Database Tables': 'Compliance Data',
    'Bullet Points List': 'Presentation Bullet Points',
    'Budget Revenue Profit': 'Revenue & Profitability',
    'Csv Uploading Process': 'CSV Uploads',
    'Software Engineering & Development': 'Software Engineering',
    'Production Support & Debugging': 'Production Support'
}

stop_words = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"])

def clean_subcat_name(name):
    name = str(name).strip()
    if name in name_fixes:
        return name_fixes[name]
    
    # Title case and fix minor things
    return name.title()

def filter_parent_category_from_keywords(row):
    # Get the raw NMF keywords
    raw_keywords = str(row['keywords']).lower()
    
    # Get the parent category name words
    parent_cat_name = str(row['category_name']).lower()
    parent_words = set(re.sub(r'[^a-z\s]', '', parent_cat_name).split())
    parent_words = {w for w in parent_words if len(w) > 2 and w not in stop_words}
    
    # Remove parent words from the keywords
    keyword_list = raw_keywords.split()
    filtered = [w for w in keyword_list if w not in parent_words]
    
    # Also ensure the words from the new subcategory name are included!
    new_subcat_words = re.sub(r'[^a-z\s]', '', str(row['subcategory_name']).lower()).split()
    new_subcat_words = [w for w in new_subcat_words if len(w) > 2 and w not in stop_words]
    
    final_keywords = filtered + new_subcat_words
    return " ".join(list(dict.fromkeys(final_keywords)))

df['subcategory_name'] = df['subcategory_name'].apply(clean_subcat_name)
df['keywords'] = df.apply(filter_parent_category_from_keywords, axis=1)

# Inject technical gravity words for specific subcategories
tech_injections = {
    'Software Error Reports': ' stacktrace exception null traceback pointer segfault error fail line file',
    'Big Data (Hadoop/Spark)': ' hadoop spark mapreduce hive kafka streaming pipeline',
    'Debugging & Error Logs': ' debug error warning log traceback exception',
    'Code Review & Changes': ' git commit push pull request merge conflict diff review',
    'Script Execution Logs': ' bash script execution run python java node logs output',
}

for i, row in df.iterrows():
    name = row['subcategory_name']
    if name in tech_injections:
        df.at[i, 'keywords'] = str(df.at[i, 'keywords']) + tech_injections[name]

df.to_csv('data/optimized_taxonomy.csv', index=False)
print("Taxonomy rebuilt with clean names and isolated subcategory keywords.")
