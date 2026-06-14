import pandas as pd

df = pd.read_csv('data/optimized_taxonomy.csv')

def smart_rename(row):
    name = str(row['subcategory_name']).strip()
    
    # Clean up underscores
    name = name.replace("_", " ").title()
    words = name.split()
    
    # Take up to 3 distinct words from the name to form the core topic
    # Using dict.fromkeys to remove duplicates while preserving order
    core_topic = " ".join(list(dict.fromkeys(words))[:3])
    
    name_lower = name.lower()
    
    if any(x in name_lower for x in ['java', 'python', 'c++', 'react', 'angular', 'node', 'express', 'api', 'graphql', 'git', 'terraform', 'software', 'code']):
        suffix = "Development Practices"
    elif any(x in name_lower for x in ['stacktrace', 'exception', 'null', 'leak', 'cpu', 'crash', 'outage', 'bug', 'defect', 'error', 'badrequest', 'grep']):
        suffix = "Error Triage"
    elif any(x in name_lower for x in ['sql', 'table', 'column', 'etl', 'nosql', 'mongodb', 'hadoop', 'bigquery', 'dax', 'data', 'row', 'csv']):
        suffix = "Data Analytics"
    elif any(x in name_lower for x in ['aws', 'kubernetes', 'docker', 'ci', 'cd', 'jenkins', 'network', 'load balancer', 'gateway', 'linux', 'system', 'directory', 'remote', 'browser']):
        suffix = "Cloud Infrastructure"
    elif any(x in name_lower for x in ['auth', 'rbac', 'encryption', 'ssl', 'vulnerability', 'phishing', 'compliance', 'gdpr', 'risk', 'security', 'guard', 'otp', 'identification']):
        suffix = "Security Compliance"
    elif any(x in name_lower for x in ['email', 'grammar', 'translate', 'presentation', 'documentation', 'slack', 'message', 'write', 'diagram', 'meeting']):
        suffix = "Business Communications"
    elif any(x in name_lower for x in ['roi', 'budget', 'roadmap', 'market', 'sales', 'revenue', 'pricing', 'finance', 'treasury', 'accounting', 'investment', 'capital', 'ebitda', 'invoice', 'loss']):
        suffix = "Financial Strategy"
    elif any(x in name_lower for x in ['agile', 'scrum', 'ticket', 'workflow', 'meeting', 'release', 'qa', 'test', 'jira', 'schedule', 'argument', 'rule']):
        suffix = "Project Management"
    elif any(x in name_lower for x in ['hr', 'talent', 'attrition', 'mentorship', 'payroll', 'training', 'immigration', 'benefits', 'employee', 'staff', 'admin', 'candidate']):
        suffix = "Human Resources"
    elif any(x in name_lower for x in ['amex', 'card', 'merchant', 'offers', 'account', 'banking']):
        suffix = "Banking Operations"
    elif any(x in name_lower for x in ['excel', 'chart', 'statistics', 'pandas', 'summarize', 'brainstorm', 'math', 'plot', 'power bi', 'tableau']):
        suffix = "Statistical Analysis"
    else:
        suffix = "Strategic Planning"

    # Ensure it reaches 4-5 words
    final_name = f"{core_topic} {suffix}".strip()
    return final_name

df['subcategory_name'] = df.apply(smart_rename, axis=1)
df.to_csv('data/optimized_taxonomy.csv', index=False)
