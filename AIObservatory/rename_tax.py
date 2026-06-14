import pandas as pd

df = pd.read_csv('data/optimized_taxonomy.csv')

def make_meaningful(row):
    name = str(row['subcategory_name']).strip()
    
    words = name.split()
    if len(words) >= 5:
        return name
        
    name_lower = name.lower()
    
    # Heuristic matching based on the words
    if any(x in name_lower for x in ['java', 'python', 'c++', 'react', 'angular', 'node', 'express', 'api', 'graphql', 'git', 'terraform', 'software', 'code']):
        return f"{name} Software Engineering and Architecture"
    elif any(x in name_lower for x in ['stacktrace', 'exception', 'null', 'leak', 'cpu', 'crash', 'outage', 'bug', 'defect', 'error']):
        return f"{name} Issue Troubleshooting and Debugging"
    elif any(x in name_lower for x in ['sql', 'table', 'column', 'etl', 'nosql', 'mongodb', 'hadoop', 'bigquery', 'dax', 'data', 'row', 'csv']):
        return f"{name} Database Querying and Analytics"
    elif any(x in name_lower for x in ['aws', 'kubernetes', 'docker', 'ci', 'cd', 'jenkins', 'network', 'load balancer', 'gateway', 'linux', 'system']):
        return f"{name} Cloud Infrastructure and Deployment"
    elif any(x in name_lower for x in ['auth', 'rbac', 'encryption', 'ssl', 'vulnerability', 'phishing', 'compliance', 'gdpr', 'risk', 'security']):
        return f"{name} Security Compliance and Risk Management"
    elif any(x in name_lower for x in ['email', 'grammar', 'translate', 'presentation', 'documentation', 'slack', 'message', 'write', 'diagram', 'meeting']):
        return f"{name} Corporate Communications and Drafting"
    elif any(x in name_lower for x in ['roi', 'budget', 'roadmap', 'market', 'sales', 'revenue', 'pricing', 'finance', 'treasury', 'accounting', 'investment', 'capital', 'ebitda']):
        return f"{name} Financial Strategy and Market Planning"
    elif any(x in name_lower for x in ['agile', 'scrum', 'ticket', 'workflow', 'meeting', 'release', 'qa', 'test', 'jira', 'schedule']):
        return f"{name} Agile Project Management and Testing"
    elif any(x in name_lower for x in ['hr', 'talent', 'attrition', 'mentorship', 'payroll', 'training', 'immigration', 'benefits', 'employee']):
        return f"{name} Human Resources and Talent Management"
    elif any(x in name_lower for x in ['amex', 'card', 'merchant', 'offers', 'account', 'banking']):
        return f"{name} AMEX Banking and Card Operations"
    elif any(x in name_lower for x in ['excel', 'chart', 'statistics', 'pandas', 'summarize', 'brainstorm', 'math', 'plot', 'power bi', 'tableau']):
        return f"{name} Data Analysis and Statistical Processing"
    else:
        return f"{name} Operations and Strategic Planning"

df['subcategory_name'] = df.apply(make_meaningful, axis=1)
df.to_csv('data/optimized_taxonomy.csv', index=False)
