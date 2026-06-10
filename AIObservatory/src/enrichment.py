def get_enrichment_map():
    return {
        0: "java python c++ javascript typescript node react angular vue compilation exception syntax loop error debugger stacktrace variables functions",
        1: "excel forecast kpi trend ab testing cohort quantitative visual tableau powerbi statistical variance metric",
        2: "email response draft follow-up tone clarify leadership summarize meeting notes memo announcement",
        3: "policy audit regulatory privacy sox aml compliance risk gap documentation scenario",
        4: "swot tradeoff roadmap roi portfolio scenario business case long-term organizational cost-benefit",
        5: "sop step-by-step workflow bottleneck escalation sla raci ticket triage checklist",
        6: "slide deck powerpoint whitepaper biography template summary condense briefing presentation",
        7: "explain definition overview simplify conceptual fundamentals clarify",
        8: "llm chatgpt prompt generative ai artificial intelligence machine learning automation productivity",
        9: "brainstorm translate rephrase grammar wording organize checklist",
        10: "aws ec2 s3 kubernetes docker k8s terraform load balancer dns vpc network infra",
        11: "budget expense revenue profit margin tax roi pricing break-even depreciation cash flow",
        12: "cx nps customer satisfaction loyalty omnichannel interaction retention complaint voice",
        13: "hr performance review succession career promotion workforce employee diversity hiring",
        14: "sox deficiency matrix testing control automation compliance dashboard reporting",
        15: "roadmap mvp feature backlog sprint agile scrum go-to-market lifecycle",
        16: "visa mastercard card payment terminal pos checkout stripe paypal transaction settlement fraud chargeback gateway",
        17: "sql select distinct insert update delete query table database row column schema etl nosql json mongodb bigquery spark pipeline",
        18: "auth oauth saml jwt mfa sso password zero trust certificate encryption vulnerability cyber phishing threat",
        19: "api microservices message broker middleware integration hybrid cloud architecture legacy modernization"
    }

def enrich_taxonomy(df):
    enrichment_map = get_enrichment_map()
    
    def apply_enrichment(row):
        cat_id = row['category_id']
        desc = row['processed_desc']
        if cat_id in enrichment_map:
            # Append enrichment keywords to the lexical vector
            return desc + " " + enrichment_map[cat_id]
        return desc
        
    df['enriched_desc'] = df.apply(apply_enrichment, axis=1)
    return df
