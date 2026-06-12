import os
import sys
import pandas as pd
import numpy as np
import string
import warnings

# Suppress TruncatedSVD division by zero warning when variance becomes extremely small
warnings.filterwarnings('ignore', category=RuntimeWarning)

import math
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import MiniBatchKMeans
from .preprocess import preprocess_prompt

def clean_and_truncate(text):
    text = str(text)
    words = text.split()
    if len(words) > 50:
        return " ".join(words[:50])
    return text

def extract_top_keywords(tfidf_matrix, vectorizer, cluster_labels, num_clusters, top_n=2):
    feature_names = np.array(vectorizer.get_feature_names_out())
    cluster_names = {}
    for i in range(num_clusters):
        cluster_indices = np.where(cluster_labels == i)[0]
        if len(cluster_indices) == 0:
            cluster_names[i] = "empty_cluster"
            continue
        cluster_tfidf = tfidf_matrix[cluster_indices]
        mean_tfidf = np.asarray(cluster_tfidf.mean(axis=0)).flatten()
        top_indices = mean_tfidf.argsort()[-top_n:][::-1]
        cluster_names[i] = "_".join(feature_names[top_indices])
    return cluster_names

def run_hybrid_discovery():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    input_file = os.path.join(data_dir, "english_prompts.txt")
    tax_file = os.path.join(data_dir, "optimized_taxonomy.csv")
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Run language_router.py first.")
        sys.exit(1)
    if not os.path.exists(tax_file):
        print(f"Error: {tax_file} not found.")
        sys.exit(1)
        
    print(f"Loading official taxonomy from {tax_file}...")
    tax_df = pd.read_csv(tax_file)
    tax_df["combined_desc"] = tax_df["category_name"] + " " + tax_df["subcategory_name"]
    
    # Preprocess taxonomy definitions
    tax_df["processed_desc"] = tax_df["combined_desc"].apply(preprocess_prompt)
    
    print(f"Loading texts from {input_file}...")
    garbage_signatures = [
        '„', 'ç', '√', 'é', '¨', 'π', 'triggerdagrunoperator', 
        'est√°', 'm√°s_qu√©', 'informaci√≥n', 'd√≠as',
        'zz_estan', 'jersey_bars', 'informaci', 'essere_september',
        'archivo_lookup', 'han_traduce', 'comp_thsi'
    ]
    with open(input_file, "r", encoding="utf-8") as f:
        lines = []
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
                
            line_lower = line_str.lower()
            
            # Explicit massive garbage strings (now using lower() so it catches them!)
            if any(sig in line_lower for sig in garbage_signatures):
                continue
                
            # Heuristic: Drop heavy mojibake/non-ASCII garbage
            # If a prompt contains more than 15 non-ASCII characters, it's likely a massive log/hex dump
            non_ascii_count = sum(1 for c in line_str if ord(c) > 127)
            if non_ascii_count > 15:
                continue
                
            lines.append(line_str)
        
    df = pd.DataFrame({"raw_text": lines})
    print(f"Loaded {len(df)} English prompts. Beginning Hybrid Mapping...")
    
    print("\n[Phase 1/4] Pre-processing User Prompts (Max 50 words)")
    df["truncated_prompt"] = df["raw_text"].apply(clean_and_truncate)
    
    processed_texts = []
    total = len(df)
    for i, text in enumerate(df["truncated_prompt"]):
        if i % 25000 == 0 and i > 0:
            print(f"  -> Processed {i:,} / {total:,} prompts ({(i/total)*100:.1f}%)")
        processed_texts.append(preprocess_prompt(text))
    df["processed_text"] = processed_texts
    
    print("\n[Phase 2/4] Zero-Shot Vectorization (TF-IDF)")
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    
    # Block common noise tokens that hijack the unsupervised clusters
    custom_noise = [
        'st', 'oo', 'hi', 'hello', 'hey', 'test', 'com', 'www', 'http', 'https', 'nd', 'rd', 'th', 'pls', 'please',
        'strategy', 'optimization', 'guidance', 'support', 'clarification', 'review', 'management', 'planning', 'improvement', 'architecture', 'design',
        'using', 'run', 'need', 'new', 'make', 'better', 'want', 'just', 'add', 'total', 'et', 'does', 'know', 'let', 'like', 'looks', 'based', 'level', 'high', 'share', 'id', 'ore', 'doing', 'hope', 'think', 'don', 'provide', 'questions', 'ask', 'check', 'good', 'morning', 'getting', 'include', 'sure', 'yes', 'work', 'al', 'dl', 'che', 'copy', 'paste', 'tell',
        'dont', 'remove', 'working', 'file', 'files', 'use', 'cases', 'que', 'el', 'la', 'en', 'thank', 'understand', 'able', 'different', 'cm', 'dw', 'excel', 'sheet', 'information', 'additional', 'days', 'day', 'say', 'way', 'dl', 'il', 'create', 'folder', 'uk', 'india', 'added', 'details', 'report', 'status', 'updated', 'update', 'plan', 'ot', 'ey', 'ok', 'lets', 'field', 'months', 'send', 'set', 'se', 'vs', 'naa', 'prj', 'au', 'variable', 'odl', 'agent', 'match', 'jan', 'mar', 'pm', 'quick', 'clever', 'regarding', 'clic', 'case', 'customer', 'engagement', 'works', 'non',
        'change', 'list', 'question', 'rate', 'count', 'key', 'company', 'sentence', 'instead', 'process', 'sent', 'today', 'local', 'members', 'para', 'los', 'experience', 'following', 'note', 'job', 'right', 'left', 'things', 'come', 'did', 'look', 'forward', 'lo', 'es', 've', 'request', 'kindly', 'teams', 'got', 'ol', 'py', 'line', 'feel', 'free', 'week', 'bit', 'sound', 'llc', 'ca', 'di', 'le', 'issues', 'identified', 'april', 'march', 'input', 'output', 'coming', 'tool', 'search', 'view', 'tomorrow', 'canada', 'united', 'discuss', 'color', 'blue', 'tools', 'die', 'und', 'em', 'session',
        'actually', 'means', 'clear', 'ar', 'active', 'used', 'values', 'section', 'open', 'word', 'por', 'una', 'base', 'years', 'll', 'mail', 'attached', 'saying', 'possible', 'size', 'template', 'place', 'instructions', 'thing', 'rewrite', 'issue', 'description', 'updates', 'okay', 'yeah', 'feedback', 'reflect', 'happy', 'friday', 'generate', 'sum', 'going', 'je', 'est', 'apr', 'feb', 'needs', 'numbers', 'number', 'suggest', 'similar', 'answer', 'answers', 'start', 'compare', 'bullets', 'multiple', 'asking', 'method', 'uploaded', 'lead', 'favor', 'factual', 'studies',
        'point', 'wait', 'min', 'example', 'people', 'little', 'tech', 'del', 'si', 'present', 'current', 'state', 'im', 'wanted', 'connect', 'currently', 'related', 'product', 'reason', 'ready', 'person', 'showing', 'variables', 'sense', 'hours', 'na', 'reply', 'great', 'weekend', 'really', 'appreciate', 'codes', 'bucket', 'dec', 'given', 'available', 'target', 'gb', 'oy', 'tr', 'convert', 'previous', 'mi', 'colleague', 'colleagues', 'launch', 'visual', 'background', 'options', 'june', 'july', 'maybe', 'phase', 'old', 'mentioned',
        'context', 'fields', 'kind', 'sentences', 'tier', 'bt', 'ip', 'mention', 'looking', 'try', 'rating', 'far', 'called', 'las', 'como', 'replace', 'said', 'lot', 'decision', 'records', 'task', 'approach', 'aligned', 'ensure', 'discussion', 'confirm', 'chat', 'coffee', 'lines', 'ich', 'complete', 'progress', 'specific', 'discussed', 'changed', 'sr', 'esta', 'final', 'version', 'default', 'apply', 'pdf', 'viewer', 'asked', 'comment', 'sounds', 'mac', 'city', 'names', 'header', 'sorry', 'delay', 'exact', 'layout', 'taking', 'exercise', 'confirmation',
        'write', 'submit', 'date', 'om', 'hai', 'second', 'completed', 'da', 'doesn', 'dates', 'ones', 'gave', 'average', 'idea', 'este', 'su', 'problem', 'break', 'received', 'talk', 'chatgpt', 'home', 'monday', 'identify', 'past', 'split', 'original', 'category', 'giving', 'rows', 'doc', 'te', 'pero', 'improve', 'box', 'focus', 'areas', 'les', 'tu', 'latest', 'needed', 'reference', 'provided', 'continue', 'frame', 'conversation', 'starters', 'tab', 'window', 'aug', 'oct', 'screen', 'shot', 'calls', 'happens', 'believe', 'fy', 'seeing', 'captions',
        'tables', 'space', 'type', 'missing', 'title', 'slide', 'ya', 'quiero', 'respond', 'doesnt', 'starting', 'exactly', 'having', 'whats', 'trying', 'car', 'lower', 'office', 'blank', 'ist', 'ho', 'role', 'evidence', 'created', 'exec', 'build', 'scenario', 'room', 'overall', 'vibes', 'results', 'pany', 'eer', 'concise', 'filter', 'ap', 'customers', 'bloc', 'round', 'thinking', 'align', 'scope', 'necesito', 'respuesta', 'record', 'insert', 'draw', 'um', 'os', 'max', 'avg', 'declare', 'interval', 'emails', 'abp', 'limited', 'classification', 'insights', 'graphs',
        'detailed', 'ppt', 'column', 'users', 'documents', 'examples', 'period', 'sending', 'bio', 'directly', 'sl', 'feels', 'white', 'reports', 'meant', 'says', 'hotels', 'client', 'thats', 'mind', 'rules', 'stage', 'shorten', 'friendly', 'comments', 'greatest', 'opportunity', 'requirement', 'meet', 'profile', 'profiles', 'times', 'sign', 'transactions', 'activity', 'implement', 'beginning', 'didn', 'reword', 'benefit', 'reduce', 'cell', 'recent', 'save', 'button', 'parent', 'child', 'meeting', 'low', 'medium', 'mo', 'ct', 'yr', 'projects', 'mindset', 'pages', 'btn', 'happened', 'store', 'element', 'describes',
        'headline', 'crisp', 'simpler', 'ja', 'items', 'adding', 'sort', 'map', 'responses', 'meaning', 'happen', 'upload', 'single', 'hola', 'esto', 'liner', 'soon', 'future', 'approved', 'properly', 'weeks', 'monthly', 'drop', 'removed', 'axp', 'aa', 'isn', 'accordingly', 'guess', 'differences', 'solve', 'du', 'pour', 'pa', 'strong', 'refresh', 'making', 'changes', 'sub', 'dim', 'cuenta', 'ser', 'self', 'site', 'reach', 'nov', 'sep', 'rtf', 'cht', 'thursday', 'wednesday', 'coll', 'direct', 'usage', 'io', 'vertx', 'cg', 'label', 'intro', 'walked',
        'elaborate', 'age', 'metrics', 'existing', 'wrong', 'shorter', 'live', 'happening', 'team', 'rephrase', 'sharing', 'main', 'mas', 'solo', 'nice', 'love', 'products', 'ingles', 'bien', 'ir', 'setting', 'mins', 'positive', 'stay', 'statements', 'features', 'english', 'history', 'cust', 'solution', 'cfrcpi', 'turn', 'suggestions', 'hold', 'revise', 'click', 'fecha', 'prepare', 'basic', 'talking', 'news', 'ad', 'hoc', 'txt', 'str', 'cd', 'wont', 'cc', 'ed', 'ni', 'london', 'expand', 'acronyms', 'review', 'check', 'make', 'sure', 'reviewing', 'update', 'status', 'tell', 'help', 'know', 'needs',
        'combine', 'including', 'channel', 'makes', 'analyze', 'platform', 'separate', 'won', 'easier', 'scheduled', 'till', 'print', 'segment', 'band', 'markets', 'double', 'outside', 'iy', 'ee', 'explanation', 'modify', 'setup', 'mm', 'percentage', 'application', 'couple', 'sa', 'jun', 'jul', 'export', 'written', 'des', 'das', 'understanding', 'empathetic', 'book', 'dev', 'cust_xref_id', 'net', 'agree', 'alignment', 'cross', 'benefits', 'experiences', 'relevant', 'roles', 'reviewed', 'trim', 'common', 'timeline', 'yesterday', 'rs', 'combined', 'theme', 'requests', 'moving', 'quickly', 'screenshot', 'validating', 'push', 'red', 'proceed',
        'early', 'avoid', 'alternate', 'implemented', 'subtitle', 'interesting', 'axis', 'speak', 'pattern', 'narrative', 'equipo', 'correo', 'party', 'ui', 'hour', 'mapping', 'challenge', 'included', 'queries', 'tabs', 'extra', 'redo', 'bad', 'success', 'weekly', 'creating', 'processing', 'ma', 'ha', 'unique', 'validate', 'desc', 'ke', 'ki', 'ce', 'une', 'fully', 'cap', 'applicable', 'checks', 'engineer', 'maximum', 'rates', 'personal', 'track', 'fast', 'equal', 'greater', 'important', 'fah', 'stuff', 'moved', 'gto',
        'ent', 'verticles', 'reaching', 'tuesday', 'clean', 'style', 'characters', 'stop', 'leaders', 'worked', 'character', 'dataset', 'nos', 'hacer', 'away', 'counts', 'email', 'chain', 'editable', 'me', 'failed', 'org', 'comes', 'normal', 'contact', 'pending', 'bu', 'ids', 'columns', 'ucc', 'initial', 'categories', 'order', 'migration', 'ft', 'sobre', 'todo', 'recommendation', 'delta', 'submitted', 'previously', 'assumptions', 'cfr', 'gap', 'guardian', 'gonna', 'demo', 'fit', 'purpose', 'innovative', 'pcm', 'rule', 'kpa', 'defensible', 'redactar',
        'brief', 'highlight', 'overview', 'acct_', 'unrecognized', 'exclusion', 'mismatch', 'analyse', 'didnt', 'gracias', 'son', 'refer', 'helpful', 'tracker', 'thought', 'closed', 'sections', 'area', 'dd', 'time', 'risks', 'charts', 'generic', 'tktg', 'told', 'fun', 'icons', 'world', 'earlier', 'polish', 'exist', 'duplicates', 'pro', 'usa', 'bring', 'skills', 'download', 'quote', 'concerns', 'aware', 'basis', 'daily', 'apart', 'initiative', 'rasc', 'started', 'register', 'higher', 'driven', 'photo', 'circle', 'arrow', 'arrows', 'dash', 'british',
        'downloadable', 'ways', 'casual', 'calculation', 'later', 'cover', 'went', 'cards', 'porque', 'ideas', 'repo', 'matching', 'mst', 'partners', 'drive', 'adjust', 'highlighted', 'assigned', 'prior', 'availability', 'wise', 'random', 'middle', 'straight', 'lol', 'deep', 'summary', 'rank', 'servers', 'cms', 'january', 'mr', 'flag', 'complex', 'differently', 'cleaner', 'pie', 'dashes', 'floor', 'cool', 'birthday', 'appear', 'appropriate', 'humor', 'dot', 'ety', 'challenges', 'pti', 'puedes', 'ayudar', 'supported', 'ics', 'plus', 'sur',
        'dy', 'wanting', 'seperate', 'wordy', 'gl', 'shows', 'suggested', 'duplicate', 'remember', 'hard', 'submission', 'opportunities', 'md', 'perfect', 'confused', 'comma', 'sy', 'lt', 'attach', 'funny', 'places', 'catch', 'ores', 'super', 'signature', 'mismatches', 'potential', 'clarify', 'ub', 'repeating', 'der', 'wir', 'primary', 'industry', 'timelines', 'hear', 'hay', 'algo', 'helps', 'telling', 'tenure', 'renewal', 'necessary', 'spacing', 'airport', 'quotes', 'require', 'ic', 'var', 'ipc', 'planned', 'basically', 'boss', 'chnage',
        'wording', 'wow', 'me_', 'ahead', 'canvas', 'exclude', 'ahora', 'idk', 'choice', 'visually', 'polished', 'inputs', 'block', 'events', 'entire', 'unable', 'labels', 'consider', 'survey', 'rated', 'elements', 'clarity', 'batch', 'operational', 'strategic', 'sheets', 'mistake', 'pas', 'den', 'cells', 'display', 'fr', 'distribution', 'material', 'involved', 'requested', 'independently', 'ive', 'dpb', 'mark', 'david', 'removing', 'recipients',
        'differ', 'doubts', 'shall', 'year', 'rid', 'useful', 'checking', 'comparison', 'caso', 'tengo', 'perspective', 'raised', 'missed', 'reframe', 'individual', 'selected', 'act', 'gaps', 'longer', 'incorporate', 'sme', 'separated', 'formatted', 'digit', 'ye', 'themes', 'building', 'impacted', 'alerts', 'typing', 'readable', 'tc', 'device', 'totals', 'bands', 'paid', 'gen', 'llamada', 'visibility', 'organization', 'run_date', 'exit', 'hit', 'inside', 'ambiguous', 'afternoon', 'special', 'comms', 'templates', 'hierarchy', 'decline', 'range', 'buckets',
        'looped', 'fixing', 'unless', 'combination', 'correctly', 'gs', 'tried', 'skip', 'light', 'consolidate', 'rename', 'yo', 'heading', 'ik', 'uma', 'filters', 'reminder', 'scenarios', 'regards', 'paragraphs', 'remind', 'listed', 'dr', 'sin', 'tiene', 'rn', 'nn', 'boy', 'inglese', 'sono', 'fits', 'sounding', 'regenerate', 'stamp', 'ne', 'nous', 'mit', 'relationship', 'effective', 'subtle', 'counting', 'topics', 'covered', 'haven', 'heard', 'ow', 'alex', 'eyes', 'posted', 'applications', 'summarise', 'appeal',
        'topic', 'fixed', 'muy', 'thoughts', 'web', 'fetch', 'succinct', 'keywords', 'closest', 'ez', 'duplicated', 'slot', 'house', 'theres', 'strings', 'pointers', 'completely', 'generated', 'incorrect', 'figure', 'joining', 'verbiage', 'seconds', 'parties', 'bro', 'mockup', 'aggregate', 'zero', 'takes', 'center', 'ti', 'effort', 'implementation', 'situation', 'transition', 'sessions', 'variant', 'chance', 'definitely', 'handle', 'ambiguity', 'family', 'wants',
        'decile', 'polite', 'negative', 'waiting', 'york', 'decir', 'tested', 'tod', 'pc', 'noted', 'confirmed', 'recommendations', 'font', 'passed', 'clearer', 'xlsx', 'moment', 'met', 'length', 'capture', 'loop', 'icon', 'ms', 'mid', 'keeping', 'clearly', 'layer', 'decimal', 'confusing', 'scripts', 'revert', 'ratio', 'black', 'mentor', 'redundant', 'addressed', 'losing', 'activities', 'discovery', 'eine', 'improvements', 'referring', 'recently', 'helping', 'concern', 'raise', 'martin', 'linkedin',
        'checked', 'restructure', 'calculated', 'est√°', 'rationale', 'mode', 'specifically', 'processes', 'connected', 'play', 'remaining', 'consolidated', 'van', 'audience', 'seen', 'repeat', 'impactful', 'della', 'parte', 'numeric', 'pictures', 'welcome', 'half', 'aw', 'sn', 'hand', 'touch', 'ayudame', 'donde', 'addressing', 'thanking', 'cheat', 'principle', 'auf', 'beautiful', 'conclude', 'determine', 'falling', 'edits', 'minor', 'pra', 'fir',
        'apologize', 'zone', 'gone', 'providing', 'upcoming', 'isnt', 'tighten', 'mas', 'est', 'phrase', 'extract', 'd_amt', 'upgrade', 'hire', 'body', 'cancelled', 'xt', 'suitable', 'amend', 'boxes', 'teh', 'walk', 'cycle', 'game', 'wrote', 'oh', 'manually', 'stories', 'near', 'copied', 'po', 'ts', 'veo', 'board', 'receive', 'skill', 'oot', 'upper', 'vi', 'collaboration', 'focused', 'agents', 'tl', 'expect', 'handling',
        'actioned', 'dive', 'deadline', 'pick', 'gt', 'informaci√≥n', 'proposed', 'quarter', 'million', 'hrs', 'references', 'john', 'resend', 'bulleted', 'lg', 'performed', 'owned', 'lane', 'splitting', 'alot', 'passing', 'grand', 'companies', 'goes', 'rebuild', 'overlapping', 'editor', 'havent', 'december', 'february', 'ago', 'minute', 'summer', 'excited', 'supposed', 'movements', 'gets', 'ageing', 'insight', 'led', 'matter', 'attention', 'drawn', 'scan', 'entirely', 'awesome', 'spots',
        'informaci', 'calculating', 'empresa', 'screenshots', 'dt', 'slightly', 'probably', 'extension', 'especially', 'guys', 'tags', 'findings', 'upto', 'fare', 'installed', 'ac', 'ae', 'air', 'par', 'club', 'alternatives', 'life', 'bca', 'snippet', 'articles', 'prompts', 'edge', 'politely', 'conversational', 'exchange', 'cps', 'discover', 'puede', 'tener', 'isp', 'gsp', 'inform', 'mco', 'recommended', 'remain', 'restaurant', 'saw', 'approvals', 'sorting', 'informed', 'sir', 'worth', 'seven',
        'whl', 'shouldn', 'keys', 'popular', 'proper', 'rcr', 'cut', 'verify', 'titles', 'estoy', 'taken', 'supporting', 'weird', 'repeated', 'opening', 'auto', 'attachment', 'goals', 'treatment', 'accurate', 'ou', 'ss', 'va', 'qu', 'bl', 'bb', 'tt', 'gy', 'corrected', 'explaining', 'mon', 'sun', 'nan', 'colour', 'shoudl', 'manage', 'energy', 'visuals', 'consulting', 'rebase', 'republic',
        'urllib', 'resolve', 'perform', 'revised', 'nope', 'descriptions', 'asks', 'joined', 'property', 'miss', 'compared', 'threshold', 'came', 'amazing', 'ice', 'pretty', 'width', 'pd', 'ws', 'divide', 'softer', 'west', 'road', 'ayudas', 'texto', 'tight', 'heads', 'ab', 'ek', 'observations', 'contains', 'ese', 'nivel', 'capability', 'demonstrated', 'apologies', 'reschedule', 'wanna', 'pos', 'evaluation', 'buying', 'interviewed', 'forma', 'mis', 'clusters', 'speed', 'contrib', 'hvnplccpwa',
        'trend', 'interested', 'm√°s', 'qu√©', 'simplify', 'visible', 'easily', 'considered', 'ytd', 'xx', 'reached', 'defined', 'emoji', 'hide', 'resolved', 'money', 'internally', 'ingl√©s', 'links', 'propose', 'finally', 'dire', 'eh', 'collaborative', 'fyi', 'enhance', 'choose', 'ao', 'versions', 'advise', 'duration', 'sat', 'appealing', 'fluff', 'e_', 'consistent', 'performers', 'tuple', 'ideation', 'checkpoints', 'overarching',
        'exceed', 'professionally', '___', 'digits', 'answered', 'considering', 'closing', 'stakeholders', 'changing', 'proceso', 'desde', 'sea', 'cuando', 'fo', 'prospect', 'master', 'finding', 'informal', 'applying', 'marked', 'sunrise', 'wondering', 'baby', 'mentioning', 'wish', 'chats', 'face', 'd√≠a', 'hoy', 'measure', 'practical', 'starts', 'backing', 'corrective', 'summarizes', 'verifying', 'bcv', 'pcv', 'verbose', 'complicated', 'eue', 'bsn', 'chawla',
        'forget', 'nyc', 'particular', 'messages', 'intent', 'ingl', 'mapped', 'analyst', 'health', 'night', 'headers', 'named', 'mo', 'ct', 'merged', 'catchy', 'hidden', 'preview', 'msg', 'questo', 'dei', 'documento', 'trust', 'gives', 'soc', 'plant', 'countries', 'refresher', 'sizes', 'faster', 'coding', 'watch', 'outs', 'flowchart', 'involvement', 'vague', 'tho',
        'evolution', 'initially', 'ag', 'notice', 'ignore', 'photos', 'punchy', 'drafted', 'biggest', 'weight', 'leaving', 'ba', 'bc', 'prop', 'australia', 'char', 'todos', 'deleted', 'lost', 'jsut', 'reformat', 'speaking', 'busy', 'eso', 'ver', 'rev', 'appears', 'frequently', 'spike', 'causing', 'wishes', 'templated', 'kona',
        'commas', 'grey', 'sequence', 'solutions', 'vous', 'deliver', 'likely', 'discussions', 'pl', 'priorities', 'er', 'clients', 'tm', 'timing', 'buy', 'wa', 'commentary', 'overlap', 'conditions', 'august', 'allocation', 'wo', 'natural', 'siguiente', 'mejor', 'sharper', 'iterate', 'sanity', 'processed', 'instances', 'similarly', 'orange',
        'eventually', 'appreciated', 'listing', 'picking', 'assume', 'shouldnt', 'headlines', 'shown', 'confidence', 'failure', 'prefer', 'ty', 'took', 'feeling', 'suppose', 'putting', 'yellow', 'extend', 'totally', 'enjoy',
        'dong', 'qe', 'eye', 'major', 'pasted', 'stand', 'addition', 'matched', 'michael', 'matches', 'separately', 'descriptive', 'ny', 'emojis', 'rec', 'reading', 'redraft', 'poster', 'assistance', 'tree', 'represent', 'peux', 'dans', 'ko', 'ka', 'unclear', 'appreciation', 'mandatory', 'replacement', 'managers', 'sports', 'reported', 'xxx', 'xxxx', 'stacked', 'radar', 'serviced',
        'evaluated', 'grammer', 'displayed', 'cup', 'eligible', 'evening', 'ending', 'groups', 'pp', 'chargeout', 'square', 'organize', 'head', 'hmm', 'pasting', 'includes', 'minimum', 'mismo', 'compartir', 'firm', 'learned', 'tema', 'espero', 'creation', 'managed', 'traffic', 'mistakes', 'developers', 'clicking', 'dropdown', 'interviewer', 'recruiter',
        'pays', 'fifth', 'populate', 'games', 'data_risk', 'flip', 'ideal', 'sk', 'vl', 'facing', 'leading', 'parts', 'attend', 'enter', 'headings', 'carry', 'problems', 'tan', 'worry', 'scratch', 'finish', 'certain', 'leaves', 'agreed', 'looping', 'am', 'improved', 'declined', 'tue', 'wed', 'br', 'supp', 'cid', 'prioritize', 'unnecessary', 'subjects', 'diego',
        'qu', 's_qu', 'informaci', '_informaci', 'data_code', 'anymore', 'restart', 'manner', 'precise', 'recheck', 'bn', 'quite', 'approve', 'breakdown', 'official', 'opt', 'ops', 'captured', 'eu', 'dos', 'produce', 'mn', 'apps', 'achieve', 'repos', 'solid', 'texts', 'minus', 'poder', 'confirming', 'receipt', 'simply', 'stable', 'remainder', 'verticle',
        'eval', 'advance', 'saved', 'updating', 'requesting', 'cl', 'ly', 'day', 'dat', 'fe', 'compress', 'dt', 'accuracy', 'confident', 'att', 'som', 'highlights', 'amber', 'cons', 'pros', 'heap', 'restated',
        'repetitive', 'appendix', 'millions', 'twice', 'observation', 'ns', 'av', 'visit', 'walkthrough', 'structured', 'according', 'speech', 'symbol', 'congratulations', 'decided', 'kick', 'knows', 'recap', 'amt', 'noticed', 'milestones', 'consideration', 'pressure', 'qui', 'avec', 'baseline', 'lengthy', 'respective', 'flows', 'cs', 'gc', 'plain', 'modern', 'arial', 'prospective', 'event',
        'capitalization', 'mix', 'responded', 'blocks', 'strengths', 'aren', 'chris', 'james', 'wasn', 'followup', 'comfortable', 'tn', 'ds', 'mais', 'faire', 'comparing', 'bryan', 'folders', 'maintains', 'repetition', 'apt', 'shaded', 'deviations', 'aakash', 'opts',
        'evidenced', 'mo', 'ct', 'contribution', 'ahve', 'aif', 'drag', 'percent', 'understood', 'piece', 'paul', 'assess', 'smart', 'speaker', 'suggesting', 'claim', 'reasons', 'optimize', 'finalize', 'tweak', 'dry', 'banco', 'phrasing', 'successful', 'reset', 'antes', 'nada', 'ts', 'ii', 'attaching', 'validated', 'maria', 'johnson', 'engineers', 'rounds',
        'evolved', 'partially', 'weaknesses', 'kya', 'snapshot', 'dear', 'traduci', 'man', 'valid', 'poner', 'waterfall', 'placeholder', 'fall', 'peer', 'views', 'dar', 'wi', 'sunday', 'train', 'cosa', 'laura', 'letting', 'val', 'honest', 'calling', 'nt', 'spec', 'literally', 'ur', 'workpapers', 'specializes',
        'venta', 'tarjeta', 'pago', 'mas', 'que', 'informacion', 'todos', 'eso', 'esta', 'cada', 'fur', 'sie', 'tenemos', 'español', 'm√°s', 'qu√©', '„äæ', '„äü', 'informaci√≥n', 'ingl√©s', 'cual', 'm√°s_qu√©', 'informaci√≥n_ingl√©s', 'qu√©_cual',
        '_cual', 'n_ingl', 'äæ', 'äü', 'mla', 'customized', 'decrease', 'assuming', 'populated', 'est√°', 'est', 'difficult', 'savings', 'workshop', 'recreate', 'af', 'star', 'takeaways', 'hilton', 'engage', 'purchase', 'justification', 'raq', 'dia', 'lock', 'humble', 'collapse', 'f√°or', 'warm', 'materials', 'selection', 'dame', 'ran', 'introduce', 'asi', 'ng', 'essence', 'divided', 'pet', 'purely', 'printed', 'kartik', 'imported', 'differs', 'arch', 'prb', 'thresholds', 'posible', 'peter', 'jack',
        '_est', 's_poa', 'çπ_éó_é', 'çπ', '_é', '_ç', 'eva', 'hyperlink', 'millennial', 'hooks', 'automatically', 'leads', 'answering', 'usually', 'myca', 'replying', 'craft', 'ace', 'driving', 'nel', 'anche', 'docx', 'prepared', 'behavior', 'opinion', 'spoke', 'sahil', 'tickets', 'sold', 'ramp', 'deeper', 'lista', 'clientes', 'cat', 'rep', 'servicios', 'ans', 'acw', 'cust', 'smooth',
        'äü_arrange', 'arrange', 's_select', 'och', 'followed', 'functionality', 'agency', 'kept', 'fl', 'ill', 'corresponding', 'ff', 'learnings', 'located', 'condensed', 'cold', 'stores', 'originally', 'frequency', 'het', 'resource', 'rsk', 'stronger', 'weak', 'mes', 'mayo', 'interactive', 'accessible', 'fresh', 'completes', 'wider', 'networking',
        'accessed', 'backed', 'figures', 'tips', 'mo_ct_ooo', '„Ç¢_É°_É¨_Ç´_É≥_filled', '_filled', 'patch', 'uno', '„ÄäÁñ≤_ÇåÊßò_Äß_Äô_drivers', '_drivers', 'm√°s_informaci√≥n', 's_informaci', 'hoping', 'closer', 'guidelines', 'picked', 'qu√©_est√°', 'f√°or_ein', 'ein', 'questa', 'connecting', 'plug', 'cleanly', 'adjustment', 'stages', 'saturday', 'november', 'otra', 'vamos', 'ls', 'paper', 'synopsis', 'fetching', 'everyday', 'strict', 'thousands', 'consumed', 'reads', 'retro', 'constructor', 'raus', 'indices', 'passionate', 'lisa', 'kanika', 'objection', 'iloc',
        'çπ_euc', '„Ç®_„Ç≠_„Çπ_Éó_É¨_„Çπ_euc', '„Ç¢_É°_É¨_Ç´_É≥_„Ç®_„Ç≠_„Çπ_Éó_É¨_„Çπ', '„ÄäÁñ≤_ÇåÊßò_Äß_Äô_bps', '_çåêßò_äß_äô_bps', 'signing', 'est√°_informaci√≥n', 'traducir', 'poa', 'waht', 'narrow', 'qu√©_admin', '_admin', 'edited', 'operation', 'hot', 'classes', 'broader', 'confusion', 'facts', 'pain', 'diagrams', 'actionable', 'lose', 'brain', 'correctness', 'compact', 'm√°s_tipo', 's_tipo', 'south', 'attending', 'punto', 'delle', 'ah', 'slice', 'evaluate', 'dhawan_„Ç®_„Ç≠_„Çπ_Éó_É¨_„Çπ', 'dhawan_', 'logging', 'stick', '_çåêßò_äß_äô_drivers', 'or_ein', 'ääáñ',
        'çπ_eusw', '„Ç®_„Ç≠_„Çπ_Éó_É¨_„Çπ_eusw', 'äü_slow', '„Äæ„Äü_slow', 's_table', 'm√°s_table', 'mo_ct_ia', 'or_ps', 'f√°or_ps', 'as_manera', 'd√°as_manera', 'mandate', 'tighter', 'presenting', 'meaningful', 'additionally', 'drivers', 'sz', 'comprehensive', 'stating', 'earliest', 'hallucinate', 'everytime', 'flat', 'art', 'otro', 'lado', 'enviar', 'pregunta', 'hearing', 'coordinating', 'necesario', 'correcto',
        'ignored', 'vic', 'honestly', 'offered', 'filed', 'arent', 'copying', 'wht', 'responding', 'practice', 'paso', 'tiempo', 'cambios', 'rerun', 'pace', 'km', 'nicer', 'caroline', 'kids', 'tea',
        'äü_', 'est', '_voy', 'mo_ct_shape', 'sh_ea', 'puedo_d', 'blend_ma', 'ana', '_f', 'ingl', 's_frase',
        'or_wie', 'n_entonces', 'maddie_', 'regular', 'chinese', 'forgot', 'begin', 'puedo', 'shape', 'plz', 'remarks', 'estimate', 'voy', 'ads', 'popup', 'worst', 'fue', 'mensaje', 'kumar', 'tienen', 'tie', 'explicitly', 'designed', 'elle', 'scripted',
        'çπ_ev', 'invent_', 'doubt', 'downloaded', 'm√°s_qu√©', 'm√', 's_qu', 'qu√©', 'm√°s', 'appointment', 'broad', 'rc_singh', 'rc', 'singh', 'estimated', 'guy', 'messaging', 'smaller', 'broken', 'ended', 'mo_ct_ia', 'mo_ct', 'ia', 'lay', 'readme', 'callout', 'breaking', 'entre', 'sus', 'tipo', 'hemos', 'alguna', 'manera', 'presented', 'depth', 'significant', 'ks',
        'triggerdagrunoperator_', '„Äæ„Äü_„Ç¢_É°_É¨_Ç´_É≥', 'n_f', 'sharma_pi', 'isnull_', 'ther', 'hook', 'weather', 'onwards', 'tabla', 'sell', 'wouldn', 'efficient', 'li_ex', 'li', 'ex', 'disconnected', 'directions', 'grupo', 'worse', 'fa_fc', 'fa', 'fc', 'formato', 'mucho', 'tutti', 'italia',
        'october', 'nuevo_replied', 'nuevo', 'replied', 'sharma_pulling', 'sharma', 'pulling', 'highest_increased', 'highest', 'increased', 'stopped_responsibility', 'stopped', 'responsibility', 'd√≠as_buenos', 'd', 'as_buenos', 'buenos', 'expecting_jessica', 'expecting', 'jessica',
        'soft_dropped', 'soft', 'dropped', 'youre_chnages', 'youre', 'chnages', 'est√°_ingl√©s', 'est', 'ingl', 'ellos_worries', 'ellos', 'worries', 'podemos_estas', 'podemos', 'estas', 'resumen_rephase', 'resumen', 'rephase'
    ]
    extended_stop_words = list(ENGLISH_STOP_WORDS) + custom_noise
    
    # We fit TFIDF on the corpus containing BOTH the taxonomy and the user prompts
    vectorizer = TfidfVectorizer(max_features=10000, stop_words=extended_stop_words, ngram_range=(1, 2))
    
    corpus = tax_df["processed_desc"].tolist() + df["processed_text"].tolist()
    tfidf_all = vectorizer.fit_transform(corpus)
    
    X_tax = tfidf_all[:len(tax_df)]
    X_prompts = tfidf_all[len(tax_df):]
    
    print("\n[Phase 3/4] Mathematical Routing (Cosine Similarity)")
    print("  -> Computing cosine distances against the official taxonomies...")
    sim_matrix = cosine_similarity(X_prompts, X_tax)
    
    max_sims = sim_matrix.max(axis=1)
    best_tax_idx = sim_matrix.argmax(axis=1)
    
    # Assign Official Categories
    df["category_id"] = -1
    df["category_name"] = ""
    df["subcategory_id"] = -1
    df["subcategory_name"] = ""
    
    # If the text shares virtually no words with ANY taxonomy, route to Other
    THRESHOLD = 0.08
    official_mask = max_sims >= THRESHOLD
    other_mask = max_sims < THRESHOLD
    
    official_count = official_mask.sum()
    other_count = other_mask.sum()
    print(f"  -> {official_count:,} prompts successfully matched an official taxonomy!")
    print(f"  -> {other_count:,} prompts had zero similarity and were routed to 'Other/Miscellaneous'.")
    
    # Map Official Prompts
    official_indices = np.where(official_mask)[0]
    matched_tax_rows = tax_df.iloc[best_tax_idx[official_indices]]
    
    df.loc[official_indices, "category_id"] = matched_tax_rows["category_id"].values
    df.loc[official_indices, "category_name"] = matched_tax_rows["category_name"].values
    df.loc[official_indices, "subcategory_id"] = matched_tax_rows["subcategory_id"].values
    df.loc[official_indices, "subcategory_name"] = matched_tax_rows["subcategory_name"].values
    
    hybrid_taxonomy_mapping = tax_df[["category_id", "category_name", "subcategory_id", "subcategory_name"]].to_dict('records')
    
    print("\n[Phase 4/4] Recursive Auto-Discovery for 'Other/Miscellaneous' Fallback Bucket")
    if other_count > 0:
        MAX_CLUSTER_SIZE = 10000
        
        max_cat_id = tax_df["category_id"].max()
        max_sub_id = tax_df["subcategory_id"].max()
        
        other_cat_id = max_cat_id + 1
        
        df.loc[other_mask, "category_id"] = other_cat_id
        df.loc[other_mask, "category_name"] = "Other/Miscellaneous"
        
        other_indices = np.where(other_mask)[0]
        X_other = X_prompts[other_indices]
        
        # Base Dimensionality Reduction
        svd = TruncatedSVD(n_components=min(150, other_count - 1), random_state=42)
        X_reduced_other = svd.fit_transform(X_other)
        
        n_clusters_other = min(20, other_count)
        print(f"  -> Step 1: Initial grouping into {n_clusters_other} baseline clusters...")
        
        kmeans = MiniBatchKMeans(n_clusters=n_clusters_other, random_state=42, batch_size=min(10000, other_count), n_init='auto')
        base_other_labels = kmeans.fit_predict(X_reduced_other)
        
        current_global_sub_id = max_sub_id + 1
        
        for local_id in range(n_clusters_other):
            sub_mask = base_other_labels == local_id
            cluster_size = sub_mask.sum()
            if cluster_size == 0:
                continue
                
            cluster_X = X_other[sub_mask]
            cluster_global_indices = other_indices[sub_mask]
            
            if cluster_size > MAX_CLUSTER_SIZE:
                # Shatter the mega-cluster
                shatter_count = min(math.ceil(cluster_size / MAX_CLUSTER_SIZE), cluster_size)
                print(f"    -> Shattering Mega-Cluster {local_id} ({cluster_size:,} prompts) into {shatter_count} micro-clusters...")
                
                # Reduce dims specifically for this dense mega-cluster
                svd_micro = TruncatedSVD(n_components=min(150, cluster_size - 1), random_state=42)
                X_reduced_micro = svd_micro.fit_transform(cluster_X)
                
                kmeans_micro = MiniBatchKMeans(n_clusters=shatter_count, random_state=42, batch_size=min(5000, cluster_size), n_init='auto')
                micro_labels = kmeans_micro.fit_predict(X_reduced_micro)
                
                for m_id in range(shatter_count):
                    m_mask = micro_labels == m_id
                    m_size = m_mask.sum()
                    if m_size == 0:
                        continue
                        
                    m_indices = cluster_global_indices[m_mask]
                    
                    if m_size >= 2:
                        top_keywords = extract_top_keywords(X_prompts[m_indices], vec, n_keywords=2)
                        sub_name = "_".join(top_keywords) if top_keywords else f"auto_micro_{current_global_sub_id}"
                    else:
                        sub_name = f"auto_micro_{current_global_sub_id}"
                        
                    df.loc[m_indices, "subcategory_id"] = current_global_sub_id
                    df.loc[m_indices, "subcategory_name"] = sub_name
                    
                    hybrid_taxonomy_mapping.append({
                        "category_id": other_cat_id,
                        "category_name": "Other/Miscellaneous",
                        "subcategory_id": current_global_sub_id,
                        "subcategory_name": sub_name
                    })
                    current_global_sub_id += 1
            else:
                if cluster_size >= 2:
                    top_keywords = extract_top_keywords(cluster_X, vec, n_keywords=2)
                    sub_name = "_".join(top_keywords) if top_keywords else f"auto_{current_global_sub_id}"
                else:
                    sub_name = f"auto_{current_global_sub_id}"
                    
                df.loc[cluster_global_indices, "subcategory_id"] = current_global_sub_id
                df.loc[cluster_global_indices, "subcategory_name"] = sub_name
                
                hybrid_taxonomy_mapping.append({
                    "category_id": other_cat_id,
                    "category_name": "Other/Miscellaneous",
                    "subcategory_id": current_global_sub_id,
                    "subcategory_name": sub_name
                })
                current_global_sub_id += 1
    print("\n[Exporting Final Hybrid Automations]")
    taxonomy_df = pd.DataFrame(hybrid_taxonomy_mapping)
    tax_path = os.path.join(data_dir, "hybrid_taxonomy_mapping.csv")
    taxonomy_df.to_csv(tax_path, index=False)
    
    full_path = os.path.join(data_dir, "fully_categorized_dataset.csv")
    df[["raw_text", "truncated_prompt", "category_id", "category_name", "subcategory_id", "subcategory_name"]].to_csv(full_path, index=False)
    
    print(f"\nDone! Automatically matched / generated {len(taxonomy_df)} taxonomy combinations.")
    print(f"1. Hybrid Taxonomy Dictionary saved to {tax_path}")
    print(f"2. Fully Mapped Training Dataset saved to {full_path}")

if __name__ == "__main__":
    run_hybrid_discovery()
