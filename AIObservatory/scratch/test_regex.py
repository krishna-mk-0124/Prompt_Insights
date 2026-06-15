import re
import json

pattern = re.compile(r'(?u)\b\w\w+\b')
text = 'm√°s_qu√©'
tokens = pattern.findall(text)

with open('scratch/test_regex_out.json', 'w', encoding='utf-8') as f:
    json.dump(tokens, f, ensure_ascii=False)
