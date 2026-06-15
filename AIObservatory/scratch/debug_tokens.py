import os
from sklearn.feature_extraction.text import TfidfVectorizer

data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
input_file = os.path.join(data_dir, "english_prompts.txt")

with open(input_file, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

vec = TfidfVectorizer()
analyzer = vec.build_analyzer()

found_trigger = False
found_moji = False

for line in lines:
    tokens = analyzer(line)
    if 'triggerdagrunoperator' in tokens and not found_trigger:
        print("FOUND TRIGGER:")
        print(line[:500])
        found_trigger = True
    if any('european' in t and 'ç' in t for t in tokens) and not found_moji:
        print("FOUND MOJI:")
        print(line[:500])
        found_moji = True
    
    if found_trigger and found_moji:
        break
