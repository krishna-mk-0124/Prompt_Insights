import pandas as pd
with open('data/english_prompts.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
c1 = sum(1 for l in lines if 'triggerdagrunoperator' in l.lower())
c2 = sum(1 for l in lines if any(k in l.lower() for k in ['„', 'ç', '√', 'é', '¨', 'π']))
print('trigger count:', c1)
print('mojibake count:', c2)
