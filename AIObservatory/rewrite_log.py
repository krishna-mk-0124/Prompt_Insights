import re

with open("ITERATION_LOG.md", "r", encoding="utf-8") as f:
    lines = f.readlines()

new_lines = []

for line in lines:
    # 1. Perspective changes (The user -> I, I -> The AI)
    # Be careful with "the user" -> "I" depending on case and context
    line = re.sub(r'\b[T]he user correctly pointed out\b', 'I correctly pointed out', line, flags=re.IGNORECASE)
    line = re.sub(r'\b[T]he user correctly noted\b', 'I correctly noted', line, flags=re.IGNORECASE)
    line = re.sub(r'\b[T]he user correctly challenged\b', 'I correctly challenged', line, flags=re.IGNORECASE)
    line = re.sub(r'\b[T]he user pointed out\b', 'I pointed out', line, flags=re.IGNORECASE)
    line = re.sub(r'\b[T]he user\b', 'I', line)
    line = re.sub(r'\buser feedback\b', 'my feedback', line, flags=re.IGNORECASE)
    line = re.sub(r'\buser request\b', 'my request', line, flags=re.IGNORECASE)
    line = re.sub(r'\bper the user\'s directive\b', 'per my directive', line, flags=re.IGNORECASE)
    line = re.sub(r"the user's explicit instruction", "my explicit instruction", line, flags=re.IGNORECASE)
    
    # "I" (AI) to "The AI"
    line = line.replace(" I incorrectly ", " the AI incorrectly ")
    line = line.replace(" I explicitly ", " the AI explicitly ")
    line = line.replace(" I implemented ", " the AI implemented ")
    line = line.replace(" I admitted ", " the AI admitted ")
    line = line.replace(" I performed ", " the AI performed ")
    line = line.replace(" I diagnosed ", " the AI diagnosed ")
    line = line.replace(" I also realized ", " the AI also realized ")
    line = line.replace(" I investigated ", " the AI investigated ")
    line = line.replace(" I had hardcoded ", " the AI had hardcoded ")
    line = line.replace(" I also included ", " the AI also included ")
    line = line.replace(" my language_router", " the AI's language_router")
    line = line.replace(" my Round", " the AI's Round")
    line = line.replace(" my Phase", " the AI's Phase")
    line = line.replace(" my heuristic", " the AI's heuristic")
    line = line.replace(" I explicitly added ", " the AI explicitly added ")
    line = line.replace(" I have completely ", " the AI completely ")
    line = line.replace(" I appended ", " the AI appended ")
    line = line.replace(" I am ", " the AI is ")
    
    # 2. Rephrase Expectations vs Outcomes
    
    if "Optimization Round 56" in line and "Expected gargantuan drop!" in line:
        line = line.replace("Expected gargantuan drop!", "**Expected Outcome**: Gargantuan drop!<br>**Actual Outcome**: Failed and reverted because useless prompts should be dropped entirely, not mapped.")
        
    if "Optimization Round 58" in line and "Massive drop expected" in line:
        line = line.replace("Massive drop expected", "**Expected Outcome**: Massive drop expected.<br>**Actual Outcome**: A massive new 66k cluster of ASCII Spanish noise survived.")
        
    if "Optimization Round 59" in line and "Huge drop in count AND 0 subcategories in Other!" in line:
        line = line.replace("Huge drop in count AND 0 subcategories in Other!", "**Expected Outcome**: Huge drop in count AND 0 subcategories in Other!<br>**Actual Outcome**: Erased valid English clusters and prevented future production discovery.")
        
    if "Optimization Round 61" in line and "100k garbage drop GUARANTEED!" in line:
        line = line.replace("100k garbage drop GUARANTEED!", "**Expected Outcome**: 100k garbage drop GUARANTEED!<br>**Actual Outcome**: Nothing happened because the vectorizer added underscores to the raw text, breaking the regex.")
        
    if "Optimization Round 62" in line and "Phase 4 fixed; 100k garbage finally dropped!" in line:
        line = line.replace("Phase 4 fixed; 100k garbage finally dropped!", "**Expected Outcome**: Phase 4 fixed; 100k garbage finally dropped!<br>**Actual Outcome**: Failed because .split() failed to strip punctuation, hiding garbage tokens.")

    if "Optimization Round 68" in line and "The \"Other\" bucket is now completely empty!" in line:
        line = line.replace("The \"Other\" bucket is now completely empty!", "**Expected Outcome**: The \"Other\" bucket is now completely empty!<br>**Actual Outcome**: Failed because the raw text lacked enough cosine similarity to break the 0.08 threshold.")

    if "Optimization Round 69" in line and "The 'Other' bucket is permanently gone." in line:
        line = line.replace("The 'Other' bucket is permanently gone.", "**Expected Outcome**: The 'Other' bucket is permanently gone.<br>**Actual Outcome**: Failed and reverted because it deleted 140,000 valid English prompts.")
        
    new_lines.append(line)

with open("ITERATION_LOG.md", "w", encoding="utf-8") as f:
    f.writelines(new_lines)
    
print("Rewrite complete.")
