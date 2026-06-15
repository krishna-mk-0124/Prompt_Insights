import re

with open("ITERATION_LOG.md", "r", encoding="utf-8") as f:
    text = f.read()

# Fix capitalization and grammar
text = text.replace(" the AI ", " The AI ")
text = text.replace(" the AI's ", " The AI's ")
text = text.replace(" The AI admitted I do not have ", " The AI admitted it did not have ")
text = text.replace(" per my directive, I reverted ", " per my directive, The AI reverted ")
text = text.replace(" I correctly challenged my language_router.py ", " I correctly challenged The AI's language_router.py ")
text = text.replace(" The AI investigated and found The AI had hardcoded ", " The AI investigated and found it had hardcoded ")
text = text.replace(". The AI ", ". The AI ")
text = text.replace("- The AI ", "- The AI ")
text = text.replace("I do not have", "it did not have")

# Some were at start of sentence but lowercase because they followed a period and space
text = re.sub(r'\.\s*the AI\b', '. The AI', text)
text = re.sub(r'\.\s*the AI\'s\b', '. The AI\'s', text)
text = re.sub(r'\|\s*the AI\b', '| The AI', text)

with open("ITERATION_LOG.md", "w", encoding="utf-8") as f:
    f.write(text)

print("Grammar fixes complete.")
