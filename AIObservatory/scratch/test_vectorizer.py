import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

custom_noise = [
    'çπ_éó_é', 'çπ_ev', 'äæ', 'äü_', '_é', '_ç', 's_qu'
]
extended_stop_words = list(ENGLISH_STOP_WORDS) + custom_noise

vectorizer = TfidfVectorizer(max_features=10000, stop_words=extended_stop_words, ngram_range=(1, 2))

corpus = [
    "„Ç®_„Ç≠_„Çπ_Éó_É¨_„Çπ_ev test",
    "„Äæ„Äü_„Ç¢_É°_É¨_Ç´_É≥ text",
    "m√°s_qu√© more text"
]

X = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names_out())
