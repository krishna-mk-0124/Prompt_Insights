from sklearn.feature_extraction.text import TfidfVectorizer
import sys

def test_tokenizer():
    # Simulate exactly what the pipeline does
    custom_noise = ['äæ', 'äü_', '_é', '_ç', 'çπ_éó_é', 'çπ_ev', 's_qu', 'm√°s_qu√©', '„Äæ„Äü_„Ç¢_É°_É¨_Ç´_É≥', '„Ç®_„Ç≠_„Çπ_Éó_É¨_„Çπ_ev']
    vectorizer = TfidfVectorizer(stop_words=custom_noise, ngram_range=(1, 2))
    
    corpus = [
        "this is a test of „Äæ„Äü_„Ç¢_É°_É¨_Ç´_É≥ inside a sentence",
        "another one with „Ç®_„Ç≠_„Çπ_Éó_É¨_„Çπ_ev here",
        "and also m√°s_qu√© works"
    ]
    
    try:
        X = vectorizer.fit_transform(corpus)
        vocab = vectorizer.get_feature_names_out()
        print("Vocabulary surviving stop words:")
        for v in vocab:
            print(f" - {v}")
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    test_tokenizer()
