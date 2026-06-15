from sklearn.feature_extraction.text import TfidfVectorizer
import json

analyzer = TfidfVectorizer().build_analyzer()
results = {
    'triggerdagrunoperator_„Ç®_„Ç≠_„Çπ_Éó_É¨_„Çπ': analyzer('triggerdagrunoperator_„Ç®_„Ç≠_„Çπ_Éó_É¨_„Çπ'),
    '„Äæ„Äü_„Ç¢_É°_É¨_Ç´_É≥': analyzer('„Äæ„Äü_„Ç¢_É°_É¨_Ç´_É≥'),
    'informaci√≥n_f√°or': analyzer('informaci√≥n_f√°or'),
    'sharma_pi√π': analyzer('sharma_pi√π'),
    'invent_„Ç®_„Ç≠_„Çπ_Éó_É¨_„Çπ': analyzer('invent_„Ç®_„Ç≠_„Çπ_Éó_É¨_„Çπ'),
    'isnull_„Ç®_„Ç≠_„Çπ_Éó_É¨_„Çπ': analyzer('isnull_„Ç®_„Ç≠_„Çπ_Éó_É¨_„Çπ')
}

with open('scratch/tokenize_out10.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
