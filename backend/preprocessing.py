"""
Preprocessing module - HARUS identik dengan logika di train_final.py
agar prediksi konsisten dengan model yang sudah dilatih.
"""

import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import nltk

nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords as nltk_stopwords

_factory = StemmerFactory()
_stemmer_id = _factory.create_stemmer()

# Cache manual + heuristik skip — lihat catatan lengkap di train_final.py.
# Tanpa ini, Sastrawi mencoba semua kombinasi imbuhan Indonesia untuk
# SETIAP kata Inggris (bisa ~100ms/kata) — penting terutama saat upload
# CSV berisi ratusan/ribuan baris sekaligus.
_stem_cache = {}
_ID_PREFIXES = ('me', 'di', 'ter', 'ber', 'per', 'pe', 'ke', 'se')
_ID_SUFFIXES = ('kan', 'an', 'i', 'lah', 'kah', 'nya', 'ku', 'mu')

def _might_be_indonesian(word):
    if len(word) < 4:
        return False
    return word.startswith(_ID_PREFIXES) or word.endswith(_ID_SUFFIXES)

def _cached_stem(word):
    if word not in _stem_cache:
        if _might_be_indonesian(word):
            _stem_cache[word] = _stemmer_id.stem(word)
        else:
            _stem_cache[word] = word
    return _stem_cache[word]

_stop_factory = StopWordRemoverFactory()
_stopwords_id = set(_stop_factory.get_stop_words())
_stopwords_en = set(nltk_stopwords.words('english'))
_stopwords_all = _stopwords_id | _stopwords_en

_slang_dict = {
    "yg": "yang", "bgt": "banget", "ga": "tidak", "gak": "tidak",
    "gk": "tidak", "tdk": "tidak", "jlk": "jelek", "lmbt": "lambat",
    "brg": "barang", "kcewa": "kecewa", "ancur": "hancur", "mantul": "mantap",
    "dgn": "dengan", "dg": "dengan",
    "tp": "tapi", "krn": "karena", "karna": "karena", "udh": "sudah",
    "udah": "sudah", "sdh": "sudah", "blm": "belum", "blum": "belum",
    "sy": "saya", "gw": "saya", "gue": "saya", "lu": "kamu", "lo": "kamu",
    "trs": "terus", "trus": "terus", "dr": "dari", "jd": "jadi",
    "jgn": "jangan", "bs": "bisa", "bsa": "bisa", "skrg": "sekarang",
    "tll": "terlalu", "bnyk": "banyak", "byk": "banyak", "tggl": "tinggal",
    "respn": "respon", "oonline": "online", "malowbat": "baterai lemah",
}

_NEGATION_WHITELIST = {
    "not", "no", "never", "tidak", "bukan", "jangan", "belum", "kurang"
}
_stopwords_all = _stopwords_all - _NEGATION_WHITELIST


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    words = text.split()
    words = [_slang_dict.get(w, w) for w in words]
    words = [w for w in words if w not in _stopwords_all]
    words = [_cached_stem(w) for w in words]
    words = [w for w in words if len(w) > 1]

    return ' '.join(words)
