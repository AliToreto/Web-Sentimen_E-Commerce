"""
ABSA — Analisis Sentimen Berbasis Aspek (rule-based)
======================================================
Catatan desain: ABSA "penuh" biasanya dilatih pakai model Transformer
berlabel per-aspek yang mahal dibuat & di-download (butuh internet ke
Hugging Face + dataset berlabel aspek, tidak tersedia di sini). Sebagai
gantinya, modul ini memakai pendekatan rule-based yang transparan:

1. Pecah review jadi klausa (dipisah tanda baca & kata kontras seperti
   "tapi", "tetapi", "but") supaya sentimen tiap bagian tidak tercampur.
2. Deteksi aspek yang disebut di tiap klausa lewat daftar kata kunci.
3. Hitung skor sentimen lokal klausa itu pakai lexicon yang sama dengan
   lexicon.py, lalu attribusikan ke aspek yang terdeteksi.

Ini bukan pengganti model ML utama — ini lapisan tambahan yang menjawab
pertanyaan "bagian mana dari review yang dipuji/dikeluhkan", cocok untuk
kasus umum di review e-commerce ("barang jelek tapi pengiriman cepat").
"""

import re
from lexicon import POSITIVE_WORDS, NEGATIVE_WORDS, NEGATORS
from preprocessing import _cached_stem

ASPECT_KEYWORDS_RAW = {
    "Produk": {
        "produk", "barang", "kualitas", "bahan", "warna", "ukuran", "model",
        "desain", "motif", "size", "product", "quality", "item", "material",
        "color", "colour", "design", "fabric", "size",
    },
    "Pengiriman": {
        "kirim", "pengiriman", "kurir", "ekspedisi", "paket", "dikirim",
        "sampai", "dateng", "datang", "delivery", "shipping", "courier",
        "package", "shipment", "arrived", "arrive", "expedition",
    },
    "Harga": {
        "harga", "murah", "mahal", "price", "worth", "value",
    },
    "Pelayanan": {
        "pelayanan", "penjual", "respon", "admin", "cs", "layanan", "seller",
        "service", "response", "customer", "toko", "olshop", "store",
    },
    "Kemasan": {
        "kemasan", "packing", "bungkus", "dus", "packaging", "box",
        "wrapped", "wrapping", "plastik", "bubble",
    },
}

# Token hasil _tokenize_simple() sudah di-stem (mis. "pelayanan" -> "layan",
# "penjual" -> "jual"), jadi daftar keyword aspek juga di-stem di sini saat
# modul di-load, supaya keduanya konsisten dan tidak meleset gara-gara
# bentuk kata yang berbeda (kata dasar vs kata berimbuhan).
ASPECT_KEYWORDS = {
    aspect: {_cached_stem(kw) for kw in keywords}
    for aspect, keywords in ASPECT_KEYWORDS_RAW.items()
}

# Pemisah klausa: tanda baca ATAU kata penghubung yang biasanya
# menandai perubahan sentimen ("bagus TAPI lambat")
CLAUSE_SPLIT_PATTERN = re.compile(
    r'[.,;!?]+|\b(?:tapi|tetapi|namun|sedangkan|meskipun|walaupun|but|however|although)\b',
    re.IGNORECASE,
)


def _tokenize_simple(clause):
    clause = clause.lower()
    clause = re.sub(r'[^a-zA-Z\s]', ' ', clause)
    tokens = clause.split()
    # stem ringan supaya kata berimbuhan ("barangnya" -> "barang") tetap
    # cocok dengan daftar kata kunci aspek & lexicon sentimen
    return [_cached_stem(t) for t in tokens]


def _clause_sentiment(tokens):
    net = 0
    hits = 0
    for i, tok in enumerate(tokens):
        polarity = 0
        if tok in POSITIVE_WORDS:
            polarity = 1
        elif tok in NEGATIVE_WORDS:
            polarity = -1
        else:
            continue

        negated = (
            (i > 0 and tokens[i - 1] in NEGATORS)
            or (i > 1 and tokens[i - 2] in NEGATORS)
        )
        if negated:
            polarity = -polarity

        net += polarity
        hits += 1
    return net, hits


def analyze_aspects(raw_text: str):
    """
    Analisis sentimen per-aspek dari teks review MENTAH (belum melalui
    clean_text() utama, supaya tanda baca & kata kontras masih ada untuk
    pemisahan klausa).

    Return: list of {"aspect": str, "sentiment": "positif"/"negatif"/"netral", "score": int}
    diurutkan dari yang paling signifikan (skor absolut terbesar).
    """
    clauses = CLAUSE_SPLIT_PATTERN.split(str(raw_text))
    aspect_scores = {}

    for clause in clauses:
        if not clause or not clause.strip():
            continue

        tokens = _tokenize_simple(clause)
        if not tokens:
            continue

        detected_aspects = {
            aspect for aspect, keywords in ASPECT_KEYWORDS.items()
            if any(tok in keywords for tok in tokens)
        }
        if not detected_aspects:
            continue

        net, hits = _clause_sentiment(tokens)
        if hits == 0:
            continue

        for aspect in detected_aspects:
            aspect_scores[aspect] = aspect_scores.get(aspect, 0) + net

    results = []
    for aspect, score in aspect_scores.items():
        if score > 0:
            sentiment = "positif"
        elif score < 0:
            sentiment = "negatif"
        else:
            sentiment = "netral"
        results.append({"aspect": aspect, "sentiment": sentiment, "score": score})

    results.sort(key=lambda x: -abs(x["score"]))
    return results
