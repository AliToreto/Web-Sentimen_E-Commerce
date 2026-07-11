"""
Lexicon sentimen hybrid — Indonesia + Inggris
================================================
Menopang model ML (TF-IDF + Naive Bayes) untuk kata-kata sentimen kuat
yang jarang muncul di data training (misal kata Indonesia "jelek", "rusak"
yang aslinya cuma muncul <5 kali di 2000 data).

Skor lexicon di-blend dengan probabilitas ML: bobot lexicon membesar kalau
teks pendek dan didominasi kata dari lexicon ini, dan mengecil kalau teks
panjang/kompleks (di mana model ML lebih bisa diandalkan).
"""

import re

POSITIVE_WORDS = {
    # Indonesia
    "bagus", "baik", "mantap", "mantul", "puas", "memuaskan", "sesuai",
    "cepat", "rapi", "ramah", "rekomendasi", "oke", "ok", "joss", "top",
    "cantik", "awet", "murah", "original", "worth", "aman", "lengkap",
    "sempurna", "nyaman", "berkualitas", "asli", "sip", "gercep",
    # English
    "good", "great", "nice", "excellent", "awesome", "love", "loved",
    "perfect", "fast", "satisfied", "satisfying", "recommended", "amazing",
    "happy", "best", "wonderful", "beautiful", "comfortable", "sturdy",
    "durable", "quick", "smooth", "fits", "fit", "cute",
}

NEGATIVE_WORDS = {
    # Indonesia
    "jelek", "buruk", "rusak", "cacat", "hancur", "kecewa",
    "mengecewakan", "lambat", "lama", "mahal", "tipu", "penipu", "bohong",
    "sampah", "parah", "gagal", "robek", "sobek", "kotor", "bocor",
    "pecah", "patah", "palsu", "kw", "replika", "salah", "jauh",
    "berbeda", "beda", "susah", "ribet", "lecet", "kusam", "luntur",
    # English
    "bad", "terrible", "awful", "broken", "damaged", "disappointed",
    "disappointing", "poor", "worst", "horrible", "defective", "late",
    "slow", "scam", "fake", "waste", "useless", "faulty", "torn",
    "cracked", "wrong", "refund", "complaint", "complain",
    "smell", "stained", "wrinkled", "moldy",
}

NEGATORS = {"tidak", "tdk", "ga", "gak", "gk", "bukan", "jangan", "not", "no", "never", "kurang"}


def lexicon_score(cleaned_text: str):
    """
    Hitung skor lexicon dari teks yang SUDAH dibersihkan (hasil clean_text()).
    Mengembalikan (net_score, hits, total_tokens).
    net_score positif -> condong positif, negatif -> condong negatif.
    Kata langsung didahului negator akan dibalik polaritasnya.
    """
    tokens = cleaned_text.split()
    net_score = 0
    hits = 0

    for i, tok in enumerate(tokens):
        polarity = 0
        if tok in POSITIVE_WORDS:
            polarity = 1
        elif tok in NEGATIVE_WORDS:
            polarity = -1
        else:
            continue

        prev_is_negator = (
            (i > 0 and tokens[i - 1] in NEGATORS)
            or (i > 1 and tokens[i - 2] in NEGATORS)
        )
        if prev_is_negator:
            polarity = -polarity

        net_score += polarity
        hits += 1

    return net_score, hits, len(tokens)


def lexicon_proba_positif(net_score: float) -> float:
    """Ubah net_score jadi probabilitas positif (0..1) pakai fungsi sigmoid."""
    import math
    return 1 / (1 + math.exp(-net_score))


def blend_with_ml(ml_proba_positif: float, cleaned_text: str, max_lexicon_weight: float = 0.7):
    """
    Gabungkan probabilitas ML dengan skor lexicon.

    Bobot lexicon (1-alpha) membesar sebanding dengan seberapa besar porsi
    teks yang tersusun dari kata-kata di lexicon (coverage), dibatasi
    maksimum `max_lexicon_weight` supaya model ML tetap punya suara.

    Return: (final_proba_positif, detail_dict)
    """
    net_score, hits, total_tokens = lexicon_score(cleaned_text)

    if hits == 0 or total_tokens == 0:
        return ml_proba_positif, {
            "lexicon_hits": 0, "lexicon_net_score": 0, "lexicon_weight": 0.0,
        }

    coverage = hits / total_tokens
    lexicon_weight = min(max_lexicon_weight, coverage)
    alpha = 1 - lexicon_weight  # bobot untuk ML

    lex_proba = lexicon_proba_positif(net_score)
    final_proba = alpha * ml_proba_positif + lexicon_weight * lex_proba

    return final_proba, {
        "lexicon_hits": hits,
        "lexicon_net_score": net_score,
        "lexicon_weight": round(lexicon_weight, 3),
    }
