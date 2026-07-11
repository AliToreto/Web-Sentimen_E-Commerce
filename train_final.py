"""
Training pipeline final - Analisis Sentimen Review E-Commerce
================================================================
Menangani dataset bilingual (Inggris + Indonesia campuran / code-switched),
yang umum ditemukan pada review Shopee/Tokopedia.

Model: Logistic Regression + TF-IDF (satu model, dipilih karena ringan,
cepat dijalankan di laptop biasa tanpa GPU, dan akurasinya solid untuk
kasus ini). Alternatif seperti XLM-RoBERTa butuh GPU dan download model
besar dari Hugging Face yang tidak praktis untuk deployment ringan ini.
"""

import pandas as pd
import re
import json
import pickle

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords as nltk_stopwords

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score

print("MEMULAI TRAINING MODEL — Logistic Regression + TF-IDF Bilingual")
print("=" * 60)

# =====================================================
# STEMMER (Indonesia) & STOPWORDS (ID + EN gabungan)
# =====================================================

factory = StemmerFactory()
stemmer_id = factory.create_stemmer()

# Cache manual untuk stemming — Sastrawi tidak nge-cache secara default,
# padahal kata yang sama (mis. "bagus", "produk", "yang") muncul ribuan
# kali di 2000 review.
_stem_cache = {}

# Sastrawi adalah stemmer KHUSUS Bahasa Indonesia. Kalau dikasih kata
# Inggris ("product", "very", "good"), ia tetap mencoba SEMUA kombinasi
# imbuhan Indonesia (me-, di-, ter-, ber-, per-, ke-, se- / -kan, -an, -i,
# -lah, -nya) sebelum menyerah — ini bisa makan ~100ms PER KATA. Karena
# dataset ini ~97% berbahasa Inggris, hampir semua kata kena jalur lambat
# ini kalau tidak disaring dulu. Heuristik ini melewatkan kata yang jelas
# tidak match pola imbuhan Indonesia sama sekali, tanpa memanggil Sastrawi.
_ID_PREFIXES = ('me', 'di', 'ter', 'ber', 'per', 'pe', 'ke', 'se')
_ID_SUFFIXES = ('kan', 'an', 'i', 'lah', 'kah', 'nya', 'ku', 'mu')

def _might_be_indonesian(word):
    if len(word) < 4:
        return False
    return word.startswith(_ID_PREFIXES) or word.endswith(_ID_SUFFIXES)

def cached_stem(word):
    if word not in _stem_cache:
        if _might_be_indonesian(word):
            _stem_cache[word] = stemmer_id.stem(word)
        else:
            _stem_cache[word] = word
    return _stem_cache[word]

stop_factory = StopWordRemoverFactory()
stopwords_id = set(stop_factory.get_stop_words())
stopwords_en = set(nltk_stopwords.words('english'))

# Gabungkan stopword dua bahasa
stopwords_all = stopwords_id | stopwords_en

# =====================================================
# SLANG DICTIONARY (Indonesia, diperluas)
# =====================================================

slang_dict = {
    "yg": "yang", "bgt": "banget", "ga": "tidak", "gak": "tidak",
    "gk": "tidak", "tdk": "tidak", "jlk": "jelek", "lmbt": "lambat",
    "brg": "barang", "kcewa": "kecewa", "ancur": "hancur", "mantul": "mantap",
    "dgn": "dengan", "dg": "dengan", "pengiriman": "pengiriman",
    "tp": "tapi", "krn": "karena", "karna": "karena", "udh": "sudah",
    "udah": "sudah", "sdh": "sudah", "blm": "belum", "blum": "belum",
    "sy": "saya", "gw": "saya", "gue": "saya", "lu": "kamu", "lo": "kamu",
    "trs": "terus", "trus": "terus", "dr": "dari", "jd": "jadi",
    "jgn": "jangan", "bs": "bisa", "bsa": "bisa", "skrg": "sekarang",
    "tll": "terlalu", "bnyk": "banyak", "byk": "banyak", "tggl": "tinggal",
    "respn": "respon", "oonline": "online", "malowbat": "baterai lemah",
    "low bat": "baterai lemah",
}

# Kata negasi yang TIDAK boleh dihapus oleh stopword removal,
# karena penting untuk makna sentimen ("not good" / "tidak bagus")
NEGATION_WHITELIST = {
    "not", "no", "never", "tidak", "bukan", "jangan", "belum", "kurang"
}
stopwords_all = stopwords_all - NEGATION_WHITELIST

# =====================================================
# CLEANING FUNCTION (bilingual)
# =====================================================

def clean_text(text):
    text = str(text).lower()

    # hapus url
    text = re.sub(r'http\S+|www\S+', ' ', text)

    # normalisasi karakter berulang (cth: "lucuuuu" -> "lucu", "makasihhhhh" -> "makasih")
    text = re.sub(r'(.)\1{2,}', r'\1', text)

    # hapus simbol/angka, sisakan huruf & spasi
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    # hapus spasi berlebih
    text = re.sub(r'\s+', ' ', text).strip()

    words = text.split()

    # slang replacement
    words = [slang_dict.get(w, w) for w in words]

    # stopword removal (gabungan ID + EN, minus negasi)
    words = [w for w in words if w not in stopwords_all]

    # stemming bahasa Indonesia (kata Inggris umumnya tidak terpengaruh signifikan,
    # tapi tetap dilewatkan supaya konsisten dengan kata Indonesia campuran)
    words = [cached_stem(w) for w in words]

    # buang kata kosong hasil stemming/sisa 1 huruf
    words = [w for w in words if len(w) > 1]

    return ' '.join(words)

# =====================================================
# LEXICON AUGMENTATION
# =====================================================
# Dataset asli 98% berbahasa Inggris — banyak kata sentimen Indonesia
# yang krusial (jelek, rusak, dst) muncul 0-1 kali sehingga terbuang
# oleh ambang batas min_df pada TF-IDF. Blok ini menambahkan contoh
# sintetis singkat untuk memastikan kata-kata kunci tersebut punya
# representasi yang cukup di vocabulary, TANPA menggantikan data asli
# (porsi sintetis dijaga kecil, hanya suplemen).

NEGATIVE_LEXICON = [
    "jelek", "rusak", "buruk", "kecewa", "mengecewakan", "cacat", "hancur",
    "parah", "lambat", "lemot", "tipu", "penipuan", "palsu", "kotor",
    "robek", "sobek", "pecah", "bocor", "basi", "kasar", "kw", "murahan",
]
POSITIVE_LEXICON = [
    "bagus", "mantap", "memuaskan", "puas", "cepat", "rapi", "ramah",
    "recommended", "oke", "keren", "sesuai", "awet", "kuat", "nyaman",
    "wangi", "original", "lengkap", "aman", "worth", "cakep",
]
TEMPLATES = [
    "barang {w}", "produk {w}", "kualitas {w}", "pelayanan {w}",
    "pengiriman {w}", "{w} banget", "{w} sekali", "barangnya {w}",
]

def build_lexicon_augmentation():
    rows = []
    for w in NEGATIVE_LEXICON:
        for t in TEMPLATES:
            rows.append({"review": t.format(w=w), "sentimen": "negatif"})
    for w in POSITIVE_LEXICON:
        for t in TEMPLATES:
            rows.append({"review": t.format(w=w), "sentimen": "positif"})
    return pd.DataFrame(rows)

# =====================================================
# LOAD DATASET
# =====================================================

df = pd.read_csv("dataset_siap_pakai.csv")
print(f"\nData awal (SEMUA baris dari CSV): {len(df)} baris")

df = df.dropna(subset=['review', 'sentimen'])
df = df[df['sentimen'].isin(['positif', 'negatif'])]

print(f"Setelah buang baris kosong (SEMUA baris valid dipakai, TIDAK ada yang di-drop karena duplikat): {len(df)} baris")
print(df['sentimen'].value_counts())

# =====================================================
# PREPROCESSING ULANG (review_clean lama dibuat dengan
# tool yang salah-bahasa, jadi kita generate ulang dari 'review' asli)
# =====================================================

print("\nMembersihkan teks (bilingual)...")
df['review_clean'] = df['review'].apply(clean_text)
df = df[df['review_clean'].str.strip() != '']
print(f"Setelah buang teks kosong: {len(df)} baris data ASLI")

# Gabungkan dengan augmentasi lexicon (porsi kecil, hanya suplemen
# supaya kata sentimen Indonesia penting tidak hilang dari vocabulary)
df_aug = build_lexicon_augmentation()
df_aug['review_clean'] = df_aug['review'].apply(clean_text)
n_real = len(df)
n_aug = len(df_aug)
df = pd.concat([df, df_aug], ignore_index=True)
print(f"Data asli: {n_real} baris, augmentasi lexicon: {n_aug} baris ({n_aug/(n_real+n_aug)*100:.1f}% dari total)")
print(f"Total data untuk training: {len(df)} baris")

# =====================================================
# TF-IDF
# =====================================================

tfidf = TfidfVectorizer(
    max_features=6000,
    ngram_range=(1, 2),
    min_df=1,
    max_df=0.9
)

X_all = tfidf.fit_transform(df['review_clean'])
y_all = df['sentimen']

# =====================================================
# SPLIT DATA — semua baris (termasuk duplikat) tetap dipakai
# untuk TRAINING, tapi test set disusun dari review UNIK saja
# supaya tidak ada baris yang sama persis muncul di train & test
# sekaligus (data leakage yang bisa menggelembungkan akurasi palsu).
# =====================================================

df_unique = df.drop_duplicates(subset='review')
train_reviews, test_reviews = train_test_split(
    df_unique['review'], test_size=0.2, random_state=42,
    stratify=df_unique['sentimen']
)
train_review_set = set(train_reviews)
test_review_set = set(test_reviews)

train_mask = df['review'].isin(train_review_set)
# test hanya 1 salinan per review unik (hindari baris kembar di test)
test_mask = df['review'].isin(test_review_set) & (~df.duplicated(subset='review'))

X_train, y_train = X_all[train_mask.values], y_all[train_mask.values]
X_test, y_test = X_all[test_mask.values], y_all[test_mask.values]

print(f"\nTotal baris dipakai untuk TRAIN (termasuk duplikat & augmentasi): {X_train.shape[0]}")
print(f"Total baris dipakai untuk TEST (review unik, tanpa duplikat): {X_test.shape[0]}")

# =====================================================
# TRAINING — LOGISTIC REGRESSION (SATU MODEL)
# =====================================================
# class_weight='balanced' menangani ketidakseimbangan kelas otomatis
# (walau dataset ini sudah 1000/1000, ini menjaga training tetap aman
# kalau nanti dataset diganti dengan proporsi positif/negatif yang timpang).

model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average='macro', pos_label='positif')
report = classification_report(y_test, y_pred, output_dict=True)

print(f"\n{'=' * 60}")
print("MODEL: Logistic Regression")
print(f"Akurasi : {accuracy * 100:.2f}%")
print(f"F1-macro: {f1_macro * 100:.2f}%")
print(classification_report(y_test, y_pred))
print("=" * 60)

# =====================================================
# SIMPAN MODEL + VECTORIZER + METADATA
# =====================================================

pickle.dump(model, open("model_sentimen.pkl", "wb"))
pickle.dump(tfidf, open("tfidf_vectorizer.pkl", "wb"))

metadata = {
    "model": "logistic_regression",
    "accuracy": accuracy,
    "f1_macro": f1_macro,
    "n_samples_total": len(df),
    "n_train": int(X_train.shape[0]),
    "n_test": int(X_test.shape[0]),
    "labels": sorted(y_all.unique().tolist()),
}
with open("model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

# Simpan juga dataset yang sudah dibersihkan ulang (untuk dashboard/EDA di web app)
df[['review', 'rating', 'sentimen', 'review_clean']].to_csv("dataset_final_clean.csv", index=False)

print("\nMODEL, VECTORIZER, METADATA, DAN DATASET BERSIH BERHASIL DISIMPAN")
