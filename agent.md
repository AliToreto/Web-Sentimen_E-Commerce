# agent.md — Sentimeter (Analisis Sentimen Review E-Commerce)

Dokumen ini adalah panduan teknis untuk siapapun (manusia atau AI agent)
yang akan membaca, mengubah, atau melanjutkan pengembangan project ini.
Tujuannya: supaya perubahan berikutnya konsisten dengan keputusan desain
yang sudah diambil, dan tidak mengulang kesalahan yang sudah pernah
ditemukan & diperbaiki.

---

## 1. Fokus Sistem

**Apa yang dikerjakan sistem ini:** menerima teks review produk
e-commerce (Bahasa Indonesia / Inggris / campuran keduanya), lalu
mengembalikan **dua lapis informasi**:

1. **Sentimen keseluruhan** — positif atau negatif, beserta tingkat
   keyakinan model (0-100%).
2. **Rincian per aspek (ABSA)** — bagian mana dari review yang dipuji
   dan bagian mana yang dikeluhkan (Produk, Pengiriman, Harga,
   Pelayanan, Kemasan), karena satu review sering memuat sentimen
   campuran ("barang jelek TAPI pengiriman cepat").

**Apa yang SENGAJA di luar cakupan (jangan ditambahkan tanpa alasan kuat):**
- Tidak ada kelas "netral" — hanya positif/negatif (data rating=3 sudah
  dibuang saat sumber data awal dibuat, sebelum masuk `dataset_siap_pakai.csv`).
- Tidak memakai model Transformer (IndoBERT/XLM-RoBERTa/dst). Sudah
  dipertimbangkan dan **ditolak secara sadar** — lihat Section 3.
- Tidak ada autentikasi/multi-user/database persisten. Ini single-tenant,
  stateless, jalan lokal.
- ABSA di sini **rule-based**, bukan model ML terlatih. Ini keputusan
  sadar karena keterbatasan environment (lihat Section 3 & 6).

---

## 2. Alur Kerja (Workflow)

### 2a. Alur training (dijalankan manual, sekali di awal / saat data berubah)

```
dataset_siap_pakai.csv (SEMUA baris, tidak ada sampling)
        │
        ▼
train_final.py
  1. Load & validasi (buang baris tanpa review/label, atau label selain positif/negatif)
  2. clean_text() — cleaning + slang + stopword + stemming (lihat Section 4)
  3. Lexicon augmentation — tambah ±336 baris sintetis pendek
     ("barang jelek", "produk mantap", dst) supaya kata sentimen
     Indonesia yang langka di data asli tetap punya representasi
     di vocabulary TF-IDF (lihat Section 3, "Kenapa augmentasi lexicon").
  4. TfidfVectorizer — ubah teks jadi vektor angka
  5. Split train/test berbasis REVIEW UNIK (hindari data leakage dari
     duplikat/augmentasi yang nyasar ke test set)
  6. LogisticRegression(class_weight='balanced') — fit
  7. Simpan: model_sentimen.pkl, tfidf_vectorizer.pkl, model_metadata.json,
     dataset_final_clean.csv
        │
        ▼
Salin manual ke backend/ (model_sentimen.pkl, tfidf_vectorizer.pkl,
model_metadata.json, dataset_final_clean.csv)
```

### 2b. Alur runtime (tiap kali user memakai web app)

```
Pengguna (input teks / upload CSV)
        │
        ▼
Frontend React ── fetch() ──▶ Backend FastAPI
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                                ▼
          preprocessing.clean_text()          absa.analyze_aspects()
          (untuk model ML)                    (untuk breakdown aspek,
                    │                          jalan di teks MENTAH,
                    ▼                          bukan hasil clean_text)
          tfidf.transform()
                    │
                    ▼
          model.predict_proba()
                    │
                    ▼
          gabungkan hasil ML + ABSA jadi satu response JSON
                    │
                    ▼
Frontend ◀── {label, confidence, proba, cleaned_text, aspects[]}
        │
        ▼
Render: gauge sentimen keseluruhan + chip per aspek
```

**Catatan penting:** ABSA jalan di **teks mentah** (sebelum
`clean_text()`), bukan di `cleaned_text`. Alasannya: `clean_text()`
membuang tanda baca dan kata hubung ("tapi", "namun") yang justru
dipakai ABSA untuk memisahkan klausa. Jangan pernah mengubah
`analyze_aspects()` untuk menerima `cleaned_text` sebagai input.

Untuk `/api/predict-batch` dan `/api/upload-csv`, ABSA **sengaja
dimatikan** (`include_aspects=False`) supaya proses banyak baris tetap
cepat. ABSA hanya aktif di `/api/predict` (satu komentar).

---

## 3. Aturan Sistem & Keputusan Desain

Bagian ini menjelaskan **kenapa** sistem dibangun seperti sekarang,
supaya perubahan berikutnya tidak mengulang jalan buntu yang sama.

### Kenapa Logistic Regression, bukan Naive Bayes atau Transformer?

- **Naive Bayes** sempat dipakai (akurasi ~88.3%), tapi dibuang karena
  scope disederhanakan jadi satu model saja. Logistic Regression dipilih
  karena hasilnya sedikit lebih baik setelah augmentasi lexicon (89.17%)
  dan mendukung `class_weight='balanced'` secara native.
- **IndoBERT / XLM-RoBERTa** (Transformer) **tidak bisa dijalankan di
  sandbox development ini** — keduanya butuh download bobot model
  (ratusan MB–GB) dari Hugging Face Hub, dan domain `huggingface.co`
  tidak ada di allowlist jaringan sandbox. Di luar itu, Transformer juga
  butuh GPU untuk training yang layak dan inference yang cukup cepat —
  tidak cocok untuk deployment ringan di laptop biasa. Kalau suatu saat
  mau dicoba, jalankan fine-tuning di Google Colab (ada GPU gratis +
  akses internet penuh), bukan di lingkungan development project ini.

### Kenapa ada "lexicon augmentation" di training, bukan cuma preprocessing biasa?

**Temuan kunci (jangan dihapus tanpa investigasi ulang):** dataset
`dataset_siap_pakai.csv` yang namanya menyiratkan review Shopee/Tokopedia
ternyata **~97% berisi teks Bahasa Inggris**. Kata sentimen Indonesia
yang justru paling penting buat kasus penggunaan nyata — "jelek", "rusak",
"buruk" — masing-masing cuma muncul 0-1 kali di 2000 baris data asli.
Akibatnya kata-kata itu tidak lolos ambang `min_df` TF-IDF dan **sama
sekali tidak dipelajari model** (bukan cuma bobotnya kecil — literally
tidak ada di vocabulary).

Ini yang menyebabkan bug awal: `"barangnya jelek"` diprediksi **positif
56%** (kata "jelek" diabaikan total, model cuma membaca "barang" yang
kebetulan lebih sering muncul di review positif).

**Fix yang dipakai:** `build_lexicon_augmentation()` di `train_final.py`
menambahkan baris sintetis pendek (kombinasi kata sentimen ID/EN × 8
template kalimat pendek) sebagai **suplemen kecil** (~14% dari total
data), bukan pengganti data asli. Ini memastikan kata-kata sentimen
penting punya representasi minimum di vocabulary & asosiasi kelas yang
benar, tanpa mengarang ulang seluruh dataset.

**Kalau mau ganti dataset di masa depan:** cek dulu distribusi bahasa &
representasi kata sentimen kunci sebelum asumsi datanya sudah representatif.

### Kenapa ada 2 lapis "kamus kata" (`lexicon.py` dan augmentasi di `train_final.py`)?

Sempat dicoba **runtime hybrid blending** (`lexicon.py: blend_with_ml()`)
sebagai pendekatan alternatif — mencampur probabilitas ML dengan skor
lexicon saat prediksi (bukan saat training). Pendekatan ini terbukti
menurunkan akurasi test set sebesar ~0.5-0.8 poin persen karena
lexicon sederhana tidak sadar-aspek (kata positif tentang "penjual"
ikut menaikkan skor walau keluhannya soal "produk"). **`lexicon.py`
masih ada di kode** dan dipakai `absa.py` (untuk daftar kata
positif/negatif), TAPI fungsi `blend_with_ml()` **tidak lagi dipanggil
di `main.py`** — root cause bug sudah diselesaikan lewat augmentasi
training di atas, bukan lewat blending runtime. Jangan aktifkan lagi
`blend_with_ml()` di jalur prediksi utama tanpa evaluasi ulang penuh.

### Kenapa ABSA rule-based, bukan model terlatih?

ABSA "penuh" butuh dataset berlabel per-aspek (mis. "kalimat X → aspek
Produk = negatif") yang tidak tersedia, dan model ABSA modern biasanya
berbasis Transformer (masalah yang sama seperti di atas: butuh
Hugging Face + GPU). Solusi yang dipakai: pemisahan klausa (dipecah di
tanda baca & kata kontras seperti "tapi"/"but") + deteksi kata kunci
aspek + skor sentimen lokal dari `lexicon.py`. Ini transparan, cepat,
tanpa dependency berat — tapi **presisinya terbatas** pada aspek dan
kata kunci yang didaftarkan manual di `ASPECT_KEYWORDS_RAW`.

### Kenapa ada heuristik `_might_be_indonesian()` di preprocessing?

**Masalah performa, bukan bug logika:** Sastrawi adalah stemmer khusus
Bahasa Indonesia. Kalau dikasih kata Inggris ("product", "very", "good"),
ia tetap mencoba SEMUA kombinasi imbuhan Indonesia sebelum menyerah —
ini bisa makan ~100ms PER KATA. Karena dataset ~97% Inggris, tanpa
penyaringan ini training bisa makan >15 menit untuk 2000 baris.
`_might_be_indonesian()` melewatkan kata yang jelas tidak match pola
awalan/akhiran Indonesia (`me-, di-, ter-, ber-, per-, pe-, ke-, se-` /
`-kan, -an, -i, -lah, -kah, -nya, -ku, -mu`) tanpa memanggil Sastrawi
sama sekali. Ini murni optimasi kecepatan (verified: hasil stemming kata
Indonesia yang sebenarnya tidak berubah) — **implementasi identik ada
di 3 tempat** (`train_final.py`, `backend/preprocessing.py`,
`backend/absa.py` lewat import) dan **harus tetap sinkron** kalau salah
satu diubah.

### Aturan lain

- **Semua baris CSV dipakai untuk training** (tidak ada sampling/limit
  buatan). Baris kosong/label invalid dibuang, tapi tidak ada
  pengurangan jumlah data secara sengaja.
- **`preprocessing.py` (backend) dan logic `clean_text()` di
  `train_final.py` harus identik.** Kalau salah satu diubah, ubah
  keduanya — kalau tidak, prediksi runtime tidak akan konsisten dengan
  apa yang dipelajari model saat training.
- **Test split selalu berbasis review unik** (`drop_duplicates`), supaya
  baris yang sama tidak muncul di train & test sekaligus (data leakage
  yang menggelembungkan akurasi secara palsu).

---

## 4. Algoritma yang Dipakai

| Tahap | Algoritma | Keterangan |
|---|---|---|
| Cleaning teks | Regex pattern matching | Hapus URL, simbol, normalisasi huruf berulang ("lucuuuu"→"lucu") |
| Normalisasi slang | Dictionary lookup | "gak"→"tidak", "bgt"→"banget", dst (`slang_dict`) |
| Stopword removal | Set difference | Gabungan stopword ID (Sastrawi) + EN (NLTK), kata negasi di-whitelist |
| Stemming | Nazief-Adriani (via Sastrawi) | Hanya dipanggil untuk kata yang lolos heuristik `_might_be_indonesian()` |
| Ekstraksi fitur | TF-IDF, n-gram (1,2) | `max_features=6000, min_df=1, max_df=0.9` |
| Klasifikasi | Logistic Regression | `class_weight='balanced'`, `max_iter=1000` |
| Augmentasi data | Template-based synthetic generation | Suplemen training, bukan pengganti data asli |
| ABSA | Rule-based: clause splitting + keyword matching + lexicon scoring | Tidak memakai ML, murni aturan |
| Negasi (lexicon & ABSA) | Window-based flip (cek 1-2 token ke belakang) | "tidak bagus", "tidak terlalu bagus" |

---

## 5. Referensi Function per File

### `train_final.py` (script training, jalan manual)
| Function | Peran |
|---|---|
| `cached_stem(word)` | Stem kata Indonesia dengan cache + skip untuk kata non-Indonesia |
| `_might_be_indonesian(word)` | Heuristik cepat: apakah kata ini layak dicoba di-stem |
| `clean_text(text)` | Pipeline cleaning lengkap (lihat Section 4) |
| `build_lexicon_augmentation()` | Bangun baris sintetis dari `NEGATIVE_LEXICON`/`POSITIVE_LEXICON` × `TEMPLATES` |
| *(bagian utama script)* | Load data → clean → augmentasi → TF-IDF → split → fit LogisticRegression → simpan artefak |

### `backend/preprocessing.py` (dipakai backend saat runtime)
| Function | Peran |
|---|---|
| `_cached_stem(word)` | Identik dengan `cached_stem` di `train_final.py` — **harus tetap sinkron** |
| `_might_be_indonesian(word)` | Identik dengan versi training |
| `clean_text(text)` | **Harus identik logikanya** dengan `clean_text` di `train_final.py` |

### `backend/lexicon.py` (daftar kata sentimen, dipakai `absa.py`)
| Function | Peran |
|---|---|
| `lexicon_score(cleaned_text)` | Hitung net score +/- dari kata di `POSITIVE_WORDS`/`NEGATIVE_WORDS`, dengan flip negasi |
| `lexicon_proba_positif(net_score)` | Ubah net_score jadi probabilitas 0-1 (sigmoid) |
| `blend_with_ml(...)` | **Tidak dipakai di jalur utama saat ini** (lihat Section 3) — disimpan untuk referensi/eksperimen lanjutan |

### `backend/absa.py` (Aspect-Based Sentiment Analysis)
| Function | Peran |
|---|---|
| `_tokenize_simple(clause)` | Lowercase, buang non-huruf, stem tiap token (pakai `_cached_stem` dari `preprocessing.py`) |
| `_clause_sentiment(tokens)` | Skor sentimen 1 klausa (reuse lexicon dari `lexicon.py`) |
| `analyze_aspects(raw_text)` | Fungsi utama: pecah klausa → deteksi aspek → skor sentimen → return list aspek |
| `ASPECT_KEYWORDS` | Dict aspek→keyword, **sudah di-stem saat load module** supaya cocok dengan token yang di-stem |

### `backend/main.py` (FastAPI app)
| Function/Endpoint | Peran |
|---|---|
| `predict_one(text, include_aspects=True)` | Fungsi inti: clean → TF-IDF → predict_proba → (opsional) ABSA |
| `GET /api/health` | Info model aktif, akurasi, jumlah data training |
| `POST /api/predict` | Prediksi 1 teks + breakdown ABSA |
| `POST /api/predict-batch` | Prediksi banyak teks via JSON list, ABSA **off** |
| `POST /api/upload-csv` | Prediksi dari file CSV (auto-deteksi kolom review), ABSA **off** |
| `GET /api/sample-stats` | Statistik dataset training untuk dashboard demo |

### `frontend/src/App.jsx` (React)
| Bagian | Peran |
|---|---|
| Tab "Satu komentar" | Form input teks → `POST /api/predict` → render gauge + chip ABSA |
| Tab "Unggah CSV" | Upload file → `POST /api/upload-csv` → render ringkasan + tabel pratinjau |
| `SentimentGauge` (component terpisah) | Meteran visual skor positif 0-100% |

---

## 6. Keterbatasan yang Diketahui (jangan dijanjikan sebagai "sudah selesai")

- ABSA rule-based bisa salah pada review dengan sarkasme, kalimat
  kompleks bersarang, atau aspek yang tidak ada di `ASPECT_KEYWORDS_RAW`.
- Model tidak punya kelas netral — review ambigu akan dipaksa ke
  positif/negatif dengan confidence rendah (~50-55%).
- `max_features=6000` di TF-IDF adalah batas atas vocabulary; kalau
  dataset diperbesar signifikan, pertimbangkan menaikkan ini.
- Tidak ada mekanisme retrain otomatis — model harus dilatih ulang
  manual (`python3 train_final.py`) tiap kali data berubah, lalu hasil
  `.pkl`/`.json`/`.csv` di-copy manual ke `backend/`.

---

## 7. Riwayat Perubahan Relevan

| Perubahan | Alasan |
|---|---|
| NB vs LR comparison → LR saja | Simplifikasi atas permintaan eksplisit: satu model, bukan perbandingan |
| Tambah augmentasi lexicon di training | Fix bug "barangnya jelek" diprediksi positif — root cause: kata sentimen ID nyaris tidak ada di data asli |
| Tambah `_might_be_indonesian()` heuristik | Fix training yang awalnya >15 menit (timeout) jadi ~1 menit |
| Tambah `absa.py` | Permintaan eksplisit: Analisis Sentimen Berbasis Aspek |
| `blend_with_ml()` dibuat lalu tidak dipakai di jalur utama | Dicoba sebagai fix awal, terbukti trade-off akurasi tidak sepadan dibanding fix di level data training |
