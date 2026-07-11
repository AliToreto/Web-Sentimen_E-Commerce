# Sentimeter — Analisis Sentimen Review E-Commerce

Web app untuk menganalisis apakah review/komentar produk di Shopee atau
Tokopedia cenderung positif atau negatif, menggunakan model Machine
Learning (TF-IDF + Logistic Regression) yang dilatih dari data review
e-commerce berbahasa campuran (Indonesia & Inggris), plus breakdown
sentimen per-aspek (ABSA) untuk memisahkan pujian/keluhan per bagian
(Produk, Pengiriman, Harga, Pelayanan, Kemasan).

> **Dokumentasi teknis lengkap:** lihat [`agent.md`](./agent.md) untuk
> alur kerja detail, keputusan desain (kenapa Logistic Regression, kenapa
> ABSA rule-based, dst), dan referensi tiap function di kode.

## Struktur folder

```
sentiment_project/
├── agent.md                  # dokumentasi teknis lengkap (workflow, aturan, algoritma, function)
├── train_final.py            # script training model (sumber: dataset_siap_pakai.csv)
├── dataset_siap_pakai.csv    # dataset asli (SEMUA baris dipakai, tidak disampling)
├── dataset_final_clean.csv   # dataset hasil preprocessing ulang (dihasilkan training)
├── model_sentimen.pkl        # model terlatih (Logistic Regression)
├── tfidf_vectorizer.pkl      # vectorizer TF-IDF
├── model_metadata.json       # info akurasi & jumlah data training
│
├── backend/                  # FastAPI backend (REST API)
│   ├── main.py                # endpoint API + fungsi predict_one()
│   ├── preprocessing.py       # HARUS identik dengan clean_text() di train_final.py
│   ├── lexicon.py             # daftar kata sentimen ID/EN (dipakai absa.py)
│   ├── absa.py                # Analisis Sentimen Berbasis Aspek (rule-based)
│   └── requirements.txt
│
└── frontend/                 # React + Vite frontend
    ├── src/
    │   ├── App.jsx             # form input, tab CSV, render hasil + chip ABSA
    │   ├── SentimentGauge.jsx
    │   └── index.css
    ├── package.json
    └── vite.config.js
```

## Hasil training

**Model: Logistic Regression + TF-IDF** — akurasi 89.17% (F1-macro 89.10%),
dilatih dari 2.326 baris (2.000 baris data asli + 336 baris augmentasi
lexicon untuk kata sentimen Indonesia yang langka — lihat `agent.md`
Section 3 untuk detail kenapa ini perlu).

Preprocessing menangani teks **bilingual** — stopwords dan stemming
digabung dari Bahasa Indonesia (Sastrawi) dan Inggris (NLTK), dengan
kata negasi ("tidak", "not", "bukan") sengaja dipertahankan karena
penting untuk makna sentimen.

## Cara menjalankan di komputer lokal

### 1. Jalankan backend (API model AI)

```bash
cd backend
pip install -r requirements.txt
python -m uvicorn main:app --reload --port 8000
```

Backend akan berjalan di `http://localhost:8000`. Cek dengan membuka
`http://localhost:8000/api/health` di browser — harus muncul info model.

### 2. Jalankan frontend (tampilan web)

Di terminal baru:

```bash
cd frontend
npm install
npm run dev
```

Buka `http://localhost:5173` di browser. Frontend otomatis terhubung ke
backend lewat proxy yang sudah dikonfigurasi di `vite.config.js`.

### 3. Build untuk produksi (opsional, kalau mau deploy)

```bash
cd frontend
npm run build
```

File hasil build ada di `frontend/dist/`, bisa di-deploy ke Vercel,
Netlify, atau hosting static lain. Backend FastAPI perlu di-deploy
terpisah (misal: Railway, Render, atau VPS) — jangan lupa update
`API_BASE` di `App.jsx` ke URL backend production-nya.

## Cara pakai aplikasi

1. **Satu komentar** — tempel satu review, dapat label positif/negatif +
   tingkat keyakinan model, **plus breakdown per-aspek** (mis. "Produk:
   negatif, Pengiriman: positif" untuk review campuran seperti
   *"barang jelek tapi pengiriman cepat"*).
2. **Unggah CSV** — upload file CSV berisi banyak review (kolom harus
   bernama salah satu dari: `review`, `comment`, `komentar`, `ulasan`,
   atau `text`). Hasilnya berupa ringkasan persentase + tabel pratinjau
   per baris (ABSA dimatikan di mode ini supaya tetap cepat untuk banyak
   baris sekaligus).

## Melatih ulang model dengan data baru

Ganti isi `dataset_siap_pakai.csv` (kolom wajib: `review`, `sentimen`
dengan nilai `positif`/`negatif`), lalu jalankan:

```bash
python train_final.py
```

Model baru akan otomatis menggantikan `model_sentimen.pkl`,
`tfidf_vectorizer.pkl`, `model_metadata.json`, dan `dataset_final_clean.csv`
di folder root. **Copy manual keempat file ini ke folder `backend/`**,
lalu restart backend supaya model baru ter-load.

Training memproses ~2.300 baris dalam ±1 menit (sudah dioptimasi dengan
cache + heuristik skip-stemming untuk kata non-Indonesia — lihat
`agent.md` Section 3 kalau mau tahu detailnya).

## Rencana pengembangan selanjutnya (saran)

- **Tambah kelas netral**: saat ini model hanya 2 kelas (positif/negatif).
  Review dengan rating 3 dibuang saat training — bisa ditambahkan kembali
  sebagai kelas ketiga kalau dibutuhkan nuansa lebih halus.
- **ABSA berbasis ML**: versi saat ini rule-based (keyword + lexicon).
  Kalau butuh presisi lebih tinggi, bisa dilatih model klasifikasi
  per-aspek — tapi butuh data berlabel aspek yang belum ada.
- **Word cloud / kata kunci dominan**: tampilkan kata-kata yang paling
  sering muncul di review positif vs negatif.
- **Autentikasi & riwayat analisis**: simpan histori upload per user
  (perlu database, misal PostgreSQL/SQLite).
- **Deploy permanen**: backend ke Railway/Render, frontend ke
  Vercel/Netlify, supaya bisa diakses tanpa menjalankan apapun secara
  lokal.
