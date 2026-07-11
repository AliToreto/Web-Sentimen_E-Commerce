"""
Backend FastAPI - Analisis Sentimen Review E-Commerce
=======================================================
Model: TF-IDF + Logistic Regression (satu model).
Endpoint:
  GET  /api/health            -> cek status & info model
  POST /api/predict           -> prediksi 1 teks (+ breakdown ABSA per-aspek)
  POST /api/predict-batch     -> prediksi banyak teks via JSON list
  POST /api/upload-csv        -> upload CSV review produk, hasil agregat + per-baris
  GET  /api/sample-stats      -> statistik dataset training (untuk dashboard demo)
"""

import io
import json
import pickle
from collections import Counter

import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from preprocessing import clean_text
from absa import analyze_aspects

app = FastAPI(title="API Analisis Sentimen E-Commerce")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# LOAD MODEL & VECTORIZER SEKALI SAAT STARTUP
# =====================================================

with open("model_sentimen.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("model_metadata.json", "r") as f:
    metadata = json.load(f)

CLASS_LABELS = [str(c) for c in model.classes_]


def predict_one(text: str, include_aspects: bool = True):
    cleaned = clean_text(text)
    if not cleaned.strip():
        return {
            "label": "netral",
            "confidence": 0.0,
            "proba": {lbl: 0.0 for lbl in CLASS_LABELS},
            "cleaned_text": cleaned,
            "aspects": [],
            "note": "Teks tidak mengandung kata yang dapat dianalisis setelah dibersihkan.",
        }

    vec = tfidf.transform([cleaned])
    label = str(model.predict(vec)[0])
    proba = model.predict_proba(vec)[0]
    proba_dict = {cls: float(p) for cls, p in zip(CLASS_LABELS, proba)}
    confidence = float(max(proba))

    result = {
        "label": label,
        "confidence": confidence,
        "proba": proba_dict,
        "cleaned_text": cleaned,
    }

    if include_aspects:
        result["aspects"] = analyze_aspects(text)

    return result


# =====================================================
# SCHEMAS
# =====================================================

class PredictRequest(BaseModel):
    text: str


class PredictBatchRequest(BaseModel):
    texts: list[str]


# =====================================================
# ENDPOINTS
# =====================================================

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model": metadata.get("model"),
        "accuracy": metadata.get("accuracy"),
        "f1_macro": metadata.get("f1_macro"),
        "labels": CLASS_LABELS,
        "n_training_samples": metadata.get("n_samples_total"),
    }


@app.post("/api/predict")
def predict(req: PredictRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Teks tidak boleh kosong")
    return predict_one(req.text)


@app.post("/api/predict-batch")
def predict_batch(req: PredictBatchRequest):
    if not req.texts:
        raise HTTPException(status_code=400, detail="Daftar teks tidak boleh kosong")

    results = [predict_one(t, include_aspects=False) for t in req.texts]

    counts = Counter(r["label"] for r in results)
    total = len(results)
    summary = {
        lbl: {"count": counts.get(lbl, 0), "percent": round(counts.get(lbl, 0) / total * 100, 2)}
        for lbl in CLASS_LABELS
    }

    return {"summary": summary, "total": total, "results": results}


@app.post("/api/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="File harus berformat CSV")

    raw = await file.read()

    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception:
        try:
            df = pd.read_csv(io.BytesIO(raw), sep=';', encoding='latin-1', on_bad_lines='skip')
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Gagal membaca CSV: {e}")

    # cari kolom teks yang paling mungkin berisi review
    candidate_cols = [c for c in df.columns if c.lower() in
                       ("review", "review_text", "comment", "komentar", "ulasan", "text")]
    text_col = candidate_cols[0] if candidate_cols else df.columns[0]

    texts = df[text_col].astype(str).fillna("").tolist()
    if len(texts) == 0:
        raise HTTPException(status_code=400, detail="CSV tidak berisi data")
    if len(texts) > 5000:
        texts = texts[:5000]

    results = [predict_one(t, include_aspects=False) for t in texts]

    counts = Counter(r["label"] for r in results)
    total = len(results)
    summary = {
        lbl: {"count": counts.get(lbl, 0), "percent": round(counts.get(lbl, 0) / total * 100, 2)}
        for lbl in CLASS_LABELS
    }

    preview = [
        {"text": texts[i], **results[i]}
        for i in range(min(50, total))
    ]

    return {
        "filename": file.filename,
        "text_column_used": text_col,
        "summary": summary,
        "total": total,
        "preview": preview,
    }


@app.get("/api/sample-stats")
def sample_stats():
    df = pd.read_csv("dataset_final_clean.csv")
    counts = df["sentimen"].value_counts().to_dict()
    return {
        "total": len(df),
        "counts": counts,
        "sample_reviews": {
            "positif": df[df.sentimen == "positif"]["review"].sample(min(5, (df.sentimen == "positif").sum()), random_state=1).tolist(),
            "negatif": df[df.sentimen == "negatif"]["review"].sample(min(5, (df.sentimen == "negatif").sum()), random_state=1).tolist(),
        },
    }
