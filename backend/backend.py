import os, pickle
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, validator

# ─────────────────────────────────────────────────────────
# BASE PATH
# Dijalankan dari ROOT → uvicorn backend.backend:app
# __file__ = PROJECT_ML-PRAKT-MAIN/backend/backend.py
# .parent  = PROJECT_ML-PRAKT-MAIN/backend/
# .parent  = PROJECT_ML-PRAKT-MAIN/   ← ROOT, tempat model ada
# ─────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
print("=" * 55)
print(f"  BASE dir : {BASE}")
print("=" * 55)

def _p(name):
    return str(BASE / name)

# ─────────────────────────────────────────────────────────
# LOAD ARTEFAK
# ─────────────────────────────────────────────────────────
def _load_model(path):
    try:
        m = tf.keras.models.load_model(path)
        print(f"  ✅  {Path(path).name}")
        return m
    except Exception as e:
        print(f"  ❌  {Path(path).name}  →  {e}")
        return None

def _load_pkl(path, label):
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        print(f"  ✅  {label}")
        return obj
    except Exception as e:
        print(f"  ❌  {label}  →  {e}")
        return None

model1       = _load_model(_p("weather_model1.keras"))
model2       = _load_model(_p("weather_model2.keras"))
scaler       = _load_pkl(_p("scaler.pkl"),       "scaler.pkl")
enc_data     = _load_pkl(_p("encoders.pkl"),     "encoders.pkl")
TEST_METRICS = _load_pkl(_p("test_metrics.pkl"), "test_metrics.pkl") or {}

label_encoder = enc_data["target"] if enc_data else None
CLASS_NAMES = (
    list(label_encoder.classes_)
    if label_encoder else ["Cloudy", "Rainy", "Snowy", "Sunny"]
)

# Kolom fitur — urutan HARUS sama dengan training
try:
    FEATURE_COLS = list(scaler.feature_names_in_)
    print(f"  ✅  Feature cols dari scaler: {len(FEATURE_COLS)} fitur")
except Exception:
    FEATURE_COLS = [
        "Temperature", "Humidity", "Wind Speed",
        "Precipitation (%)", "Atmospheric Pressure",
        "UV Index", "Visibility (km)",
        "Cloud Cover_clear", "Cloud Cover_cloudy",
        "Cloud Cover_overcast", "Cloud Cover_partly cloudy",
        "Season_Autumn", "Season_Spring",
        "Season_Summer", "Season_Winter",
        "Location_coastal", "Location_inland", "Location_mountain",
    ]
    print(f"  ⚠️  Fallback FEATURE_COLS: {len(FEATURE_COLS)} fitur")

print(f"  Kelas  : {CLASS_NAMES}")
print(f"  M1 acc : {TEST_METRICS.get('model1_acc', 0)*100:.2f}%")
print(f"  M2 acc : {TEST_METRICS.get('model2_acc', 0)*100:.2f}%")
print("=" * 55 + "\n")

# ─────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────
app = FastAPI(title="Weather Classification API", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve UI — akses via http://localhost:8000/ui/index.html
UI_DIR = BASE / "ui"
UI_DIR.mkdir(exist_ok=True)
app.mount("/ui", StaticFiles(directory=str(UI_DIR), html=True), name="ui")

# ─────────────────────────────────────────────────────────
# ERROR HANDLER — tampilkan validasi error yang jelas
# ─────────────────────────────────────────────────────────
@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    errors = []
    for err in exc.errors():
        field = " → ".join(str(x) for x in err["loc"])
        errors.append(f"{field}: {err['msg']}  (nilai: {err.get('input','?')})")
    return JSONResponse(
        status_code=422,
        content={"error": "Input tidak valid", "detail": errors}
    )

# ─────────────────────────────────────────────────────────
# SCHEMA
# ─────────────────────────────────────────────────────────
class WeatherInput(BaseModel):
    temperature:          float = Field(..., ge=-20,  le=55)
    humidity:             float = Field(..., ge=0,    le=100)
    wind_speed:           float = Field(..., ge=0,    le=150)
    precipitation:        float = Field(..., ge=0,    le=100)
    atmospheric_pressure: float = Field(1013.0, ge=900, le=1050)
    uv_index:             float = Field(..., ge=0,    le=15)
    visibility_km:        float = Field(..., ge=0,    le=50)
    cloud_cover:          str   = Field(...)
    season:               str   = Field(...)
    location:             str   = Field(...)
    model_choice:         str   = Field("best")

    # Validator — toleran terhadap tipe dan case dari HTML slider/select
    @validator("uv_index", pre=True)
    def parse_uv(cls, v): return round(float(v))

    @validator("cloud_cover", pre=True)
    def norm_cloud(cls, v): return str(v).strip().lower()

    @validator("season", pre=True)
    def norm_season(cls, v): return str(v).strip().capitalize()

    @validator("location", pre=True)
    def norm_location(cls, v): return str(v).strip().lower()

    @validator("model_choice", pre=True)
    def norm_model(cls, v): return str(v).strip().lower()

# ─────────────────────────────────────────────────────────
# HELPER — build fitur
# ─────────────────────────────────────────────────────────
def build_features(data: WeatherInput) -> np.ndarray:
    row = {
        "Temperature":          float(data.temperature),
        "Humidity":             float(data.humidity),
        "Wind Speed":           float(data.wind_speed),
        "Precipitation (%)":    float(data.precipitation),
        "Atmospheric Pressure": float(data.atmospheric_pressure),
        "UV Index":             float(data.uv_index),
        "Visibility (km)":      float(data.visibility_km),
    }
    for opt in ["clear", "cloudy", "overcast", "partly cloudy"]:
        row[f"Cloud Cover_{opt}"] = 1.0 if data.cloud_cover == opt else 0.0
    for opt in ["Autumn", "Spring", "Summer", "Winter"]:
        row[f"Season_{opt}"] = 1.0 if data.season == opt else 0.0
    for opt in ["coastal", "inland", "mountain"]:
        row[f"Location_{opt}"] = 1.0 if data.location == opt else 0.0

    df = pd.DataFrame([row])
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0
    df = df[FEATURE_COLS]
    return scaler.transform(df.values).astype(np.float32)

# ─────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status": "ok",
        "ui": "http://localhost:8000/ui/index.html",
        "docs": "http://localhost:8000/docs",
        "classes": CLASS_NAMES,
        "models_loaded": {
            "model1": model1 is not None,
            "model2": model2 is not None,
        },
    }

@app.get("/health")
def health():
    return {
        "ready": all([model1, model2, scaler]),
        "classes": CLASS_NAMES,
        "models": {"model1": model1 is not None, "model2": model2 is not None},
        "test_metrics": {
            "model1_acc": round(TEST_METRICS.get("model1_acc", 0) * 100, 2),
            "model2_acc": round(TEST_METRICS.get("model2_acc", 0) * 100, 2),
        },
    }

@app.post("/predict")
def predict(data: WeatherInput):
    if not all([model1, model2, scaler]):
        raise HTTPException(
            status_code=503,
            detail=(
                "Model belum di-load. "
                "Cek apakah weather_model1.keras, weather_model2.keras, "
                "dan scaler.pkl ada di folder root PROJECT_ML-PRAKT-MAIN/"
            )
        )

    try:
        X = build_features(data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal build fitur: {e}")

    try:
        prob1 = model1.predict(X, verbose=0)[0].tolist()
        prob2 = model2.predict(X, verbose=0)[0].tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal prediksi: {e}")

    pred1_idx = int(np.argmax(prob1))
    pred2_idx = int(np.argmax(prob2))

    acc1 = round(TEST_METRICS.get("model1_acc", 0.0) * 100, 2)
    acc2 = round(TEST_METRICS.get("model2_acc", 0.0) * 100, 2)
    best = "model1" if acc1 >= acc2 else "model2"

    if data.model_choice == "model1":
        final_idx, final_prob, used = pred1_idx, prob1, "model1"
    elif data.model_choice == "model2":
        final_idx, final_prob, used = pred2_idx, prob2, "model2"
    else:
        final_idx  = pred1_idx if best == "model1" else pred2_idx
        final_prob = prob1     if best == "model1" else prob2
        used = best

    return {
        "prediction":  CLASS_NAMES[final_idx],
        "confidence":  round(final_prob[final_idx] * 100, 2),
        "probabilities": {
            cls: round(p * 100, 2)
            for cls, p in zip(CLASS_NAMES, final_prob)
        },
        "model_used": used,
        "model1_detail": {
            "prediction":    CLASS_NAMES[pred1_idx],
            "confidence":    round(prob1[pred1_idx] * 100, 2),
            "test_accuracy": acc1,
        },
        "model2_detail": {
            "prediction":    CLASS_NAMES[pred2_idx],
            "confidence":    round(prob2[pred2_idx] * 100, 2),
            "test_accuracy": acc2,
        },
    }

@app.get("/model-info")
def model_info():
    return {
        "model1": {
            "name": "ANN Standar",
            "architecture": "Dense(128→64→32) + Dropout",
            "optimizer": "Adam",
            "test_accuracy": round(TEST_METRICS.get("model1_acc", 0) * 100, 2),
        },
        "model2": {
            "name": "Wide & Deep ANN",
            "architecture": "Dense(256→128→64→32) + BatchNorm + LeakyReLU",
            "optimizer": "AdamW + weight_decay",
            "test_accuracy": round(TEST_METRICS.get("model2_acc", 0) * 100, 2),
        },
        "classes": CLASS_NAMES,
        "n_features": len(FEATURE_COLS),
    }