import os
os.environ["TF_NUM_INTEROP_THREADS"] = "2"
os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "2"
import random
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split, StratifiedKFold

RUN_CV = True
tf.random.set_seed(42); np.random.seed(42); random.seed(42)
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(2)

df = pd.read_csv("dataset/weather_clean.csv")
X = df.drop("Weather Type", axis=1)
y_raw = df["Weather Type"]

X_train, X_temp, y_train_raw, y_temp_raw = train_test_split(
    X, y_raw, test_size=0.30, random_state=42, stratify=y_raw)
X_val, X_test, y_val_raw, y_test_raw = train_test_split(
    X_temp, y_temp_raw, test_size=1/3, random_state=42, stratify=y_temp_raw)

print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# One-hot encode target
n_classes = len(np.unique(y_raw))
y_train = to_categorical(y_train_raw, n_classes)
y_val = to_categorical(y_val_raw,   n_classes)
y_test = to_categorical(y_test_raw,  n_classes)

# Arsitektur model 
def build_model2(input_dim, n_classes):
    return Sequential([
        Dense(256, input_shape=(input_dim,)),
        LeakyReLU(negative_slope=0.1),
        BatchNormalization(),
        Dropout(0.35),

        Dense(128),
        LeakyReLU(negative_slope=0.1),
        BatchNormalization(),
        Dropout(0.25),

        Dense(64),
        LeakyReLU(negative_slope=0.1),
        BatchNormalization(),
        Dropout(0.20),

        Dense(32, activation="relu"),
        Dense(n_classes, activation="softmax")
    ], name="Model2_DeepANN")

model2 = build_model2(X_train.shape[1], n_classes)
model2.summary()
model2.compile(
    optimizer=AdamW(learning_rate=0.001, weight_decay=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Training
print("\n" + "="*50)
print("TRAINING MODEL 2 — Deep ANN")
print("="*50)

history2 = model2.fit(
    X_train, y_train,
    epochs=50,
    batch_size=64,
    validation_data=(X_val, y_val),
    callbacks=[
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=5, min_lr=1e-7, verbose=1),
        ModelCheckpoint("best_model2.keras", monitor="val_accuracy", save_best_only=True, verbose=1),
    ],
    verbose=1
)

# Evaluasi 
loss2, acc2 = model2.evaluate(X_test, y_test, verbose=0)
print(f"\nModel 2 Test Accuracy : {acc2:.4f}")
print(f"Model 2 Test Loss     : {loss2:.4f}")

# Cross-validation
cv_scores = []
if RUN_CV:
    print("\n" + "-"*45)
    print("CROSS-VALIDATION MODEL 2 (5-Fold Stratified)")
    print("-"*45)

    X_cv = np.vstack([X_train, X_val])
    y_cv = np.hstack([y_train_raw.values, y_val_raw.values])
    skf  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (tr_idx, vl_idx) in enumerate(skf.split(X_cv, y_cv), 1):
        X_tr, X_vl = X_cv[tr_idx], X_cv[vl_idx]
        y_tr = to_categorical(y_cv[tr_idx], n_classes)
        y_vl = to_categorical(y_cv[vl_idx], n_classes)

        m_cv = build_model2(X_tr.shape[1], n_classes)
        m_cv.compile(
            optimizer=AdamW(learning_rate=0.001, weight_decay=0.0001),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        m_cv.fit(
            X_tr, y_tr,
            epochs=15,
            batch_size=64,
            validation_data=(X_vl, y_vl),
            callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)],
            verbose=0
        )
        _, cv_acc = m_cv.evaluate(X_vl, y_vl, verbose=0)
        cv_scores.append(cv_acc)
        print(f"  Fold {fold}: {cv_acc:.4f}")
        del m_cv

    print(f"\n  CV Mean : {np.mean(cv_scores):.4f}")
    print(f"  CV Std  : {np.std(cv_scores):.4f}")
    print(f"  Status  : {'STABIL' if np.std(cv_scores) < 0.02 else 'Std tinggi, cek data'}")

model2.save("weather_model2.keras")
with open("history_model2.pkl", "wb") as f:  pickle.dump(history2.history, f)
with open("cv_scores_model2.pkl", "wb") as f: pickle.dump(cv_scores, f)

try:
    with open("test_metrics.pkl", "rb") as f: metrics = pickle.load(f)
except FileNotFoundError:
    metrics = {}
metrics["model2_acc"] = float(acc2)
with open("test_metrics.pkl", "wb") as f: pickle.dump(metrics, f)

print("\n" + "="*50)
print(f"Test Accuracy : {acc2:.4f}")
if cv_scores:
    print(f"CV Mean       : {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
print("="*50)