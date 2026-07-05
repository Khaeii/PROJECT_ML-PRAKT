import os
os.environ["TF_NUM_INTEROP_THREADS"] = "2"
os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from keras.callbacks import (EarlyStopping, ReduceLROnPlateau,ModelCheckpoint)
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pickle
import random

tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(2)

tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

df = pd.read_csv("dataset/weather_clean.csv")
X = df.drop("Weather Type", axis=1)
y_raw = df["Weather Type"]
print("Shape data:", X.shape)
print("Distribusi kelas:\n", y_raw.value_counts())

# Stratified split 70% train, 20% validation, 10% test
X_train, X_temp, y_train_raw, y_temp_raw = train_test_split(
    X, y_raw,
    test_size=0.30,
    random_state=42,
    stratify=y_raw          
)

X_val, X_test, y_val_raw, y_test_raw = train_test_split(
    X_temp, y_temp_raw,
    test_size=1/3,
    random_state=42,
    stratify=y_temp_raw     
)

print(f"\nTrain = {X_train.shape}")
print(f"Val   = {X_val.shape}")
print(f"Test  = {X_test.shape}")

# Scaling
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)   
X_val   = scaler.transform(X_val)         
X_test  = scaler.transform(X_test)        
with open("scaler.pkl", "wb") as f: pickle.dump(scaler, f)

# One-hot encode target
n_classes = len(np.unique(y_raw))
y_train = to_categorical(y_train_raw, num_classes=n_classes)
y_val = to_categorical(y_val_raw,   num_classes=n_classes)
y_test = to_categorical(y_test_raw,  num_classes=n_classes)

# MODEL 1 — ANN Standar
model1 = Sequential([
    Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dense(n_classes, activation="softmax")
], name="Model1_ANN")

model1.summary()

model1.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Callbacks
callbacks_m1 = [
    EarlyStopping(
        monitor="val_loss",
        patience=10,                  
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,                   
        patience=5,
        min_lr=1e-6,
        verbose=1
    ),
    ModelCheckpoint(
        filepath="best_model1.keras",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )
]

# Training
print("\n" + "="*50)
print("TRAINING MODEL 1 — ANN Standar")
print("="*50)

history1 = model1.fit(
    X_train, y_train,
    epochs=50,
    batch_size=64,
    validation_data=(X_val, y_val),
    callbacks=callbacks_m1,
    verbose=2
)

# Evaluate on test set
loss1, acc1 = model1.evaluate(X_test, y_test, verbose=0)
print(f"\nModel 1 Test Accuracy : {acc1:.4f}")
print(f"Model 1 Test Loss     : {loss1:.4f}")

# Cross-validation
print("\n" + "-"*45)
print("CROSS-VALIDATION MODEL 1  (5-Fold Stratified)")
print("-"*45)

X_cv = np.vstack([X_train, X_val])
y_cv = np.hstack([y_train_raw.values, y_val_raw.values])

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for fold, (tr_idx, vl_idx) in enumerate(skf.split(X_cv, y_cv), 1):
    X_tr, X_vl = X_cv[tr_idx], X_cv[vl_idx]
    y_tr = to_categorical(y_cv[tr_idx], num_classes=n_classes)
    y_vl = to_categorical(y_cv[vl_idx], num_classes=n_classes)

    m_cv = Sequential([
        Dense(128, activation="relu", input_shape=(X_tr.shape[1],)),
        Dropout(0.3),
        Dense(64,  activation="relu"),
        Dropout(0.2),
        Dense(32,  activation="relu"),
        Dense(n_classes, activation="softmax")
    ])
    m_cv.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    m_cv.fit(
        X_tr, y_tr,
        epochs=30,              
        batch_size=64,
        validation_data=(X_vl, y_vl),
        callbacks=[EarlyStopping(
            monitor="val_loss",
            patience=7,
            restore_best_weights=True
        )],
        verbose=0               
    )
    _, cv_acc = m_cv.evaluate(X_vl, y_vl, verbose=0)
    cv_scores.append(cv_acc)
    print(f"Fold {fold}: {cv_acc:.4f}")

print(f"\nCV Mean  : {np.mean(cv_scores):.4f}")
print(f"CV Std   : {np.std(cv_scores):.4f}")
print(f"Interpretasi: {'STABIL ✓' if np.std(cv_scores) < 0.02 else 'Perlu dicek, Std agak tinggi'}")


model1.save("weather_model1.keras")
with open("history_model1.pkl", "wb") as f:
    pickle.dump(history1.history, f)
with open("cv_scores_model1.pkl", "wb") as f:
    pickle.dump(cv_scores, f)


try:
    with open("test_metrics.pkl", "rb") as f:
        metrics = pickle.load(f)
except FileNotFoundError:
    metrics = {}

metrics["model1_acc"] = float(acc1)

with open("test_metrics.pkl", "wb") as f:
    pickle.dump(metrics, f)

print(f"\ntest_metrics.pkl diupdate → model1_acc = {acc1:.4f}")
print("\n" + "="*50)
print(f"Test Accuracy : {acc1:.4f}")
print(f"CV Mean       : {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
print("="*50)