import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

df = pd.read_csv("dataset/weather_clean.csv")
X = df.drop("Weather Type", axis=1)
y_raw = df["Weather Type"]

X_train, X_temp, y_train_raw, y_temp_raw = train_test_split(
    X, y_raw, test_size=0.30, random_state=42, stratify=y_raw
)
X_val, X_test, y_val_raw, y_test_raw = train_test_split(
    X_temp, y_temp_raw, test_size=1/3, random_state=42, stratify=y_temp_raw
)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

X_test_sc = scaler.transform(X_test)

with open("encoders.pkl", "rb") as f:
    enc_data = pickle.load(f)

label_encoder = enc_data["target"]
class_names   = label_encoder.classes_
print("Data Testing:", X_test_sc.shape)

model1 = tf.keras.models.load_model("weather_model1.keras")
model2 = tf.keras.models.load_model("weather_model2.keras")

models = {
    "Model 1 — ANN Standar":    model1,
    "Model 2 — Deep ANN": model2,
}

# Test metrics
results = {}
for name, model in models.items():
    print(f"\n{'='*45}")
    print(name)
    print('='*45)

    pred_prob = model.predict(X_test_sc)
    pred_class = np.argmax(pred_prob, axis=1)
    y_true = y_test_raw.values

    acc = accuracy_score(y_true, pred_class)
    prec = precision_score(y_true, pred_class, average="macro", zero_division=0)
    rec = recall_score(y_true, pred_class, average="macro", zero_division=0)
    f1 = f1_score(y_true, pred_class, average="macro", zero_division=0)

    results[name] = {
        "Accuracy":  acc,
        "Precision": prec,
        "Recall":    rec,
        "F1-Score":  f1
    }

    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, pred_class,
                                target_names=class_names, zero_division=0))

    # Confusion Matrix
    cm = confusion_matrix(y_true, pred_class)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True, fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title(f"Confusion Matrix\n{name}", fontsize=12)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_{'1' if 'Standar' in name else '2'}.png",
                dpi=150, bbox_inches="tight")
    plt.show()

# Perbandingan hasil
print("\n\n=== PERBANDINGAN HASIL ===")
df_result = pd.DataFrame(results).T.round(4)
print(df_result.to_string())

# Plot komparasi bar chart
fig, ax = plt.subplots(figsize=(8, 5))
df_result.plot(kind="bar", ax=ax, colormap="Set2", edgecolor="white")
ax.set_title("Komparasi Metrik Model 1 vs Model 2", fontsize=13)
ax.set_ylabel("Score")
ax.set_xticklabels(["Model 1\nANN Standar", "Model 2\nDeep ANN"], rotation=0, fontsize=10)
ax.legend(loc="lower right")
ax.set_ylim(0, 1.05)
plt.tight_layout()
plt.savefig("komparasi_model.png", dpi=150, bbox_inches="tight")
plt.show()

# Training plot
for pkl_file, model_name in [("history_model1.pkl", "Model 1 — ANN Standar"), ("history_model2.pkl", "Model 2 — Deep ANN")]:
    with open(pkl_file, "rb") as f:
        history = pickle.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(model_name, fontsize=13)

    # Accuracy
    ax1.plot(history["accuracy"], label="Train Accuracy", color="#4C72B0")
    ax1.plot(history["val_accuracy"], label="Validation Accuracy", color="#DD8452", linestyle="--")
    ax1.set_title("Accuracy per Epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Loss
    ax2.plot(history["loss"], label="Train Loss", color="#4C72B0")
    ax2.plot(history["val_loss"], label="Validation Loss", color="#DD8452", linestyle="--")
    ax2.set_title("Loss per Epoch")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    suffix = "1" if "Standar" in model_name else "2"
    plt.savefig(f"training_plot_model{suffix}.png", dpi=150, bbox_inches="tight")
    plt.show()

# Cross-validation summary
print("\n=== CROSS-VALIDATION SUMMARY ===")
for pkl_file, mname in [("cv_scores_model1.pkl", "Model 1"), ("cv_scores_model2.pkl", "Model 2")]:
    try:
        with open(pkl_file, "rb") as f:
            cv_scores = pickle.load(f)
        print(f"{mname}: CV Mean={np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    except FileNotFoundError:
        print(f"{mname}: file CV tidak ditemukan.")