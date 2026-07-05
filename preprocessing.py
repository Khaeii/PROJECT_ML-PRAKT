import pandas as pd
import numpy as np
import json
import re
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

DATA_PATH  = "dataset/weather data classification.csv"
OUTPUT_CSV = "dataset/weather_clean.csv"
SCALER_PKL = "scaler.pkl"
ENCODER_PKL = "encoders.pkl"

df = pd.read_csv(DATA_PATH)
print(df.head())

def extract_temp(text):
    data = json.loads(text)
    return data["Temperature"]["value"]

def extract_humidity(text):
    data = json.loads(text)
    return data["Humidity"]["value"]

if "{Temperature},{Humdity}" in df.columns:
    df["Temperature"] = df["{Temperature},{Humdity}"].apply(extract_temp)
    df["Humidity"]    = df["{Temperature},{Humdity}"].apply(extract_humidity)
    df.drop(columns=["{Temperature},{Humdity}"], inplace=True)

# Wind Speed cleaning
def clean_wind(x):
    x = str(x).upper()
    if "KM/S" in x:
        value = float(re.sub(r"[^0-9.]", "", x))
        value = value * 3600
    elif "M/H" in x:
        value = float(re.sub(r"[^0-9.]", "", x))
        value = value / 100
    else:
        value = float(re.sub(r"[^0-9.]", "", x))
    return value

df["Wind Speed"] = df["Wind Speed"].apply(clean_wind)

# Precipitation cleaning
def clean_precip(x):
    x = str(x).upper()
    x = re.sub(r"[^0-9.]", "", x)
    return float(x)

df["Precipitation (%)"] = df["Precipitation (%)"].apply(clean_precip)

# deteksi & handle outlier (IQR method)
print("\n=== Outlier Detection (IQR) ===")
numeric_cols = ["Temperature", "Humidity", "Wind Speed", "Precipitation (%)", "Atmospheric Pressure", "UV Index", "Visibility (km)"]

for col in numeric_cols:
    if col not in df.columns: continue
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = ((df[col] < lower) | (df[col] > upper)).sum()
    print(f"{col}: {outliers} outlier → clip ke [{lower:.2f}, {upper:.2f}]")
    df[col] = df[col].clip(lower, upper)

# Missing value handling
for col in numeric_cols:
    if col in df.columns and df[col].isnull().sum() > 0:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"  {col}: imputed dengan median {median_val:.2f}")

# Label Encoding untuk target
encoder_target = LabelEncoder()
df["Weather Type"] = encoder_target.fit_transform(df["Weather Type"])

# Kategorikal nominal
cat_cols  = [c for c in ["Cloud Cover", "Season", "Location"] if c in df.columns]
num_cols  = [c for c in numeric_cols if c in df.columns]

print(f"\nKolom numerik  : {num_cols}")
print(f"Kolom kategorikal: {cat_cols}")

# One-Hot encode
df_cat = pd.get_dummies(df[cat_cols], prefix=cat_cols, drop_first=False)
df_num = df[num_cols].copy()
y = df["Weather Type"].copy()

df_clean = pd.concat([df_num, df_cat, y.rename("Weather Type")], axis=1)

df_clean.to_csv(OUTPUT_CSV, index=False)
print(df_clean.head())
    
with open(ENCODER_PKL, "wb") as f:
    pickle.dump({"target": encoder_target, "cat_cols": cat_cols}, f)
print(f"\nEncoder disimpan: {ENCODER_PKL}")