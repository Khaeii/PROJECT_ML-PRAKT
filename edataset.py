import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"dataset/weather data classification.csv")

print("5 Data Pertama")
print(df.head())

print("\nUkuran Dataset:")
print(df.shape)

print("\nInfo Dataset:")
print(df.info())

print("\nStatistik:")
print(df.describe())

print(df["Wind Speed"].unique()[:20])
print(df["Precipitation (%)"].unique()[:20])

print("\nMissing Value:")
print(df.isnull().sum())

print("\nDuplicate:")
print(df.duplicated().sum())

print("\nDistribusi Weather Type:")
print(df["Weather Type"].value_counts())

plt.figure(figsize=(6,4))
sns.countplot(data=df, x="Weather Type")
plt.title("Distribusi Weather Type")
plt.show()

df.hist(figsize=(12,8))
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.show()