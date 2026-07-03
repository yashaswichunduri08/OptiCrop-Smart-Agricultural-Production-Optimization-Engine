import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# ==========================
# Load Dataset
# ==========================

df = pd.read_csv("dataset/Crop_recommendation.csv")

print("First 5 Rows")
print(df.head())

print("\nDataset Shape")
print(df.shape)

print("\nDataset Info")
print(df.info())

print("\nMissing Values")
print(df.isnull().sum())

print("\nStatistical Summary")
print(df.describe())

# ==========================
# Crop Distribution
# ==========================

print("\nCrop Count")
print(df["label"].value_counts())

plt.figure(figsize=(12,6))
sns.countplot(x="label", data=df)
plt.xticks(rotation=90)
plt.title("Crop Distribution")
#plt.show()

# ==========================
# Correlation Heatmap
# ==========================

plt.figure(figsize=(8,6))
sns.heatmap(df.drop("label", axis=1).corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
#plt.show()

# ==========================
# Histograms
# ==========================

df.hist(figsize=(12,10))
#plt.show()

# ==========================
# Features & Target
# ==========================

X = df.drop("label", axis=1)
y = df["label"]

# ==========================
# Encode Labels
# ==========================

encoder = LabelEncoder()
y = encoder.fit_transform(y)

# ==========================
# Train Test Split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================
# Train Model
# ==========================

model = RandomForestClassifier(random_state=42)

model.fit(X_train, y_train)

# ==========================
# Prediction
# ==========================

y_pred = model.predict(X_test)

# ==========================
# Accuracy
# ==========================

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy :", accuracy)

print("\nClassification Report")
print(classification_report(y_test, y_pred))

# ==========================
# Save Model
# ==========================

# ==========================
# Save Model
# ==========================

import os

os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/crop_model.pkl")
joblib.dump(encoder, "model/label_encoder.pkl")

print("\nModel Saved Successfully!")