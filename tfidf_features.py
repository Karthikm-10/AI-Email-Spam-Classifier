
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz

# Load cleaned dataset
df = pd.read_csv("data/processed/cleaned_emails.csv")

# Separate input text and target labels
X = df["cleaned_text"]
y = df["label"]

# Split dataset: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# Initialize TF-IDF vectorizer
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english",
    ngram_range=(1, 2)
)

# Learn vocabulary and transform training emails
X_train_tfidf = vectorizer.fit_transform(X_train)

# Transform testing emails using the same vocabulary
X_test_tfidf = vectorizer.transform(X_test)

print("\nTF-IDF conversion completed!")

print("Training feature shape:", X_train_tfidf.shape)
print("Testing feature shape:", X_test_tfidf.shape)

print("\nSample vocabulary:")
print(list(vectorizer.vocabulary_.keys())[:20])

# Create output directories
MODEL_DIR = Path("models")
PROCESSED_DIR = Path("data/processed")

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Save vectorizer and numerical features
joblib.dump(vectorizer, MODEL_DIR / "tfidf_vectorizer.pkl")

save_npz(PROCESSED_DIR / "X_train.npz", X_train_tfidf)
save_npz(PROCESSED_DIR / "X_test.npz", X_test_tfidf)

# Save corresponding labels
pd.Series(y_train).to_csv(
    PROCESSED_DIR / "y_train.csv",
    index=False,
    header=["label"]
)

pd.Series(y_test).to_csv(
    PROCESSED_DIR / "y_test.csv",
    index=False,
    header=["label"]
)

print("\nAll TF-IDF files saved successfully!")
