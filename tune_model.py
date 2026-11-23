
import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    make_scorer,
    f1_score
)

# Load cleaned dataset
df = pd.read_csv("data/processed/cleaned_emails.csv")

X = df["cleaned_text"]
y = df["label"]

# Recreate the same train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Create a complete ML pipeline
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        ngram_range=(1, 2)
    )),
    ("classifier", MultinomialNB())
])

# Score specifically for spam F1-score
spam_f1 = make_scorer(f1_score, pos_label="spam")

# Values of alpha to test
parameters = {
    "classifier__alpha": [0.01, 0.1, 0.5, 1, 2, 5, 10]
}

# Perform 5-fold cross-validation
grid_search = GridSearchCV(
    pipeline,
    parameters,
    scoring=spam_f1,
    cv=5,
    n_jobs=-1,
    verbose=1
)

print("Tuning model parameters...")

grid_search.fit(X_train, y_train)

# Get best model
best_model = grid_search.best_estimator_

print("\nBest alpha:", grid_search.best_params_)
print("Best cross-validation spam F1-score:",
      round(grid_search.best_score_, 4))

# Evaluate only once on held-out test data
y_pred = best_model.predict(X_test)

print("\nTest Accuracy:")
print(f"{accuracy_score(y_test, y_pred) * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=["ham", "spam"]))

# Save complete pipeline
Path("models").mkdir(exist_ok=True)

joblib.dump(best_model, "models/spam_pipeline.pkl")

print("\nBest pipeline saved successfully!")
