
import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    make_scorer,
    f1_score
)

# Load dataset
df = pd.read_csv("data/processed/cleaned_emails.csv")

X = df["cleaned_text"]
y = df["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Build pipeline
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        ngram_range=(1, 2)
    )),
    ("classifier", MultinomialNB())
])

# Define models and parameters
parameter_grid = [
    {
        "classifier": [MultinomialNB()],
        "classifier__alpha": [0.01, 0.05, 0.1, 0.5, 1]
    },
    {
        "classifier": [ComplementNB()],
        "classifier__alpha": [0.01, 0.05, 0.1, 0.5, 1]
    },
    {
        "classifier": [
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced"
            )
        ],
        "classifier__C": [0.1, 1, 10]
    }
]

# Optimize spam F1-score
spam_f1 = make_scorer(f1_score, pos_label="spam")

search = GridSearchCV(
    pipeline,
    parameter_grid,
    scoring=spam_f1,
    cv=5,
    n_jobs=-1,
    verbose=1
)

print("Comparing machine learning models...")

search.fit(X_train, y_train)

# Best model
best_model = search.best_estimator_

print("\nBest parameters:")
print(search.best_params_)

print("\nBest cross-validation spam F1-score:")
print(round(search.best_score_, 4))

# Evaluate selected model on test data
y_pred = best_model.predict(X_test)

print("\nTest accuracy:")
print(f"{accuracy_score(y_test, y_pred) * 100:.2f}%")

print("\nClassification report:")
print(classification_report(y_test, y_pred))

print("\nConfusion matrix:")
print(confusion_matrix(
    y_test,
    y_pred,
    labels=["ham", "spam"]
))

# Save the selected pipeline separately
Path("models").mkdir(exist_ok=True)

joblib.dump(
    best_model,
    "models/spam_pipeline_comparison.pkl"
)

print("\nBest comparison model saved!")
