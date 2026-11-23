
import pandas as pd
import joblib

from pathlib import Path
from scipy.sparse import load_npz

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Load training and testing features
X_train = load_npz("data/processed/X_train.npz")
X_test = load_npz("data/processed/X_test.npz")

# Load corresponding labels
y_train = pd.read_csv("data/processed/y_train.csv")["label"]
y_test = pd.read_csv("data/processed/y_test.csv")["label"]

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

# Initialize the Naive Bayes model
model = MultinomialNB()

# Train the model
print("\nTraining model...")

model.fit(X_train, y_train)

print("Model training completed!")

# Predict labels for test emails
y_pred = model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(f"{accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save trained model
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

joblib.dump(model, MODEL_DIR / "spam_classifier.pkl")

print("\nTrained model saved successfully!")
