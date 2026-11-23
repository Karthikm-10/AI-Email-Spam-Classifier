
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split

# Load the cleaned dataset
df = pd.read_csv("data/processed/cleaned_emails.csv")

# Load our tuned model
model = joblib.load("models/spam_pipeline.pkl")

# Recreate the same train/test split
X = df["cleaned_text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Combine the actual test emails and their labels
test_data = pd.DataFrame({
    "text": X_test,
    "actual_label": y_test
})

# Select five actual spam and five actual ham emails
spam_samples = test_data[
    test_data["actual_label"] == "spam"
].sample(n=5, random_state=42)

ham_samples = test_data[
    test_data["actual_label"] == "ham"
].sample(n=5, random_state=42)

samples = pd.concat([spam_samples, ham_samples])

print("\nTesting actual unseen emails...\n")

for index, row in samples.iterrows():

    email_text = row["text"]

    prediction = model.predict([email_text])[0]

    probabilities = model.predict_proba([email_text])[0]

    probability_dict = dict(
        zip(model.classes_, probabilities)
    )

    print("-----------------------------------")
    print("Actual label:", row["actual_label"])
    print("Predicted label:", prediction)
    print("Spam probability:",
          round(probability_dict["spam"] * 100, 2), "%")
    print("Ham probability:",
          round(probability_dict["ham"] * 100, 2), "%")

print("\nTesting completed!")
