
import pandas as pd
import re

# Load the processed email dataset
df = pd.read_csv("data/processed/emails.csv")

print("Original email:")
print(df["text"].iloc[0][:300])


# Function to clean email text
def clean_text(text):

    # Convert text to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+", " ", text)

    # Remove punctuation and special characters
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# Apply cleaning to all emails
df["cleaned_text"] = df["text"].apply(clean_text)

# Remove empty messages after cleaning
df = df[df["cleaned_text"].str.strip() != ""]

# Save cleaned dataset
df.to_csv("data/processed/cleaned_emails.csv", index=False)

print("\nCleaned email:")
print(df["cleaned_text"].iloc[0][:300])

print("\nDataset shape:", df.shape)

print("\nCleaned dataset saved successfully!")
