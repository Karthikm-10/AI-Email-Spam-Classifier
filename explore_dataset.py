
from pathlib import Path
from email import policy
from email.parser import BytesParser

import pandas as pd

# Dataset locations
RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)



def extract_email_text(file_path):
    """Extract subject and plain-text body from an email."""

    try:
        with open(file_path, "rb") as file:
            message = BytesParser(policy=policy.default).parse(file)

        subject = str(message.get("Subject", ""))

        body = ""

        if message.is_multipart():
            email_body = message.get_body(preferencelist=("plain",))

            if email_body:
                body = email_body.get_content()
        else:
            if message.get_content_type() == "text/plain":
                body = message.get_content()

    except (LookupError, UnicodeError):
        # Fallback for emails with unsupported encodings
        with open(file_path, "rb") as file:
            raw_content = file.read()

        body = raw_content.decode("utf-8", errors="replace")
        subject = ""

    except Exception as error:
        print(f"Could not read {file_path.name}: {error}")
        return ""

    return subject + "\n" + str(body)

    """Extract the subject and plain-text body from an email."""

    try:
        with open(file_path, "rb") as file:
            message = BytesParser(policy=policy.default).parse(file)

        subject = str(message.get("Subject", ""))

        body = ""

        # Prefer the plain-text part of the email
        if message.is_multipart():
            email_body = message.get_body(preferencelist=("plain",))

            if email_body:
                body = email_body.get_content()
        else:
            if message.get_content_type() == "text/plain":
                body = message.get_content()

        return subject + "\n" + str(body)

    except Exception as error:
        print(f"Could not read {file_path.name}: {error}")
        return ""


# Collect email data
emails = []

folders = {
    "easy_ham": "ham",
    "spam": "spam"
}

for folder_name, label in folders.items():

    folder_path = RAW_DIR / folder_name

    print(f"\nReading {label} emails...")

    for file_path in folder_path.iterdir():

        # Skip directories and README files
        if not file_path.is_file() or file_path.name.lower() == "readme":
            continue

        text = extract_email_text(file_path)

        if text.strip():
            emails.append({
                "text": text,
                "label": label
            })

# Convert the collected data into a DataFrame
df = pd.DataFrame(emails)

# Display dataset information
print("\nDataset loaded successfully!")

print("\nFirst five rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nLabel distribution:")
print(df["label"].value_counts())

print("\nMissing values:")
print(df.isnull().sum())

# Save processed dataset
output_file = OUTPUT_DIR / "emails.csv"
df.to_csv(output_file, index=False)

print(f"\nProcessed dataset saved to: {output_file}")

# Display one example
if not df.empty:
    print("\nSample email:")
    print(df.iloc[0]["text"][:500])
    print("\nSample label:", df.iloc[0]["label"])
