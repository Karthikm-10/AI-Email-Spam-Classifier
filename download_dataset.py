
from pathlib import Path
from urllib.request import urlopen
from io import BytesIO
import tarfile

# Folder where the dataset will be stored
RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Official SpamAssassin dataset URLs
datasets = {
    "easy_ham": "https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham.tar.bz2",
    "spam": "https://spamassassin.apache.org/old/publiccorpus/20030228_spam.tar.bz2"
}

for name, url in datasets.items():
    print(f"\nDownloading {name} dataset...")

    with urlopen(url) as response:
        file_data = BytesIO(response.read())

    print(f"Extracting {name} dataset...")

    with tarfile.open(fileobj=file_data, mode="r:bz2") as archive:
        archive.extractall(path=RAW_DIR, filter="data")

    print(f"{name} downloaded and extracted successfully!")

# Count extracted email files
for folder in ["easy_ham", "spam"]:
    folder_path = RAW_DIR / folder

    if folder_path.exists():
        files = [f for f in folder_path.rglob("*") if f.is_file()]
        print(f"{folder}: {len(files)} files")

print("\nDataset download completed!")
