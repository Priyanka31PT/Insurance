
from pathlib import Path
import pandas as pd
import re
import unicodedata

# Define folders
BASE_DIR = Path(__file__).resolve().parent
UPLOADS_FOLDER = BASE_DIR / "uploads"
CLEANED_FOLDER = BASE_DIR / "cleaned"

CLEANED_FOLDER.mkdir(parents=True, exist_ok=True)

def clean_csv(file_path):
    """Clean CSV files."""
    try:
        df = pd.read_csv(file_path)

        # Remove completely empty rows and columns
        df = df.dropna(how="all")
        df = df.dropna(axis=1, how="all")

        # Remove extra spaces from column names
        df.columns = df.columns.astype(str).str.strip()

        # Clean text columns
        for col in df.select_dtypes(include=["object", "string"]):
            df[col] = (
                df[col]
                .astype("string")
                .str.replace(r"\s+", " ", regex=True)
                .str.strip()
            )

        # Convert whitespace-only cells to missing values
        df = df.replace(r"^\s*$", pd.NA, regex=True)

        # Remove exact duplicate rows
        df = df.drop_duplicates()

        # Save cleaned CSV
        output_path = CLEANED_FOLDER / file_path.name
        df.to_csv(output_path, index=False, encoding="utf-8")

        print(f"Cleaned CSV: {file_path.name}")

    except Exception as e:
        print(f"Error processing {file_path.name}: {e}")


def clean_txt(file_path):
    """Clean text files."""
    try:
        # Read text file
        text = file_path.read_text(encoding="utf-8-sig")

        # Normalize Unicode characters
        text = unicodedata.normalize("NFC", text)

        # Remove unwanted control characters
        text = "".join(
            char for char in text
            if char in "\n\t" or ord(char) >= 32
        )

        # Standardize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Remove extra spaces from each line
        lines = [line.strip() for line in text.split("\n")]
        text = "\n".join(lines)

        # Collapse excessive blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Save cleaned text
        output_path = CLEANED_FOLDER / file_path.name
        output_path.write_text(text.strip(), encoding="utf-8")

        print(f"Cleaned TXT: {file_path.name}")

    except Exception as e:
        print(f"Error processing {file_path.name}: {e}")


def preprocess_files():
    """Loop through uploads and clean supported files."""
    if not UPLOADS_FOLDER.exists():
        print("Uploads folder does not exist!")
        return

    for file_path in UPLOADS_FOLDER.iterdir():
        if not file_path.is_file():
            continue

        extension = file_path.suffix.lower()

        if extension == ".csv":
            clean_csv(file_path)

        elif extension == ".txt":
            clean_txt(file_path)

        else:
            print(f"Skipped unsupported file: {file_path.name}")

    print("Preprocessing completed!")


if __name__ == "__main__":
    preprocess_files()
