
from pathlib import Path
import pandas as pd
import re
import unicodedata

BASE_DIR = Path(__file__).resolve().parent
CLEANED_FOLDER = BASE_DIR / "cleaned"


def validate_csv(file_path):
    issues = []

    try:
        df = pd.read_csv(file_path)

        # Check completely empty rows and columns
        if df.isna().all(axis=1).any():
            issues.append("Contains completely empty rows")

        if df.isna().all(axis=0).any():
            issues.append("Contains completely empty columns")

        # Check duplicate rows
        if df.duplicated().any():
            issues.append(
                f"Contains {df.duplicated().sum()} duplicate rows"
            )

        # Check column names
        if any(str(col) != str(col).strip() for col in df.columns):
            issues.append("Column names contain extra spaces")

        # Check text values
        for col in df.select_dtypes(include=["object", "string"]):
            values = df[col].dropna().astype(str)

            if values.str.match(r"^\s|\s$").any():
                issues.append(
                    f"Column '{col}' contains leading/trailing spaces"
                )

            if values.str.contains(r"\s{2,}", regex=True).any():
                issues.append(
                    f"Column '{col}' contains repeated whitespace"
                )

        return issues

    except Exception as e:
        return [f"Cannot read CSV: {e}"]


def validate_txt(file_path):
    issues = []

    try:
        text = file_path.read_text(encoding="utf-8")

        # Check excessive blank lines
        if re.search(r"\n{3,}", text):
            issues.append("Contains excessive blank lines")

        # Check leading/trailing spaces on lines
        lines = text.splitlines()

        if any(line != line.strip() for line in lines):
            issues.append("Lines contain leading/trailing spaces")

        # Check unwanted control characters
        if any(
            unicodedata.category(char) == "Cc"
            and char not in "\n\t\r"
            for char in text
        ):
            issues.append("Contains unwanted control characters")

        # Check Unicode normalization
        if unicodedata.normalize("NFC", text) != text:
            issues.append("Text is not Unicode-normalized")

        return issues

    except Exception as e:
        return [f"Cannot read TXT: {e}"]


def validate_cleaned_files():
    if not CLEANED_FOLDER.exists():
        print("ERROR: cleaned folder does not exist.")
        return

    files = [
        file for file in CLEANED_FOLDER.iterdir()
        if file.is_file()
    ]

    if not files:
        print("No files found in cleaned folder.")
        return

    passed = 0
    failed = 0

    for file_path in files:
        extension = file_path.suffix.lower()

        if extension == ".csv":
            issues = validate_csv(file_path)
        elif extension == ".txt":
            issues = validate_txt(file_path)
        else:
            print(f"SKIPPED: {file_path.name} (unsupported file)")
            continue

        if issues:
            failed += 1
            print(f"\nCHECK FAILED: {file_path.name}")
            for issue in issues:
                print(f"  - {issue}")
        else:
            passed += 1
            print(f"\nCHECK PASSED: {file_path.name}")

    print("\n========== VALIDATION SUMMARY ==========")
    print(f"Files passed: {passed}")
    print(f"Files with issues: {failed}")
    print(f"Files checked: {passed + failed}")


if __name__ == "__main__":
    validate_cleaned_files()
