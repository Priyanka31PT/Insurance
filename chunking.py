import re
from pathlib import Path
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Define folders
BASE_DIR = Path(__file__).resolve().parent
CLEANED_FOLDER = BASE_DIR / "cleaned"

# 1. Configure the Token-Aware Text Splitter strictly for TXT files
# Target ~300 tokens to easily sit inside BGE's 512-token limit
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    model_name="gpt-4o-mini",  # Used simply as a high-speed token counter
    chunk_size=300,
    chunk_overlap=50
)

all_chunks = []

def extract_primary_id(row_content: str) -> str:
    """Helper function to find claim_id, customer_id, or policy_id inside a CSV row string."""
    # Look for common patterns like CLM123456, CUST123456, POL123456
    match = re.search(r'(CLM|CUST|POL|HOSP)\d+', row_content, re.IGNORECASE)
    return match.group(0).upper() if match else "UNKNOWN"

def process_csv(file_path):
    """Load CSV rows as LangChain documents and enrich metadata for deterministic routing."""
    loader = CSVLoader(
        file_path=str(file_path),
        encoding="utf-8"
    )

    # Base documents loaded by row (e.g., page_content = "claim_id: CLM300001\nstatus: Under Review...")
    documents = loader.load()

    # CRITICAL FIX: Do NOT break the CSV row apart using a text splitter.
    # Keep 1 row = 1 chunk so columns stay physically coupled together.
    for doc in documents:
        # Extract unique ID strings directly out of the row context block
        id_ref = extract_primary_id(doc.page_content)
        
        # Inject metadata parameters required by your Regex/Deterministic Router
        doc.metadata["is_factual"] = True
        doc.metadata["id_reference"] = id_ref
        doc.metadata["source"] = file_path.name  # Standardize path string format

    print(f"CSV: {file_path.name}")
    print(f"  Rows loaded: {len(documents)}")
    print(f"  Chunks created: {len(documents)}")

    return documents

def process_txt(file_path):
    """Load and split text files using token-aware chunk rules."""
    text = file_path.read_text(encoding="utf-8")

    # Split using token boundaries
    chunks = text_splitter.create_documents(
        texts=[text],
        metadatas=[{
            "source": file_path.name,
            "is_factual": False,
            "data_type": "policy_text"
        }]
    )

    print(f"TXT: {file_path.name}")
    print(f"  Chunks created: {len(chunks)}")

    return chunks

def chunk_files():
    """Iterate through cleaned files and chunk them."""
    global all_chunks
    if not CLEANED_FOLDER.exists():
        print("Error: cleaned folder does not exist.")
        return []

    for file_path in CLEANED_FOLDER.rglob("*"):
        if not file_path.is_file():
            continue

        extension = file_path.suffix.lower()

        try:
            if extension == ".csv":
                chunks = process_csv(file_path)
            elif extension == ".txt":
                chunks = process_txt(file_path)
            else:
                print(f"Skipped unsupported file: {file_path.name}")
                continue

            all_chunks.extend(chunks)

        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")

    print("\n========== CHUNKING SUMMARY ==========")
    print(f"Total chunks created: {len(all_chunks)}")

    return all_chunks

if __name__ == "__main__":
    chunks = chunk_files()

    # Find and preview a CSV chunk to check metadata changes
    csv_sample = next((c for c in chunks if c.metadata.get("is_factual")), None)
    # Find and preview a Text chunk to check token changes
    txt_sample = next((c for c in chunks if not c.metadata.get("is_factual")), None)

    if csv_sample:
        print("\n========== SAMPLE ENRICHED CSV CHUNK ==========")
        print("Content:")
        print(csv_sample.page_content)
        print("\nMetadata (Note the structural IDs injected here):")
        print(csv_sample.metadata)

    if txt_sample:
        print("\n========== SAMPLE TOKEN-AWARE TEXT CHUNK ==========")
        print("Content:")
        print(txt_sample.page_content)
        print("\nMetadata:")
        print(txt_sample.metadata)
