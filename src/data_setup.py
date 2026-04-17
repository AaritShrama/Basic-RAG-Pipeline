"""
data_setup.py
---------
Responsible for:
- Loading PDFs/TXT Files 
- Chunking text with a defined strategy (size + overlap)
- Embedding chunks using HuggingFace sentence-transformers
- Saving the FAISS index + chunk metadata to /vector_store
"""

import os
import json
import numpy as np
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# CONSTANTS
DATA_DIR        = "data"
VECTOR_STORE_DIR = "vector_store"
INDEX_PATH      = os.path.join(VECTOR_STORE_DIR, "index.faiss")
METADATA_PATH   = os.path.join(VECTOR_STORE_DIR, "metadata.json")

CHUNK_SIZE      = 500   # characters per chunk
CHUNK_OVERLAP   = 50    # overlap between consecutive chunks
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# DATA LOADING
def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    documents = []

    supported = (".pdf", ".txt")
    all_files = [f for f in os.listdir(data_dir) if f.endswith(supported)]
    if not all_files:
        raise FileNotFoundError(f"No PDF or TXT files found in '{data_dir}/'")

    for filename in all_files:
        filepath = os.path.join(data_dir, filename)

        if filename.endswith(".txt"):
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                full_text = f.read().strip()
        else:
            reader = PdfReader(filepath)
            full_text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    full_text += page_text + "\n"
            full_text = full_text.strip()

        if full_text:
            documents.append({"text": full_text, "source": filename})
            print(f"  Loaded: {filename} ({len(full_text)} chars)")
        else:
            print(f"  WARNING: No text extracted from {filename}")

    print(f"\n  Total documents loaded: {len(documents)}")
    return documents


# CHUNKING
def chunk_documents(
    documents: list[dict],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP
) -> list[dict]:
   
    chunks = []
    chunk_id = 0

    for doc in documents:
        text   = doc["text"]
        source = doc["source"]
        start  = 0

        while start < len(text):
            end        = start + chunk_size
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append({
                    "chunk_id":   chunk_id,
                    "chunk_text": chunk_text,
                    "source":     source,
                })
                chunk_id += 1

            # Move forward by (chunk_size - overlap) to create overlap
            start += chunk_size - overlap

    print(f"  Total chunks created: {len(chunks)}")
    return chunks


# EMBEDDING
def embed_and_index(chunks: list[dict]) -> None:
   
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

    print(f"\n  Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [c["chunk_text"] for c in chunks]

    print("  Embedding chunks...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    # Normalize for cosine similarity via inner product
    faiss.normalize_L2(embeddings)

    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # Inner Product = cosine after normalization
    index.add(embeddings)

    # Save FAISS index
    faiss.write_index(index, INDEX_PATH)
    print(f"\n  FAISS index saved to: {INDEX_PATH}")
    print(f"  Index contains {index.ntotal} vectors of dim {dim}")

    # Save chunk as JSON
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"  Metadata saved to: {METADATA_PATH}")


# main()
if __name__ == "__main__":
    print("=" * 50)
    print("  INGESTION PIPELINE")
    print("=" * 50)

    print("\n[1/3] Loading documents...")
    docs = load_documents()

    print("\n[2/3] Chunking documents...")
    chunks = chunk_documents(docs)

    print("\n[3/3] Embedding and indexing...")
    embed_and_index(chunks)

    print("\n  Ingestion complete!")