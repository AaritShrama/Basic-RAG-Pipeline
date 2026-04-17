"""
retriever.py
------------
Loads the saved FAISS index and chunk metadata.
Given a query, returns the top most similar chunks
with their similarity scores.
"""

import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# CONSTANTS
VECTOR_STORE_DIR = "vector_store"
INDEX_PATH       = os.path.join(VECTOR_STORE_DIR, "index.faiss")
METADATA_PATH    = os.path.join(VECTOR_STORE_DIR, "metadata.json")
EMBEDDING_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"

# LOAD DATA
print("  [retriever] Loading FAISS index and embedding model...")
_model    = SentenceTransformer(EMBEDDING_MODEL)
_index    = faiss.read_index(INDEX_PATH)
with open(METADATA_PATH, "r", encoding="utf-8") as f:
    _metadata = json.load(f)
print("  [retriever] Ready.")


# RETRIEVAL
def retrieve(query: str, top_k: int = 5) -> list[dict]:

    # Embed and normalize query (same as how chunks were indexed)
    query_embedding = _model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)

    # Search FAISS — returns scores and indices of top_k matches
    scores, indices = _index.search(query_embedding, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            # FAISS returns -1 if fewer results than top_k exist
            continue
        chunk = _metadata[idx].copy()
        chunk["score"] = float(score)
        results.append(chunk)

    return results



if __name__ == "__main__":
    test_query = "What are the risk categories in AI regulation?"
    print(f"\nQuery: {test_query}\n")

    results = retrieve(test_query, top_k=3)
    for i, r in enumerate(results):
        print(f"--- Result {i+1} ---")
        print(f"Source : {r['source']}")
        print(f"Score  : {r['score']:.4f}")
        print(f"Text   : {r['chunk_text'][:200]}...")
        print()