# Agentic RAG System — AI Regulation Documents

An agentic Retrieval-Augmented Generation (RAG) system that reasons about *how* to answer a query rather than blindly retrieving and generating. Built for Q&A over 4 AI regulation documents.

---

## What Makes This "Agentic"?

Most RAG systems do the same thing for every query: retrieve → generate. This system first **classifies the query** into one of three types and responds differently for each:

| Query Type | What It Means | What The System Does |
|---|---|---|
| `factual` | Answer exists directly in one chunk | Retrieves top 2 chunks, generates grounded answer |
| `synthesis` | Answer requires combining multiple sources | Retrieves top 5 chunks, synthesizes across documents |
| `out_of_scope` | Documents don't contain the answer | Returns a decline message, never calls the LLM |

---
## System Architecture

**Pipeline Overview:**

```
User Query
    │
    ▼
[ Retriever ] — Embeds query → searches FAISS index → returns top-k chunks with scores
    │
    ▼
[ Router ] — Explicit rule-based classification (NOT a black-box LLM decision)
    │
    ├── factual      ──→ [ Generator ] top 2 chunks, focused prompt
    │
    ├── synthesis    ──→ [ Generator ] top 5 chunks, synthesis prompt
    │
    └── out_of_scope ──→ Decline message (LLM is never called)
```
---

## Chunking Strategy

- **Method:** Fixed-size character chunking with overlap
- **Chunk size:** 500 characters
- **Overlap:** 50 characters

**Why this approach?**  
Fixed-size chunking keeps embeddings uniform and predictable. The 50-character overlap ensures that sentences split across chunk boundaries still retain enough context in the adjacent chunk. This is intentionally simple — the documents are regulation text, not narrative prose, so semantic chunking would add complexity without meaningful gain at this scale.

---

## Embedding Model

- **Model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Runs:** Fully locally, no API key required
- **Dimensions:** 384
- **Why:** Free, fast, well-tested on factual/technical text, and produces strong cosine similarity scores for regulatory language

---

## Routing Logic

The router is **fully rule-based and inspectable** — no LLM involved in the routing decision.

**Step 1 — Out of scope check:**  
If the best similarity score across all retrieved chunks is below `0.45`, the documents don't contain relevant information → `out_of_scope`

**Step 2 — Synthesis check:**  
If the query contains synthesis keywords (compare, contrast, summarize, across, difference, etc.) AND at least 2 chunks score above `0.55` → `synthesis`

**Step 3 — Factual (default):**  
Score is high enough and no synthesis intent detected → `factual`

**Thresholds:**
| Threshold | Value | Meaning |
|---|---|---|
| `OOS_THRESHOLD` | 0.45 | Below this = not in documents |
| `SYNTHESIS_THRESHOLD` | 0.55 | Min score to count a chunk as useful for synthesis |
| `MIN_SYNTHESIS_CHUNKS` | 2 | Need at least 2 good chunks for synthesis |

---

## Tech Stack

| Component | Choice | Reason |
|---|---|---|
| Embeddings | `all-MiniLM-L6-v2` | Free, local, no API key |
| Vector Store | FAISS (`IndexFlatIP`) | No server, simple file-based, fast |
| LLM | Groq (`llama3-8b-8192`) | Free tier, very fast inference |
| PDF/TXT Parsing | `pypdf` + native Python | Handles both file types |
| Evaluation | ROUGE-L + keyword overlap | Quantitative, reproducible |

---

## Local Setup

### 1. Clone the repo
```bash
git clone https://github.com/AaritShrama/Basic-RAG-Agent
cd Basic-RAG-Agent
```

### 2. Create and activate a virtual environment
```bash
# Windows
python -m venv myenv
myenv\Scripts\activate

# Mac/Linux
python -m venv myenv
source myenv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up your API key
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key_here
```
Get a free key at [console.groq.com](https://console.groq.com)

### 5. Add the documents
Place the 4 AI regulation documents in the `/data` folder.  
Download them from: [Google Drive Link](https://drive.google.com/drive/folders/18jlAr6bPEKHEL6km7dNKf-C6bjB4yTH9?usp=sharing)

### 6. Run ingestion (builds the vector store)
```bash
python src/ingest.py
```
This creates `vector_store/index.faiss` and `vector_store/metadata.json` locally.

---
_______________________________________________________________________________________
**AARIT**
