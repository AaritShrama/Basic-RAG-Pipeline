"""
router.py
-----------------------------
Classifies each query into one of:
- "factual"      : answer exists directly in one chunk
- "synthesis"    : answer requires combining multiple chunks
- "out_of_scope" : documents don't contain enough info

Routing Logic (fully inspectable, 3-step):
  Step 1 — Check top similarity score against thresholds
           If best score < OOS_THRESHOLD → out_of_scope
  Step 2 — Check if query has synthesis intent (keywords)
           If yes AND enough good chunks → synthesis
  Step 3 — Otherwise → factual
"""

# CONSTANTS
OOS_THRESHOLD       = 0.45   # below this = not in documents
SYNTHESIS_THRESHOLD = 0.55   # min score to count a chunk as "useful"
MIN_SYNTHESIS_CHUNKS = 2     # need at least 2 good chunks for synthesis

# KEYWORDS FOR MULTIPLE RETRIEVALS
SYNTHESIS_KEYWORDS = [
    "compare", "comparison", "contrast", "difference", "differences",
    "across", "between", "both", "all documents", "overall",
    "summarize", "summary", "overview", "generally", "in general",
    "how do", "what are the", "what is the relationship",
    "similarities", "common", "trends", "multiple", "various"
]


def classify_query(query: str, top_chunks: list[dict]) -> str:
    
    best_score    = get_top_similarity_score(top_chunks)
    good_chunks   = count_good_chunks(top_chunks, threshold=SYNTHESIS_THRESHOLD)
    synthesis_intent = has_synthesis_intent(query)


    if best_score < OOS_THRESHOLD:
        return "out_of_scope"


    if synthesis_intent and good_chunks >= MIN_SYNTHESIS_CHUNKS:
        return "synthesis"

    return "factual"


def has_synthesis_intent(query: str) -> bool:
    query_lower = query.lower()
    return any(kw in query_lower for kw in SYNTHESIS_KEYWORDS)


def get_top_similarity_score(top_chunks: list[dict]) -> float:
    if not top_chunks:
        return 0.0
    return max(c["score"] for c in top_chunks)


def count_good_chunks(top_chunks: list[dict], threshold: float) -> int:
    return sum(1 for c in top_chunks if c["score"] >= threshold)


def explain_routing(query: str, top_chunks: list[dict]) -> dict:

    best_score       = get_top_similarity_score(top_chunks)
    good_chunks      = count_good_chunks(top_chunks, threshold=SYNTHESIS_THRESHOLD)
    synthesis_intent = has_synthesis_intent(query)
    decision         = classify_query(query, top_chunks)

    return {
        "query"            : query,
        "decision"         : decision,
        "best_score"       : round(best_score, 4),
        "good_chunks_count": good_chunks,
        "synthesis_intent" : synthesis_intent,
        "oos_threshold"    : OOS_THRESHOLD,
        "synthesis_threshold": SYNTHESIS_THRESHOLD,
    }



if __name__ == "__main__":
    from retriever import retrieve

    test_cases = [
        # Expected: factual
        "What does the EU AI Act say about high risk AI systems?",
        # Expected: synthesis
        "Compare how different documents approach AI risk categories.",
        # Expected: out_of_scope
        "What is the GDP of France in 2023?",
    ]

    print("=" * 55)
    print("  ROUTER TEST")
    print("=" * 55)

    for query in test_cases:
        chunks = retrieve(query, top_k=5)
        explanation = explain_routing(query, chunks)

        print(f"\nQuery    : {query}")
        print(f"Decision : {explanation['decision'].upper()}")
        print(f"Best Score     : {explanation['best_score']}")
        print(f"Good Chunks    : {explanation['good_chunks_count']}")
        print(f"Synthesis Intent: {explanation['synthesis_intent']}")
        print("-" * 55)