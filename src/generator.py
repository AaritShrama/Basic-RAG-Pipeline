"""
generator.py
------------
Handles answer generation using Groq LLM.

- factual queries    : answer grounded in top 2 chunks
- synthesis queries  : combine top 5 chunks into a unified answer
- out_of_scope       : return fixed decline message, NO LLM call ever
"""

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# CONSTANTS
GROQ_MODEL     = "llama-3.1-8b-instant"
MAX_TOKENS     = 512
TEMPERATURE    = 0.2   # low = more grounded, less creative

DECLINE_MESSAGE = (
    "The provided documents do not contain sufficient information "
    "to answer this question. Please refer to other sources."
)


_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# PROMPT
def _build_factual_prompt(query: str, chunks: list[dict]) -> str:

    context = "\n\n".join(
        f"[Source: {c['source']}]\n{c['chunk_text']}"
        for c in chunks[:2]
    )
    return f"""You are a precise assistant answering questions about AI regulation documents.

Use ONLY the context below to answer the question.
Do NOT use any outside knowledge.
If the context does not contain the answer, say "The documents do not contain this information."
Keep your answer concise and factual.

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""


def _build_synthesis_prompt(query: str, chunks: list[dict]) -> str:
   
    context = "\n\n".join(
        f"[Source: {c['source']}]\n{c['chunk_text']}"
        for c in chunks[:5]
    )
    return f"""You are an analytical assistant working with AI regulation documents.

Multiple document excerpts are provided below.
Synthesize the information across ALL sources to answer the question.
Mention which documents agree, differ, or complement each other where relevant.
Do NOT use any outside knowledge — only what is in the context.

CONTEXT:
{context}

QUESTION: {query}

SYNTHESIZED ANSWER:"""



def _call_groq(prompt: str) -> str:
    """Send prompt to Groq and return the response text."""
    response = _client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
    )
    return response.choices[0].message.content.strip()


def answer_factual(query: str, chunks: list[dict]) -> str:
    """Generate a grounded factual answer from top chunks."""
    prompt = _build_factual_prompt(query, chunks)
    return _call_groq(prompt)


def answer_synthesis(query: str, chunks: list[dict]) -> str:
    """Synthesize an answer across multiple chunks."""
    prompt = _build_synthesis_prompt(query, chunks)
    return _call_groq(prompt)


def answer_out_of_scope() -> str:
    """Return decline message. NEVER calls the LLM."""
    return DECLINE_MESSAGE



def generate_answer(query: str, chunks: list[dict], query_type: str) -> str:
   
    if query_type == "factual":
        return answer_factual(query, chunks)
    elif query_type == "synthesis":
        return answer_synthesis(query, chunks)
    elif query_type == "out_of_scope":
        return answer_out_of_scope()
    else:
        raise ValueError(f"Unknown query_type: '{query_type}'")



if __name__ == "__main__":
    from retriever import retrieve
    from router import classify_query, explain_routing

    test_cases = [
        "What does the EU AI Act say about high risk AI systems?",
        "Compare how different documents approach AI risk categories.",
        "What is the GDP of France in 2023?",
    ]

    print("=" * 60)
    print("  GENERATOR TEST")
    print("=" * 60)

    for query in test_cases:
        chunks     = retrieve(query, top_k=5)
        query_type = classify_query(query, chunks)
        answer     = generate_answer(query, chunks, query_type)
        explanation = explain_routing(query, chunks)

        print(f"\nQuery      : {query}")
        print(f"Route      : {query_type.upper()}")
        print(f"Best Score : {explanation['best_score']}")
        print(f"Answer     :\n{answer}")
        print("-" * 60)