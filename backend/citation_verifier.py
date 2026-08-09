import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

from groq import Groq
from typing import List

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def verify_citation(answer: str, chunk: str, source: str) -> dict:
    """
    Verify if a chunk actually supports the generated answer.
    Returns verification result with status and reason.
    """
    prompt = f"""You are a citation verifier. Your job is to check if a source chunk actually supports a given answer.

ANSWER:
{answer}

SOURCE CHUNK (from {source}):
{chunk}

Does this source chunk contain information that directly supports the answer above?

Respond in this exact format:
VERDICT: YES or NO
REASON: One sentence explanation

Do not add anything else."""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100
    )

    result = response.choices[0].message.content.strip()

    # Parse verdict
    lines = result.split("\n")
    verdict = "NO"
    reason = "Could not verify"

    for line in lines:
        if line.startswith("VERDICT:"):
            verdict = "YES" if "YES" in line.upper() else "NO"
        if line.startswith("REASON:"):
            reason = line.replace("REASON:", "").strip()

    return {
        "source": source,
        "verdict": verdict,
        "reason": reason,
        "verified": verdict == "YES"
    }

def verify_all_citations(answer: str, chunks: List[str], metadatas: List[dict]) -> List[dict]:
    """Verify all citations used in an answer."""
    results = []
    for chunk, meta in zip(chunks, metadatas):
        result = verify_citation(answer, chunk, meta["source"])
        result["chunk_preview"] = chunk[:120] + "..." if len(chunk) > 120 else chunk
        result["chunk_index"] = meta["chunk_index"]
        results.append(result)
        print(f"Citation [{meta['source']} Chunk {meta['chunk_index']}]: {result['verdict']} — {result['reason']}")
    return results