import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from sentence_transformers import CrossEncoder
from typing import List, Tuple

print("Loading cross-encoder re-ranker model...")
reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
print("Re-ranker model loaded!")

def rerank_chunks(
    query: str,
    chunks: List[str],
    metadatas: List[dict],
    original_scores: List[float],
    top_k: int = 5
) -> Tuple[List[str], List[dict], List[float]]:
    """Re-rank chunks using cross-encoder, return with normalized scores."""
    if not chunks:
        return [], [], []

    pairs = [[query, chunk] for chunk in chunks]
    raw_scores = reranker_model.predict(pairs)

    # Normalize scores to 0-100%
    min_s, max_s = min(raw_scores), max(raw_scores)
    if max_s - min_s > 0:
        normalized = [round((s - min_s) / (max_s - min_s) * 100, 1) for s in raw_scores]
    else:
        normalized = [100.0] * len(raw_scores)

    # Sort by score descending
    scored = sorted(
        zip(normalized, chunks, metadatas),
        key=lambda x: x[0],
        reverse=True
    )

    top_scores = [item[0] for item in scored[:top_k]]
    top_chunks = [item[1] for item in scored[:top_k]]
    top_metas = [item[2] for item in scored[:top_k]]

    print(f"Re-ranked {len(chunks)} chunks → top {top_k} selected")
    print(f"Top scores: {top_scores}")

    return top_chunks, top_metas, top_scores