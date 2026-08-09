import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

import chromadb
from typing import List, Tuple
from embeddings import get_embeddings, get_single_embedding
from rank_bm25 import BM25Okapi

client = chromadb.PersistentClient(path=os.getenv("CHROMA_DB_PATH", "./data/chromadb"))

GLOBAL_COLLECTION = "documind_all_docs"

# In-memory BM25 index
bm25_index = None
bm25_chunks = []
bm25_metadatas = []

def get_collection():
    return client.get_or_create_collection(name=GLOBAL_COLLECTION)

def build_bm25_index():
    """Rebuild BM25 index from all stored chunks."""
    global bm25_index, bm25_chunks, bm25_metadatas

    collection = get_collection()
    result = collection.get(include=["documents", "metadatas"])

    if not result["documents"]:
        bm25_index = None
        return

    bm25_chunks = result["documents"]
    bm25_metadatas = result["metadatas"]

    tokenized = [doc.lower().split() for doc in bm25_chunks]
    bm25_index = BM25Okapi(tokenized)
    print(f"BM25 index built with {len(bm25_chunks)} chunks")

def store_chunks(doc_id: str, chunks: List[str]):
    """Store chunks with embeddings in ChromaDB and rebuild BM25 index."""
    collection = get_collection()
    embeddings = get_embeddings(chunks)
    ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": doc_id, "chunk_index": i} for i in range(len(chunks))]

    collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas
    )
    print(f"Stored {len(chunks)} chunks for doc: {doc_id}")

    # Rebuild BM25 index after new doc added
    build_bm25_index()

def semantic_search(query: str, top_k: int = 5, doc_ids: List[str] = None) -> Tuple[List[str], List[dict], List[float]]:
    """Pure semantic search using ChromaDB."""
    collection = get_collection()
    query_embedding = get_single_embedding(query)

    where_filter = {"source": {"$in": doc_ids}} if doc_ids else None

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where_filter,
        include=["documents", "metadatas", "distances"]
    )
    # Convert distances to similarity scores (0-100%)
    distances = results["distances"][0]
    scores = [round((1 - min(d, 1)) * 100, 1) for d in distances]
    return results["documents"][0], results["metadatas"][0], scores

def bm25_search(query: str, top_k: int = 5, doc_ids: List[str] = None) -> Tuple[List[str], List[dict], List[float]]:
    """BM25 keyword search."""
    if bm25_index is None:
        build_bm25_index()
    if bm25_index is None:
        return [], [], []

    tokenized_query = query.lower().split()
    scores = bm25_index.get_scores(tokenized_query)

    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k * 2]

    chunks = []
    metadatas = []
    bm25_scores = []
    for idx in top_indices:
        meta = bm25_metadatas[idx]
        if doc_ids and meta["source"] not in doc_ids:
            continue
        if scores[idx] > 0:
            chunks.append(bm25_chunks[idx])
            metadatas.append(meta)
            bm25_scores.append(round(float(scores[idx]), 2))
        if len(chunks) >= top_k:
            break

    # Normalize BM25 scores to 0-100%
    if bm25_scores:
        max_score = max(bm25_scores)
        bm25_scores = [round((s / max_score) * 100, 1) if max_score > 0 else 0 for s in bm25_scores]

    return chunks, metadatas, bm25_scores

def retrieve_similar_chunks(query: str, top_k: int = 5, doc_ids: List[str] = None) -> Tuple[List[str], List[dict], List[float]]:
    """Hybrid search — combine BM25 + semantic, then re-rank."""
    candidate_k = top_k * 3

    semantic_chunks, semantic_metas, semantic_scores = semantic_search(query, candidate_k, doc_ids)
    bm25_chunks_result, bm25_metas, bm25_scores = bm25_search(query, candidate_k, doc_ids)

    # Combine and deduplicate
    seen = set()
    combined_chunks = []
    combined_metas = []
    combined_scores = []

    for chunk, meta, score in zip(semantic_chunks, semantic_metas, semantic_scores):
        if chunk not in seen:
            seen.add(chunk)
            combined_chunks.append(chunk)
            combined_metas.append(meta)
            combined_scores.append(score)

    for chunk, meta, score in zip(bm25_chunks_result, bm25_metas, bm25_scores):
        if chunk not in seen:
            seen.add(chunk)
            combined_chunks.append(chunk)
            combined_metas.append(meta)
            combined_scores.append(score)

    # Re-rank
    from reranker import rerank_chunks
    reranked_chunks, reranked_metas, reranked_scores = rerank_chunks(
        query, combined_chunks, combined_metas, combined_scores, top_k
    )

    return reranked_chunks, reranked_metas, reranked_scores