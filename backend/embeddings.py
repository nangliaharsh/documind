from sentence_transformers import SentenceTransformer
from typing import List

model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for a list of texts."""
    return model.encode(texts, convert_to_numpy=True).tolist()

def get_single_embedding(text: str) -> List[float]:
    """Generate embedding for a single query."""
    return model.encode([text], convert_to_numpy=True)[0].tolist()