import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

from groq import Groq
from typing import List
from vector_store import retrieve_similar_chunks
from langchain.memory import ConversationBufferWindowMemory

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Global memory store per session
memory_store = {}

def get_memory(session_id: str) -> ConversationBufferWindowMemory:
    """Get or create memory for a session."""
    if session_id not in memory_store:
        memory_store[session_id] = ConversationBufferWindowMemory(
            k=5,
            return_messages=True,
            memory_key="chat_history"
        )
    return memory_store[session_id]

def clear_memory(session_id: str):
    """Clear memory for a session."""
    if session_id in memory_store:
        del memory_store[session_id]

def build_prompt(query: str, chunks: List[str], metadatas: List[dict], chat_history: str = "") -> str:
    context_parts = []
    for i, (chunk, meta) in enumerate(zip(chunks, metadatas)):
        context_parts.append(f"[Source: {meta['source']} | Chunk {meta['chunk_index']}]\n{chunk}")
    context = "\n\n---\n\n".join(context_parts)

    return f"""You are a helpful document assistant with memory of the conversation.
Answer questions based ONLY on the context provided below.
Each context block has a [Source] label — always cite the source document name in your answer.
If the answer is not in the context, say "I couldn't find that in the uploaded documents."

CONVERSATION HISTORY:
{chat_history if chat_history else "No previous conversation."}

CONTEXT:
{context}

QUESTION: {query}

When answering, end with a "Sources:" line listing which documents you used."""

def query_documents(query: str, session_id: str = "default", doc_ids: List[str] = None):
    """RAG pipeline with memory, multi-doc support, citation verification and query reformulation."""
    from citation_verifier import verify_all_citations
    from query_reformulator import reformulate_query

    # Get memory for this session
    memory = get_memory(session_id)

    # Load chat history from memory
    history_messages = memory.load_memory_variables({})
    chat_history_text = ""
    if history_messages.get("chat_history"):
        for msg in history_messages["chat_history"]:
            role = "User" if msg.type == "human" else "Assistant"
            chat_history_text += f"{role}: {msg.content}\n"

    # Query reformulation pass
    reformulation = reformulate_query(query, chat_history_text)
    search_query = reformulation["reformulated"]

    # Retrieve relevant chunks using reformulated query
    chunks, metadatas, scores = retrieve_similar_chunks(search_query, top_k=5, doc_ids=doc_ids)

    if not chunks:
        return "No documents found. Please upload a PDF first.", [], [], [], {}

    # Build prompt with history
    prompt = build_prompt(query, chunks, metadatas, chat_history_text)

    # Call Groq
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024
    )
    answer = response.choices[0].message.content

    # Save to memory
    memory.save_context(
        {"input": query},
        {"output": answer}
    )

    # Citation verification pass
    print("\nRunning citation verification...")
    citation_results = verify_all_citations(answer, chunks, metadatas)

    # Only show verified sources in main sources list
    verified_sources = list(set(
        r["source"] for r in citation_results if r["verified"]
    ))
    all_sources = list(set(m["source"] for m in metadatas))

    # Build explainability data with verification
    explainability = [
        {
            "chunk_preview": chunk[:120] + "..." if len(chunk) > 120 else chunk,
            "source": meta["source"],
            "chunk_index": meta["chunk_index"],
            "relevance_score": score,
            "verified": citation_results[i]["verified"],
            "reason": citation_results[i]["reason"]
        }
        for i, (chunk, meta, score) in enumerate(zip(chunks, metadatas, scores))
    ]

    return answer, verified_sources, explainability, citation_results, reformulation

def summarize_document(doc_id: str) -> str:
    """Generate a summary of the document using retrieved chunks."""
    chunks, metadatas, _ = retrieve_similar_chunks(
        query="What is this document about? Summarize the main topics and key points.",
        top_k=5,
        doc_ids=[doc_id]
    )

    if not chunks:
        return "No content found to summarize."

    context = "\n\n".join(chunks)

    prompt = f"""You are a document summarizer.
Based on the following document content, write a concise 3-4 line summary covering:
- What the document is about
- Key topics or sections
- Most important information

DOCUMENT CONTENT:
{context}

Write only the summary, no preamble."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200
    )
    return response.choices[0].message.content