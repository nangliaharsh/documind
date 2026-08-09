import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def reformulate_query(query: str, chat_history: str = "") -> dict:
    """
    Analyze and reformulate a query if it's vague or unclear.
    Returns original query, reformulated query, and whether it was changed.
    """

    prompt = f"""You are a search query optimizer for a document Q&A system.

Your job:
1. Analyze if the query is clear and specific enough for document retrieval
2. If vague, ambiguous, or too short — rewrite it to be more specific
3. If already clear and specific — return it unchanged

A query needs reformulation if it:
- Is too vague ("tell me about it", "explain", "what about this?")
- Uses unclear pronouns without context ("what did he do?", "explain that")
- Is too short to be meaningful ("summary", "more", "continue")
- Could benefit from expansion based on conversation history

CONVERSATION HISTORY:
{chat_history if chat_history else "No previous conversation."}

ORIGINAL QUERY: {query}

Respond in this exact format:
NEEDS_REFORMULATION: YES or NO
REFORMULATED: the rewritten query (or original if no change needed)
REASON: one sentence explanation

Do not add anything else."""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150
    )

    result = response.choices[0].message.content.strip()

    # Parse response
    needs_reformulation = False
    reformulated = query
    reason = "Query was clear"

    for line in result.split("\n"):
        if line.startswith("NEEDS_REFORMULATION:"):
            needs_reformulation = "YES" in line.upper()
        elif line.startswith("REFORMULATED:"):
            reformulated = line.replace("REFORMULATED:", "").strip()
        elif line.startswith("REASON:"):
            reason = line.replace("REASON:", "").strip()

    if needs_reformulation:
        print(f"Query reformulated: '{query}' → '{reformulated}'")
        print(f"Reason: {reason}")
    else:
        print(f"Query unchanged: '{query}'")

    return {
        "original": query,
        "reformulated": reformulated,
        "was_reformulated": needs_reformulation,
        "reason": reason
    }