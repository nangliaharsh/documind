import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from ragas.run_config import RunConfig
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

from langchain_groq import ChatGroq
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from datasets import Dataset
from rag_pipeline import query_documents
from vector_store import retrieve_similar_chunks

# Tell RAGAS to use Groq + HuggingFace instead of OpenAI
groq_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)
hf_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

ragas_llm = LangchainLLMWrapper(groq_llm)
ragas_embeddings = LangchainEmbeddingsWrapper(hf_embeddings)

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from datasets import Dataset
from rag_pipeline import query_documents
from vector_store import retrieve_similar_chunks

def run_evaluation(test_questions: list, ground_truths: list, doc_ids: list = None):
    """
    Run RAGAS evaluation on your RAG pipeline.
    test_questions: list of questions to ask
    ground_truths: list of expected correct answers
    """
    questions = []
    answers = []
    contexts = []
    truths = []

    print(f"\nRunning evaluation on {len(test_questions)} questions...\n")

    for i, (question, truth) in enumerate(zip(test_questions, ground_truths)):
        print(f"Q{i+1}: {question}")

        # Get answer from pipeline
        answer, sources = query_documents(question, [], doc_ids)

        # Get retrieved chunks for context
        chunks, metadatas = retrieve_similar_chunks(question, top_k=5, doc_ids=doc_ids)

        questions.append(question)
        answers.append(answer)
        contexts.append(chunks)   # list of retrieved chunks
        truths.append(truth)

        print(f"A{i+1}: {answer[:100]}...")
        print(f"Sources: {sources}\n")

    # Build RAGAS dataset
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": truths
    }
    dataset = Dataset.from_dict(data)

    # Run RAGAS evaluation
    print("Running RAGAS scoring...\n")
    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        run_config=RunConfig(max_workers=1, timeout=120)
    )

    return result