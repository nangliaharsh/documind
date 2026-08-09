import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

from evaluate import run_evaluation

# ──────────────────────────────────────────────
# EDIT THESE based on your uploaded PDFs
# Add 5-10 question/answer pairs from your docs
# ──────────────────────────────────────────────

test_questions = [
    "What is the candidate's name?",
    "What programming languages does the candidate know?",
    "Where did the candidate work most recently?",
]

ground_truths = [
    "The candidate's name is Paras Kaushik.",
    "The candidate knows programming languages including Python, and has experience with AI and ML technologies.",
    "The candidate most recently worked at Infutrix as an AI Engineer starting from June 2025.",
]

# ──────────────────────────────────────────────

if __name__ == "__main__":
    results = run_evaluation(test_questions, ground_truths)
    
    print("\n" + "="*50)
    print("RAGAS EVALUATION RESULTS")
    print("="*50)
    print(f"Faithfulness:        {results['faithfulness']:.2%}")
    print(f"Answer Relevancy:    {results['answer_relevancy']:.2%}")
    print(f"Context Precision:   {results['context_precision']:.2%}")
    print(f"Context Recall:      {results['context_recall']:.2%}")
    
    avg = (
        results['faithfulness'] +
        results['answer_relevancy'] +
        results['context_precision'] +
        results['context_recall']
    ) / 4
    
    print(f"\nOverall Score:       {avg:.2%}")
    print("="*50)
    print("\nCopy your best metric for your resume bullet!")