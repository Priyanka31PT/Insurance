# D:\projects\insurance\evaluate_pipeline.py
import os
import requests
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevance

def run_ragas_evaluation():
    print("Running evaluation on active Flask LangGraph Engine...")
    
    test_questions = [
        "What is the claim status for CLM300000?",
        "Does Car Shield Plus policy cover maternity expenses?"
    ]
    ground_truths = [
        "claim_id: CLM300000 is Approved.",
        "Maternity is Not applicable under Car Shield Plus."
    ]
    
    generated_answers = []
    contexts_collected = []
    
    for q in test_questions:
        try:
            res = requests.post("http://127.0.0", json={"query": q}).json()
            generated_answers.append(res.get("response", ""))
            contexts_collected.append([res.get("response", "")])
        except Exception:
            generated_answers.append("Error")
            contexts_collected.append([""])

    dataset = Dataset.from_dict({
        "question": test_questions,
        "answer": generated_answers,
        "contexts": contexts_collected,
        "ground_truth": ground_truths
    })
    
    # Evaluate 2 test items instantly instead of 1 million items
    result = evaluate(dataset=dataset, metrics=[faithfulness, answer_relevance])
    print("\n========== EVALUATION METRICS ==========\n", result)

if __name__ == "__main__":
    run_ragas_evaluation()
