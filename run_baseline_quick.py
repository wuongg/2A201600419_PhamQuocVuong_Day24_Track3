"""Quick baseline eval."""
import os, sys
os.makedirs("reports", exist_ok=True)

from src.m1_chunking import load_documents, chunk_basic
from src.m2_search import DenseSearch
from src.m4_eval import load_test_set, evaluate_ragas, save_report
from config import NAIVE_COLLECTION

docs = load_documents()
chunks = []
for doc in docs:
    for c in chunk_basic(doc["text"], metadata=doc["metadata"]):
        chunks.append({"text": c.text, "metadata": c.metadata})
print(f"  {len(chunks)} basic chunks")

search = DenseSearch()
search.index(chunks, collection=NAIVE_COLLECTION)

test_set = load_test_set()
questions, answers, all_contexts, ground_truths = [], [], [], []
for item in test_set:
    results = search.search(item["question"], top_k=3, collection=NAIVE_COLLECTION)
    contexts = [r.text for r in results]
    answers.append(contexts[0] if contexts else "Không tìm thấy.")
    questions.append(item["question"])
    all_contexts.append(contexts)
    ground_truths.append(item["ground_truth"])

print("Running RAGAS baseline...")
results = evaluate_ragas(questions, answers, all_contexts, ground_truths)
print("\nBASELINE SCORES")
for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
    print(f"  {m}: {results.get(m, 0):.4f}")
save_report(results, [], path="reports/naive_baseline_report.json")
