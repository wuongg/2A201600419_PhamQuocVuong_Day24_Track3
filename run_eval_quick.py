"""Quick eval script — chỉ chạy production pipeline với full 20 câu."""
import os, sys, time, json

os.makedirs("reports", exist_ok=True)

from src.m1_chunking import load_documents, chunk_hierarchical
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import load_test_set, evaluate_ragas, failure_analysis, save_report
from config import RERANK_TOP_K
from openai import OpenAI

client = OpenAI()

def run_query(query, search, reranker):
    results = search.search(query)
    docs = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results]
    reranked = reranker.rerank(query, docs, top_k=RERANK_TOP_K)
    contexts = [r.text for r in reranked] if reranked else [r.text for r in results[:3]]

    context_str = "\n\n".join(contexts)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Trả lời CHỈ dựa trên context. Ngắn gọn, đúng trọng tâm. Nếu không có thông tin → nói 'Tài liệu không đề cập.'"},
            {"role": "user", "content": f"Context:\n{context_str}\n\nCâu hỏi: {query}"},
        ],
        temperature=0, max_tokens=200,
    )
    return resp.choices[0].message.content.strip(), contexts

# Build pipeline
print("=== Building pipeline ===")
docs = load_documents()
all_chunks = []
for doc in docs:
    parents, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
    for child in children:
        all_chunks.append({"text": child.text, "metadata": {**child.metadata, "parent_id": child.parent_id}})
print(f"  {len(all_chunks)} chunks from {len(docs)} docs")

search = HybridSearch()
search.index(all_chunks)
reranker = CrossEncoderReranker()

# Run queries
print("\n=== Running queries ===")
test_set = load_test_set()
questions, answers, all_contexts, ground_truths = [], [], [], []
for i, item in enumerate(test_set):
    ans, ctx = run_query(item["question"], search, reranker)
    questions.append(item["question"])
    answers.append(ans)
    all_contexts.append(ctx)
    ground_truths.append(item["ground_truth"])
    print(f"  [{i+1}/{len(test_set)}] Q: {item['question'][:50]}")
    print(f"         A: {ans[:80]}")

# RAGAS eval
print("\n=== Running RAGAS ===")
results = evaluate_ragas(questions, answers, all_contexts, ground_truths)

print("\n" + "="*50)
print("PRODUCTION SCORES")
print("="*50)
for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
    s = results.get(m, 0)
    print(f"  {'✓' if s >= 0.75 else '✗'} {m}: {s:.4f}")

failures = failure_analysis(results.get("per_question", []))
save_report(results, failures, path="reports/ragas_report.json")
print("\nDone! Report saved to reports/ragas_report.json")
