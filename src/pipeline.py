"""Production RAG Pipeline — Bài tập NHÓM: ghép M1+M2+M3+M4."""

import os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.m1_chunking import load_documents, chunk_hierarchical
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import (
    load_test_set,
    evaluate_ragas,
    failure_analysis,
    failure_cluster_analysis,
    attach_distributions,
    distribution_breakdown,
    save_report,
)
from src.m5_enrichment import enrich_chunks
from src.llm_client import get_openai_compat_client, chat_completion_model
from config import (
    RERANK_TOP_K,
    SKIP_M5_ENRICHMENT,
    PIPELINE_MAX_CONTEXT_CHARS,
    SKIP_CROSS_ENCODER_RERANK,
)


def _configure_stdio_utf8() -> None:
    """Windows console (cp1258/cp1252): tránh UnicodeEncodeError khi print tiếng Việt."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


_configure_stdio_utf8()


def build_pipeline():
    """Build production RAG pipeline."""
    print("=" * 60)
    print("PRODUCTION RAG PIPELINE")
    print("=" * 60)

    # Step 1: Load & Chunk (M1)
    print("\n[1/3] Chunking documents...")
    docs = load_documents()
    all_chunks = []
    for doc in docs:
        parents, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
        for child in children:
            all_chunks.append({"text": child.text, "metadata": {**child.metadata, "parent_id": child.parent_id}})
    print(f"  {len(all_chunks)} chunks from {len(docs)} documents")

    # Step 2: Enrichment (M5)
    if SKIP_M5_ENRICHMENT:
        print("\n[2/4] SKIP_M5_ENRICHMENT=1 — bỏ gọi LLM enrichment, giữ raw chunks")
    else:
        print("\n[2/4] Enriching chunks (M5)...")
        enriched = enrich_chunks(all_chunks, methods=["contextual", "hyqa", "metadata"])
        if enriched:
            all_chunks = [{"text": e.enriched_text, "metadata": e.auto_metadata} for e in enriched]
            print(f"  Enriched {len(enriched)} chunks")
        else:
            print("  [WARN] M5 not implemented — using raw chunks (fallback)")

    # Step 3: Index (M2)
    print("\n[3/4] Indexing (BM25 + Dense)...")
    search = HybridSearch()
    search.index(all_chunks)

    # Step 4: Reranker (M3)
    print("\n[4/4] Loading reranker...")
    if SKIP_CROSS_ENCODER_RERANK:
        print("  [INFO] SKIP_CROSS_ENCODER_RERANK=1 — không load cross-encoder (score-only top-k)")
    reranker = CrossEncoderReranker()

    return search, reranker


def run_query(query: str, search: HybridSearch, reranker: CrossEncoderReranker) -> tuple[str, list[str]]:
    """Run single query through pipeline."""
    results = search.search(query)
    docs = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results]
    reranked = reranker.rerank(query, docs, top_k=RERANK_TOP_K)
    contexts = [r.text for r in reranked] if reranked else [r.text for r in results[:3]]

    # LLM generation
    try:
        client = get_openai_compat_client()
        if client is None:
            raise RuntimeError("no_llm_client")
        context_str = "\n\n".join(contexts)
        cap = PIPELINE_MAX_CONTEXT_CHARS
        if cap > 0 and len(context_str) > cap:
            context_str = context_str[:cap].rsplit("\n\n", 1)[0] + "\n\n...[context truncated]"
        resp = client.chat.completions.create(
            model=chat_completion_model(),
            messages=[
                {"role": "system", "content": "Trả lời CHỈ dựa trên context được cung cấp. Trả lời ngắn gọn, đúng trọng tâm. Nếu context không có thông tin → nói 'Tài liệu không đề cập.'"},
                {"role": "user", "content": f"Context:\n{context_str}\n\nCâu hỏi: {query}"},
            ],
            temperature=0,
            max_tokens=300,
        )
        answer = resp.choices[0].message.content.strip()
    except Exception:
        answer = contexts[0] if contexts else "Không tìm thấy thông tin."
    return answer, contexts


def evaluate_pipeline(search: HybridSearch, reranker: CrossEncoderReranker):
    """Run evaluation on test set."""
    print("\n[Eval] Running queries...")
    test_set = load_test_set()
    questions, answers, all_contexts, ground_truths = [], [], [], []

    for i, item in enumerate(test_set):
        answer, contexts = run_query(item["question"], search, reranker)
        questions.append(item["question"])
        answers.append(answer)
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        print(f"  [{i+1}/{len(test_set)}] {item['question'][:50]}...")

    print("\n[Eval] Running RAGAS...")
    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)

    print("\n" + "=" * 60)
    print("PRODUCTION RAG SCORES")
    print("=" * 60)
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        s = results.get(m, 0)
        print(f"  {'+' if s >= 0.75 else '-'} {m}: {s:.4f}")

    per_q = results.get("per_question", [])
    attach_distributions(per_q, test_set)
    failures = failure_analysis(per_q, bottom_n=10)
    clusters = failure_cluster_analysis(failures)
    dist_agg = distribution_breakdown(per_q)
    os.makedirs("reports", exist_ok=True)
    save_report(
        results,
        failures,
        path="reports/ragas_report.json",
        failure_clusters=clusters,
        distribution_breakdown_dict=dist_agg,
    )
    return results


if __name__ == "__main__":
    start = time.time()
    search, reranker = build_pipeline()
    evaluate_pipeline(search, reranker)
    print(f"\nTotal: {time.time() - start:.1f}s")
