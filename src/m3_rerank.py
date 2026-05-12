"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import os, sys, time
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K, SKIP_CROSS_ENCODER_RERANK


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None
        self._model_type = None  # "flag" or "cross"
        self._skip_heavy_model = SKIP_CROSS_ENCODER_RERANK

    def _load_model(self):
        """Load cross-encoder model — try FlagReranker first, then CrossEncoder."""
        if self._skip_heavy_model:
            self._model = None
            self._model_type = None
            return None
        if self._model is None:
            # Option A: FlagEmbedding FlagReranker
            try:
                from FlagEmbedding import FlagReranker
                self._model = FlagReranker(self.model_name, use_fp16=True)
                self._model_type = "flag"
                return self._model
            except (ImportError, Exception):
                pass

            # Option B: sentence_transformers CrossEncoder
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
                self._model_type = "cross"
                return self._model
            except (ImportError, Exception):
                pass

            # Fallback: no model available
            self._model = None
            self._model_type = None

        return self._model

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents: top-20 → top-k using cross-encoder scores."""
        if not documents:
            return []

        model = self._load_model()

        if model is None:
            # Fallback: return top-k by original score
            sorted_docs = sorted(documents, key=lambda d: d.get("score", 0), reverse=True)
            return [
                RerankResult(
                    text=doc["text"],
                    original_score=doc.get("score", 0.0),
                    rerank_score=doc.get("score", 0.0),
                    metadata=doc.get("metadata", {}),
                    rank=i
                )
                for i, doc in enumerate(sorted_docs[:top_k])
            ]

        # Build query-document pairs
        pairs = [(query, doc["text"]) for doc in documents]

        # Score with model
        if self._model_type == "flag":
            scores = model.compute_score(pairs)
        else:
            scores = model.predict(pairs)

        # Ensure scores is a list
        if not hasattr(scores, '__iter__'):
            scores = [scores]
        scores = list(scores)

        # Combine scores with documents and sort
        scored_docs = sorted(
            zip(scores, documents),
            key=lambda x: x[0],
            reverse=True
        )

        # Return top_k RerankResult
        results = []
        for i, (score, doc) in enumerate(scored_docs[:top_k]):
            results.append(RerankResult(
                text=doc["text"],
                original_score=doc.get("score", 0.0),
                rerank_score=float(score),
                metadata=doc.get("metadata", {}),
                rank=i
            ))

        return results


class FlashrankReranker:
    """Lightweight alternative (<5ms). Optional."""
    def __init__(self):
        self._model = None

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank using flashrank — fast, no GPU needed."""
        if not documents:
            return []

        try:
            from flashrank import Ranker, RerankRequest

            if self._model is None:
                self._model = Ranker()

            passages = [{"id": i, "text": doc["text"]} for i, doc in enumerate(documents)]
            request = RerankRequest(query=query, passages=passages)
            reranked = self._model.rerank(request)

            results = []
            for i, item in enumerate(reranked[:top_k]):
                orig_idx = item.get("id", i)
                orig_doc = documents[orig_idx] if orig_idx < len(documents) else documents[i]
                results.append(RerankResult(
                    text=item.get("text", orig_doc["text"]),
                    original_score=orig_doc.get("score", 0.0),
                    rerank_score=float(item.get("score", 0.0)),
                    metadata=orig_doc.get("metadata", {}),
                    rank=i
                ))
            return results

        except (ImportError, Exception):
            # Fallback to CrossEncoder
            return CrossEncoderReranker().rerank(query, documents, top_k=top_k)


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """
    Benchmark reranker latency over n_runs.

    Returns:
        {"avg_ms": float, "min_ms": float, "max_ms": float}
    """
    times = []

    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        elapsed_ms = (time.perf_counter() - start) * 1000
        times.append(elapsed_ms)

    avg_ms = sum(times) / len(times)
    return {
        "avg_ms": avg_ms,
        "min_ms": min(times),
        "max_ms": max(times)
    }


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")
