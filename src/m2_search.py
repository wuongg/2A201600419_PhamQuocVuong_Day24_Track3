"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os, sys, uuid
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, DATA_DIR,
                    EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """
    Segment Vietnamese text into words using underthesea.
    BM25 needs word boundaries — "nghỉ phép" = 1 word, not 2.
    """
    try:
        from underthesea import word_tokenize
        return word_tokenize(text, format="text")
    except ImportError:
        # Fallback: return text as-is if underthesea not installed
        return text


class BM25Search:
    def __init__(self):
        self.corpus_tokens: list[list[str]] = []
        self.documents: list[dict] = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        from rank_bm25 import BM25Okapi

        self.documents = chunks
        # Segment each chunk and tokenize
        self.corpus_tokens = [
            segment_vietnamese(chunk["text"]).split()
            for chunk in chunks
        ]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if self.bm25 is None or not self.documents:
            return []

        # Segment and tokenize query
        tokenized_query = segment_vietnamese(query).split()
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices sorted by score descending
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only return relevant results
                doc = self.documents[idx]
                results.append(SearchResult(
                    text=doc["text"],
                    score=float(scores[idx]),
                    metadata=doc.get("metadata", {}),
                    method="bm25"
                ))

        return results


class DenseSearch:
    def __init__(self, storage_slug: str = "default"):
        """
        storage_slug: thư mục nhúng tách riêng (naive vs hybrid) tránh lock khi 2 process Qdrant local.
        Ghi đè toàn bộ: env QDRANT_LOCAL_PATH (một đường dẫn tuyệt đối).
        """
        self.client = None
        self._storage_slug = storage_slug
        try:
            from qdrant_client import QdrantClient

            force_embedded = os.getenv("QDRANT_EMBEDDED", "").lower() in ("1", "true", "yes")
            if not force_embedded:
                try:
                    remote = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
                    remote.get_collections()
                    self.client = remote
                except Exception:
                    self.client = None
            if self.client is None:
                base_path = os.getenv("QDRANT_LOCAL_PATH") or os.path.join(
                    DATA_DIR, f".qdrant_storage_{storage_slug}"
                )
                path_try = base_path
                last_err: Exception | None = None
                for attempt in range(5):
                    try:
                        os.makedirs(path_try, exist_ok=True)
                        client_try = QdrantClient(path=path_try)
                        self.client = client_try
                        if path_try != base_path:
                            print(f"  [INFO] Embedded Qdrant — fallback path (lock avoided): {path_try}")
                        else:
                            print(f"  [INFO] Embedded Qdrant (no Docker): {path_try}")
                        break
                    except RuntimeError as e:
                        last_err = e
                        msg = str(e).lower()
                        if "already accessed" not in msg:
                            raise
                        path_try = os.path.join(
                            DATA_DIR,
                            f".qdrant_storage_{storage_slug}_{os.getpid()}_{uuid.uuid4().hex[:8]}",
                        )

                if self.client is None and last_err is not None:
                    raise last_err
        except Exception as e:
            print(f"  [WARN] Qdrant init failed ({type(e).__name__}): {e}")
            self.client = None
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            # Giảm crash subprocess/tokenizers trên Windows khi dùng cùng job embedding + LM Studio GPU.
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
            os.environ.setdefault("OMP_NUM_THREADS", "1")
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(EMBEDDING_MODEL)
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        if self.client is None:
            print("  [WARN] Qdrant not available — skipping dense indexing")
            return

        try:
            from qdrant_client.models import Distance, VectorParams, PointStruct

            # Recreate collection
            self.client.recreate_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
            )

            texts = [c["text"] for c in chunks]
            vectors = self._get_encoder().encode(texts, show_progress_bar=True)

            points = [
                PointStruct(
                    id=i,
                    vector=v.tolist(),
                    payload={**c.get("metadata", {}), "text": c["text"]}
                )
                for i, (v, c) in enumerate(zip(vectors, chunks))
            ]

            # Upload in batches
            batch_size = 100
            for i in range(0, len(points), batch_size):
                self.client.upsert(collection_name=collection, points=points[i:i + batch_size])

        except Exception as e:
            print(f"  [WARN] Dense indexing error: {e}")

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        if self.client is None:
            return []

        try:
            query_vector = self._get_encoder().encode(query).tolist()
            hits = self.client.search(
                collection_name=collection,
                query_vector=query_vector,
                limit=top_k
            )

            return [
                SearchResult(
                    text=hit.payload.get("text", ""),
                    score=hit.score,
                    metadata={k: v for k, v in hit.payload.items() if k != "text"},
                    method="dense"
                )
                for hit in hits
            ]
        except Exception as e:
            print(f"  [WARN] Dense search error: {e}")
            return []


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """
    Merge ranked lists using RRF: score(d) = Σ 1/(k + rank).
    Combines BM25 and dense rankings into a single hybrid ranking.
    """
    rrf_scores: dict[str, dict] = {}

    for result_list in results_list:
        for rank, result in enumerate(result_list):
            key = result.text
            if key not in rrf_scores:
                rrf_scores[key] = {"score": 0.0, "result": result}
            # RRF formula: 1 / (k + rank + 1)  (rank is 0-indexed)
            rrf_scores[key]["score"] += 1.0 / (k + rank + 1)

    # Sort by RRF score descending
    sorted_items = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)

    # Return top_k results with method="hybrid"
    merged = []
    for item in sorted_items[:top_k]:
        r = item["result"]
        merged.append(SearchResult(
            text=r.text,
            score=item["score"],
            metadata=r.metadata,
            method="hybrid"
        ))

    return merged


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch("hybrid")

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print(f"Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
