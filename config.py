"""Shared configuration for Lab 18."""

import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")

# --- Qdrant ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "lab18_production"
NAIVE_COLLECTION = "lab18_naive"

# --- Embedding ---
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

# --- Chunking ---
HIERARCHICAL_PARENT_SIZE = 2048
HIERARCHICAL_CHILD_SIZE = 256
SEMANTIC_THRESHOLD = 0.85

# --- Search ---
BM25_TOP_K = 20
DENSE_TOP_K = 20
HYBRID_TOP_K = 20
RERANK_TOP_K = 3

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TEST_SET_PATH = os.path.join(os.path.dirname(__file__), "test_set.json")

# --- Eval speed / cost ---
# Chỉ đánh giá N câu đầu (0 = hết test_set). VD: LAB_EVAL_LIMIT=8 để dev nhanh, rẻ.
LAB_EVAL_LIMIT = int(os.getenv("LAB_EVAL_LIMIT", "0") or "0")

# --- Pipeline robustness (Windows / LM Studio / thiếu VRAM) ---
# Bỏ M5 LLM enrichment — tránh hàng chục gọi API trước khi eval; vẫn là hybrid + rerank score-only.
SKIP_M5_ENRICHMENT = os.getenv("SKIP_M5_ENRICHMENT", "").lower() in ("1", "true", "yes")
# Không load BGE cross-encoder (tránh crash native / OOM); rerank = sort theo điểm hybrid.
SKIP_CROSS_ENCODER_RERANK = os.getenv("SKIP_CROSS_ENCODER_RERANK", "").lower() in ("1", "true", "yes")
# Cắt context gửi LLM (0 = không cắt). VD 12000 giảm lỗi "Context size exceeded" trên local.
PIPELINE_MAX_CONTEXT_CHARS = int(os.getenv("PIPELINE_MAX_CONTEXT_CHARS", "0") or "0")
