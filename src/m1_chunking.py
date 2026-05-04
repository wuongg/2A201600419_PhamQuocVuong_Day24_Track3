"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import os, sys, glob, re
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load all markdown/text files from data/. (Đã implement sẵn)"""
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})
    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.

    Args:
        text: Input text.
        threshold: Cosine similarity threshold. Dưới threshold → tách chunk mới.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        List of Chunk objects grouped by semantic similarity.
    """
    metadata = metadata or {}

    # 1. Split text into sentences
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n\n', text) if s.strip()]
    if not sentences:
        return []

    if len(sentences) == 1:
        return [Chunk(text=sentences[0], metadata={**metadata, "chunk_index": 0, "strategy": "semantic"})]

    # 2. Encode sentences using a lightweight model
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np

        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(sentences)

        def cosine_sim(a, b):
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return float(np.dot(a, b) / (norm_a * norm_b))

        # 3. Group sentences by similarity
        chunks = []
        current_group = [sentences[0]]

        for i in range(1, len(sentences)):
            sim = cosine_sim(embeddings[i - 1], embeddings[i])
            if sim < threshold:
                # Start new chunk
                chunks.append(Chunk(
                    text=" ".join(current_group),
                    metadata={**metadata, "chunk_index": len(chunks), "strategy": "semantic"}
                ))
                current_group = []
            current_group.append(sentences[i])

        # Don't forget last group
        if current_group:
            chunks.append(Chunk(
                text=" ".join(current_group),
                metadata={**metadata, "chunk_index": len(chunks), "strategy": "semantic"}
            ))

        return chunks

    except ImportError:
        # Fallback: split by paragraph if sentence_transformers not available
        return chunk_basic(text, metadata=metadata)


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Args:
        text: Input text.
        parent_size: Chars per parent chunk.
        child_size: Chars per child chunk.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    metadata = metadata or {}
    parents: list[Chunk] = []
    children: list[Chunk] = []

    # 1. Split text into parents by accumulating paragraphs up to parent_size
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    current_text = ""
    p_index = 0

    for para in paragraphs:
        if len(current_text) + len(para) > parent_size and current_text:
            # Save current parent
            pid = f"parent_{p_index}"
            parent = Chunk(
                text=current_text.strip(),
                metadata={**metadata, "chunk_type": "parent", "parent_id": pid}
            )
            parents.append(parent)

            # 2. Split parent into children using sliding window
            parent_text = current_text.strip()
            c_start = 0
            while c_start < len(parent_text):
                child_text = parent_text[c_start:c_start + child_size]
                if child_text.strip():
                    child = Chunk(
                        text=child_text.strip(),
                        metadata={**metadata, "chunk_type": "child"},
                        parent_id=pid
                    )
                    children.append(child)
                c_start += child_size

            p_index += 1
            current_text = ""

        current_text += para + "\n\n"

    # Handle remaining text
    if current_text.strip():
        pid = f"parent_{p_index}"
        parent = Chunk(
            text=current_text.strip(),
            metadata={**metadata, "chunk_type": "parent", "parent_id": pid}
        )
        parents.append(parent)

        parent_text = current_text.strip()
        c_start = 0
        while c_start < len(parent_text):
            child_text = parent_text[c_start:c_start + child_size]
            if child_text.strip():
                child = Chunk(
                    text=child_text.strip(),
                    metadata={**metadata, "chunk_type": "child"},
                    parent_id=pid
                )
                children.append(child)
            c_start += child_size

    return parents, children


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.

    Args:
        text: Markdown text.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        List of Chunk objects, mỗi chunk = 1 section (header + content).
    """
    metadata = metadata or {}

    # 1. Split by markdown headers (H1, H2, H3)
    sections = re.split(r'(^#{1,3}\s+.+$)', text, flags=re.MULTILINE)

    chunks = []
    current_header = ""
    current_content = ""

    for part in sections:
        if re.match(r'^#{1,3}\s+', part):
            # Save previous section if it has content
            if current_content.strip():
                chunk_text = f"{current_header}\n{current_content}".strip() if current_header else current_content.strip()
                chunks.append(Chunk(
                    text=chunk_text,
                    metadata={**metadata, "section": current_header.strip(), "strategy": "structure"}
                ))
            current_header = part.strip()
            current_content = ""
        else:
            current_content += part

    # Don't forget last section
    if current_content.strip():
        chunk_text = f"{current_header}\n{current_content}".strip() if current_header else current_content.strip()
        chunks.append(Chunk(
            text=chunk_text,
            metadata={**metadata, "section": current_header.strip(), "strategy": "structure"}
        ))

    # If no headers found, fall back to basic chunking
    if not chunks:
        return chunk_basic(text, metadata=metadata)

    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.

    Returns:
        {"basic": {...}, "semantic": {...}, "hierarchical": {...}, "structure": {...}}
    """
    results = {}

    for strategy_name in ["basic", "semantic", "hierarchical", "structure"]:
        all_chunks = []

        for doc in documents:
            text = doc["text"]
            meta = doc.get("metadata", {})

            if strategy_name == "basic":
                chunks = chunk_basic(text, metadata=meta)
            elif strategy_name == "semantic":
                chunks = chunk_semantic(text, metadata=meta)
            elif strategy_name == "hierarchical":
                parents, children = chunk_hierarchical(text, metadata=meta)
                # For stats, combine both
                all_chunks.extend(parents)
                chunks = children
            elif strategy_name == "structure":
                chunks = chunk_structure_aware(text, metadata=meta)

            all_chunks.extend(chunks)

        if all_chunks:
            lengths = [len(c.text) for c in all_chunks]
            stats = {
                "num_chunks": len(all_chunks),
                "avg_length": int(sum(lengths) / len(lengths)),
                "min_length": min(lengths),
                "max_length": max(lengths),
            }
        else:
            stats = {"num_chunks": 0, "avg_length": 0, "min_length": 0, "max_length": 0}

        results[strategy_name] = stats

    # Print comparison table
    print(f"\n{'Strategy':<15} | {'Chunks':>6} | {'Avg Len':>7} | {'Min':>5} | {'Max':>5}")
    print("-" * 50)
    for name, s in results.items():
        print(f"{name:<15} | {s['num_chunks']:>6} | {s['avg_length']:>7} | {s['min_length']:>5} | {s['max_length']:>5}")

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
