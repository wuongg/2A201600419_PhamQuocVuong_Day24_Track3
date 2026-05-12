"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import os, sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.llm_client import get_openai_compat_client, chat_completion_model


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


def _get_openai_client():
    """OpenAI-compatible client (cloud hoặc LOCAL_LLM_BASE_URL)."""
    return get_openai_compat_client()


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.

    Args:
        text: Raw chunk text.

    Returns:
        Summary string (2-3 câu).
    """
    client = _get_openai_client()

    if client:
        # Option A: LLM summarization
        try:
            resp = client.chat.completions.create(
                model=chat_completion_model(),
                messages=[
                    {
                        "role": "system",
                        "content": "Tóm tắt đoạn văn sau trong 2-3 câu ngắn gọn bằng tiếng Việt. Giữ lại thông tin quan trọng nhất."
                    },
                    {"role": "user", "content": text},
                ],
                max_tokens=150,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            pass

    # Option B: Extractive fallback — lấy 2 câu đầu
    sentences = [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]
    if len(sentences) >= 2:
        return ". ".join(sentences[:2]) + "."
    return text[:200] if len(text) > 200 else text


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).

    Args:
        text: Raw chunk text.
        n_questions: Số câu hỏi cần generate.

    Returns:
        List of question strings.
    """
    client = _get_openai_client()

    if client:
        try:
            resp = client.chat.completions.create(
                model=chat_completion_model(),
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"Dựa trên đoạn văn, tạo {n_questions} câu hỏi mà đoạn văn có thể trả lời. "
                            "Trả về mỗi câu hỏi trên 1 dòng, không đánh số."
                        )
                    },
                    {"role": "user", "content": text},
                ],
                max_tokens=200,
            )
            raw = resp.choices[0].message.content.strip()
            questions = [
                q.strip().lstrip("0123456789.-) ")
                for q in raw.split("\n")
                if q.strip()
            ]
            return questions[:n_questions]
        except Exception:
            pass

    # Extractive fallback: generate simple questions from key phrases
    questions = []
    if "ngày" in text.lower():
        questions.append("Có bao nhiêu ngày được đề cập?")
    if "nhân viên" in text.lower():
        questions.append("Nhân viên có quyền lợi gì?")
    if "chính sách" in text.lower() or "quy định" in text.lower():
        questions.append("Chính sách/quy định này là gì?")
    # Generic fallback
    if not questions:
        questions = [f"Đoạn văn này nói về điều gì?"]
    return questions[:n_questions]


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).

    Args:
        text: Raw chunk text.
        document_title: Tên document gốc.

    Returns:
        Text với context prepended.
    """
    client = _get_openai_client()

    if client:
        try:
            doc_info = f"Tài liệu: {document_title}\n\n" if document_title else ""
            resp = client.chat.completions.create(
                model=chat_completion_model(),
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Viết 1 câu ngắn mô tả đoạn văn này nằm ở đâu trong tài liệu "
                            "và nói về chủ đề gì. Chỉ trả về 1 câu duy nhất."
                        )
                    },
                    {"role": "user", "content": f"{doc_info}Đoạn văn:\n{text}"},
                ],
                max_tokens=80,
            )
            context_sentence = resp.choices[0].message.content.strip()
            return f"{context_sentence}\n\n{text}"
        except Exception:
            pass

    # Extractive fallback: prepend document title as context
    if document_title:
        return f"Trích từ tài liệu: {document_title}\n\n{text}"
    return text


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.

    Args:
        text: Raw chunk text.

    Returns:
        Dict with extracted metadata fields.
    """
    client = _get_openai_client()

    if client:
        try:
            import json as _json
            resp = client.chat.completions.create(
                model=chat_completion_model(),
                messages=[
                    {
                        "role": "system",
                        "content": (
                            'Trích xuất metadata từ đoạn văn. '
                            'Trả về JSON hợp lệ: {"topic": "...", "entities": ["..."], '
                            '"category": "policy|hr|it|finance|other", "language": "vi|en"}'
                        )
                    },
                    {"role": "user", "content": text},
                ],
                max_tokens=150,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content.strip()
            return _json.loads(raw)
        except Exception:
            pass

    # Extractive fallback: simple keyword-based metadata
    text_lower = text.lower()
    category = "other"
    if any(w in text_lower for w in ["nghỉ phép", "lương", "thưởng", "nhân viên", "tuyển dụng"]):
        category = "hr"
    elif any(w in text_lower for w in ["mật khẩu", "vpn", "bảo mật", "phần mềm", "máy tính"]):
        category = "it"
    elif any(w in text_lower for w in ["tài chính", "ngân sách", "chi phí", "hóa đơn"]):
        category = "finance"
    elif any(w in text_lower for w in ["chính sách", "quy định", "quy trình", "hướng dẫn"]):
        category = "policy"

    return {
        "topic": text[:50].strip(),
        "entities": [],
        "category": category,
        "language": "vi"
    }


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks.

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: List of methods to apply. Default: ["contextual", "hyqa", "metadata"]
                 Options: "summary", "hyqa", "contextual", "metadata", "full"

    Returns:
        List of EnrichedChunk objects.
    """
    if methods is None:
        methods = ["contextual", "hyqa", "metadata"]

    enriched = []

    for chunk in chunks:
        text = chunk.get("text", "")
        meta = chunk.get("metadata", {})
        doc_title = meta.get("source", "")

        # Apply each technique based on methods list
        use_summary = "summary" in methods or "full" in methods
        use_hyqa = "hyqa" in methods or "full" in methods
        use_contextual = "contextual" in methods or "full" in methods
        use_metadata = "metadata" in methods or "full" in methods

        summary = summarize_chunk(text) if use_summary else ""
        questions = generate_hypothesis_questions(text) if use_hyqa else []
        enriched_text = contextual_prepend(text, doc_title) if use_contextual else text
        auto_meta = extract_metadata(text) if use_metadata else {}

        enriched.append(EnrichedChunk(
            original_text=text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={**meta, **auto_meta},
            method="+".join(methods),
        ))

    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")
