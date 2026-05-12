"""
Client OpenAI-compatible: API cloud hoặc LLM cục bộ (Ollama, LM Studio, vLLM).

.env ví dụ Ollama:
  LOCAL_LLM_BASE_URL=http://127.0.0.1:11434/v1
  LOCAL_LLM_MODEL=qwen2.5:latest
  LOCAL_LLM_API_KEY=ollama

Embedding cho RAGAS (answer_relevancy) — cùng host hoặc URL riêng:
  LOCAL_EMBEDDING_MODEL=nomic-embed-text
  # LOCAL_EMBEDDING_BASE_URL=http://127.0.0.1:11434/v1   # mặc định = LOCAL_LLM_BASE_URL
"""

from __future__ import annotations

import os


def use_local_llm() -> bool:
    return bool(os.getenv("LOCAL_LLM_BASE_URL", "").strip())


def get_openai_compat_client():
    """
    Cloud: cần OPENAI_API_KEY.
    Local: đặt LOCAL_LLM_BASE_URL (vd. Ollama .../v1), LOCAL_LLM_API_KEY tuỳ chọn (mặc định ollama).
    """
    try:
        from openai import OpenAI
    except ImportError:
        return None

    base = os.getenv("LOCAL_LLM_BASE_URL", "").strip().rstrip("/")
    if base:
        key = os.getenv("LOCAL_LLM_API_KEY", "ollama")
        return OpenAI(base_url=base, api_key=key)

    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        return None
    return OpenAI(api_key=key)


def chat_completion_model() -> str:
    if use_local_llm():
        return os.getenv("LOCAL_LLM_MODEL", "llama3.2").strip()
    return os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini").strip()


def embedding_langchain_kwargs() -> dict:
    """Tham số cho langchain_openai.OpenAIEmbeddings (RAGAS answer_relevancy)."""
    emb_base = os.getenv("LOCAL_EMBEDDING_BASE_URL", "").strip().rstrip("/")
    llm_base = os.getenv("LOCAL_LLM_BASE_URL", "").strip().rstrip("/")
    base = emb_base or llm_base
    if base:
        return {
            "model": os.getenv("LOCAL_EMBEDDING_MODEL", "nomic-embed-text").strip(),
            "openai_api_key": os.getenv("LOCAL_LLM_API_KEY", "ollama"),
            "openai_api_base": base,
        }
    return {
        "model": os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small").strip(),
        "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
    }


def ragas_chat_llm():
    """LangChain ChatOpenAI cho ragas.evaluate(llm=...) — cloud hoặc local."""
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        return None

    if use_local_llm():
        base = os.getenv("LOCAL_LLM_BASE_URL", "").strip().rstrip("/")
        return ChatOpenAI(
            model=chat_completion_model(),
            openai_api_base=base,
            openai_api_key=os.getenv("LOCAL_LLM_API_KEY", "ollama"),
            temperature=0,
        )
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        return None
    return ChatOpenAI(
        model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        openai_api_key=key,
        temperature=0,
    )


def ragas_embeddings_lc():
    """Embeddings LangChain cho ragas.evaluate(embeddings=...)."""
    try:
        from langchain_openai import OpenAIEmbeddings
    except ImportError:
        return None

    kw = embedding_langchain_kwargs()
    if not kw.get("openai_api_key") and not use_local_llm():
        return None
    return OpenAIEmbeddings(**kw)
