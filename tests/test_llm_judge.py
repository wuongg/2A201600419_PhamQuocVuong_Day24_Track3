"""Tests for LLM judge utilities."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.llm_judge import cohen_kappa, pairwise_swap_average


def test_cohen_kappa_perfect():
    a = ["A", "B", "A", "tie"]
    b = ["A", "B", "A", "tie"]
    assert cohen_kappa(a, b) == 1.0


def test_pairwise_swap_offline_deterministic(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    w, meta = pairwise_swap_average(
        "WireGuard VPN và AES-256 được quy định như thế nào trong tài liệu?",
        "VPN dùng WireGuard với mã hóa AES-256.",
        "Lorem ipsum dolor sit amet consectetur.",
        client=None,
    )
    assert w == "A"
    assert meta["votes"]["A"] >= meta["votes"]["B"]
