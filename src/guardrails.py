"""
Guardrails: Presidio PII, topic validator, Llama Guard 3–compatible output moderation.

Đo P95 latency qua run_guardrails_benchmark.py (không import side-effects nặng ở import-time).
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field

# --- Topic scope (Sổ tay nhân viên / IT / HR trong corpus lab) ---
_HANDBOOK_KEYWORDS = (
    "nhân viên",
    "phép năm",
    "phép",
    "quy trình",
    "đơn xin",
    "ứng viên",
    "chính sách",
    "nghỉ phép",
    "nghỉ ốm",
    "thử việc",
    "tuyển dụng",
    "phỏng vấn",
    "đào tạo",
    "mentoring",
    "phúc lợi",
    "lương",
    "bảo hiểm",
    "vpn",
    "mật khẩu",
    "2fa",
    "xác thực",
    "it ",
    " it",
    "email",
    "phần mềm",
    "thiết bị",
    "mã hóa",
    "bitlocker",
    "filevault",
    "onedrive",
    "sharepoint",
    "hr.company",
    "hr ",
)


@dataclass
class GuardrailResult:
    ok: bool
    blocked_reason: str | None = None
    pii_spans: list[dict] = field(default_factory=list)
    topic_ok: bool = True
    output_safe: bool = True
    moderation_categories: list[str] = field(default_factory=list)
    backend_notes: str = ""


def _regex_pii_scan(text: str) -> list[dict]:
    """Fallback nhẹ khi Presidio không cài (email/số điện thoại Việt đơn giản)."""
    spans = []
    for m in re.finditer(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text):
        spans.append({"type": "EMAIL", "start": m.start(), "end": m.end(), "score": 0.9})
    for m in re.finditer(r"(?:\+84|0)(?:\d[\s.-]?){9,10}", text):
        spans.append({"type": "PHONE", "start": m.start(), "end": m.end(), "score": 0.85})
    return spans


def detect_pii_presidio(text: str) -> tuple[list[dict], str]:
    """
    Microsoft Presidio nếu có; ngược lại regex fallback.
    Trả về (spans, backend_name).
    """
    try:
        from presidio_analyzer import AnalyzerEngine

        engine = AnalyzerEngine()
        results = engine.analyze(text=text, language="en")
        spans = [
            {"type": r.entity_type, "start": r.start, "end": r.end, "score": float(r.score)}
            for r in results
        ]
        return spans, "presidio"
    except Exception as e:
        spans = _regex_pii_scan(text)
        return spans, f"regex_fallback({type(e).__name__})"


def validate_topic_on_handbook(text: str) -> tuple[bool, str]:
    """Topic validator: chỉ cho phép truy vấn liên quan nội dung sổ tay (keyword heuristic)."""
    low = text.lower()
    if any(k.strip() in low for k in _HANDBOOK_KEYWORDS):
        return True, "on_topic"
    if len(text.strip()) < 6:
        return False, "too_short_or_unclear"
    return False, "off_topic_not_handbook"


def moderate_output_llama_guard_compatible(text: str) -> tuple[bool, list[str], str]:
    """
    Kiểm duyệt đầu ra — tương thích Llama Guard 3 (meta categories).

    Thứ tự ưu tiên:
    1) LLAMA_GUARD_BACKEND=hf — thử pipeline HF nếu user cấu hình (model gated).
    2) LLAMA_GUARD_BACKEND=openai_moderation hoặc mặc định — OpenAI Moderations API (proxy vận hành).
    3) none — luôn safe (chỉ dùng khi benchmark infra).
    """
    backend = os.getenv("LLAMA_GUARD_BACKEND", "openai_moderation").lower()

    if backend == "none":
        return True, [], "backend=none_skip"

    if backend == "hf":
        try:
            return _llama_guard_hf_stub(text)
        except Exception as e:
            return False, ["guard_error"], f"hf_failed:{type(e).__name__}"

    # openai_moderation (default)
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        return True, [], "openai_moderation_skipped_no_api_key"

    try:
        from openai import OpenAI

        client = OpenAI(api_key=key)
        model = os.getenv("OPENAI_MODERATION_MODEL", "omni-moderation-latest")
        resp = client.moderations.create(model=model, input=text)
        cat = getattr(resp.results[0], "categories", None)
        flagged = []
        if cat:
            for name, val in cat.model_dump().items():
                if val:
                    flagged.append(name)
        safe = not flagged
        return safe, flagged, f"openai_moderation:{model}"
    except Exception as e:
        return False, ["moderation_error"], f"openai_failed:{type(e).__name__}"


def _llama_guard_hf_stub(text: str) -> tuple[bool, list[str], str]:
    """
    Stub HF — Llama Guard 3 là gated model; sinh viên có HF token có thể thay bằng inference thật.
    Hiện trả unsafe nếu có pattern jailbreak rõ ràng để demo logic pipeline.
    """
    banned = ("ignore previous", "system prompt", "jailbreak", "hack into")
    low = text.lower()
    if any(b in low for b in banned):
        return False, ["S14"], "hf_stub_keywords"
    return True, [], "hf_stub_pass"


def run_input_guards(user_query: str) -> GuardrailResult:
    """PII + topic trên đầu vào người dùng."""
    spans, pii_be = detect_pii_presidio(user_query)
    topic_ok, topic_reason = validate_topic_on_handbook(user_query)
    blocked = bool(spans) or not topic_ok
    reason = None
    if spans:
        reason = f"pii_detected:{pii_be}"
    elif not topic_ok:
        reason = f"topic:{topic_reason}"
    return GuardrailResult(
        ok=not blocked,
        blocked_reason=reason,
        pii_spans=spans,
        topic_ok=topic_ok,
        backend_notes=f"pii={pii_be}",
    )


def run_output_guards(llm_answer: str) -> GuardrailResult:
    """Moderation đầu ra (Llama Guard 3–compatible / OpenAI moderation)."""
    safe, cats, note = moderate_output_llama_guard_compatible(llm_answer)
    return GuardrailResult(ok=safe, output_safe=safe, moderation_categories=cats, backend_notes=note)


def run_full_guard_chain(user_query: str, llm_answer: str) -> GuardrailResult:
    inp = run_input_guards(user_query)
    if not inp.ok:
        return inp
    out = run_output_guards(llm_answer)
    if not out.ok:
        return GuardrailResult(
            ok=False,
            blocked_reason="unsafe_output",
            pii_spans=inp.pii_spans,
            topic_ok=inp.topic_ok,
            output_safe=out.output_safe,
            moderation_categories=out.moderation_categories,
            backend_notes=f"input={inp.backend_notes};output={out.backend_notes}",
        )
    return GuardrailResult(
        ok=True,
        pii_spans=inp.pii_spans,
        topic_ok=True,
        output_safe=True,
        backend_notes=f"input={inp.backend_notes};output={out.backend_notes}",
    )


def percentile_95(values: list[float]) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    k = (len(xs) - 1) * 0.95
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def benchmark_guard_latencies(n_iterations: int = 30) -> dict:
    """Đo thời gian chạy full guard chain trên câu mẫu (input + output cố định)."""
    sample_q = "Nhân viên được nghỉ phép năm bao nhiêu ngày?"
    sample_a = "12 ngày làm việc mỗi năm cho nhân viên chính thức."
    times: list[float] = []
    for _ in range(n_iterations):
        t0 = time.perf_counter()
        run_full_guard_chain(sample_q, sample_a)
        times.append((time.perf_counter() - t0) * 1000.0)
    return {
        "iterations": n_iterations,
        "p95_ms": round(percentile_95(times), 3),
        "median_ms": round(sorted(times)[len(times) // 2], 3),
        "max_ms": round(max(times), 3),
    }


def load_adversarial_prompts(path: str | None = None) -> list[dict]:
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = path or os.path.join(base, "data", "adversarial_prompts.json")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def evaluate_adversarial_prompts() -> dict:
    """Chạy 20 prompt đối kháng qua input guards; ghi nhận block rate."""
    prompts = load_adversarial_prompts()
    rows = []
    blocked = 0
    for item in prompts:
        text = item.get("prompt", "")
        g = run_input_guards(text)
        rows.append(
            {
                "id": item.get("id"),
                "category": item.get("category"),
                "blocked": not g.ok,
                "reason": g.blocked_reason,
            }
        )
        if not g.ok:
            blocked += 1
    return {
        "total": len(prompts),
        "blocked": blocked,
        "allow_rate": round((len(prompts) - blocked) / len(prompts), 4) if prompts else 0.0,
        "details": rows,
    }
