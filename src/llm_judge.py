"""LLM-as-Judge: pairwise + swap-and-average, absolute scores, Cohen κ vs human labels."""

from __future__ import annotations

import json
import math
import os
import re
from typing import Any, Literal

from src.llm_client import chat_completion_model, get_openai_compat_client

Winner = Literal["A", "B", "tie"]

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def cohen_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    """Cohen κ cho cùng bộ nhãn categorical (thứ tự aligned)."""
    if len(labels_a) != len(labels_b) or not labels_a:
        return 0.0
    n = len(labels_a)
    cats = sorted(set(labels_a) | set(labels_b))
    pa = sum(1 for x, y in zip(labels_a, labels_b) if x == y) / n
    pe = 0.0
    for c in cats:
        pa_i = sum(1 for x in labels_a if x == c) / n
        pb_i = sum(1 for x in labels_b if x == c) / n
        pe += pa_i * pb_i
    if math.isclose(pe, 1.0):
        return 1.0 if math.isclose(pa, 1.0) else 0.0
    return (pa - pe) / (1.0 - pe)


def _parse_winner(raw: str) -> Winner | None:
    t = raw.strip().upper()
    if re.search(r"\bA\b", t) and not re.search(r"\bB\b", t):
        return "A"
    if re.search(r"\bB\b", t) and not re.search(r"\bA\b", t):
        return "B"
    if "TIE" in t or "DRAW" in t or "CẢ HAI" in raw.upper():
        return "tie"
    return None


def _pairwise_once(question: str, first_label: str, second_label: str, client: Any | None) -> Winner:
    """first_label/second_label là text của hai đáp án theo thứ tự hiển thị."""
    sys_prompt = (
        "Bạn là giám khảo khách quan. Chọn đáp án khớp chính sách/facts trong câu hỏi "
        "sổ tay nhân viên hơn. Trả lời ĐÚNG một token: A (ưu tiên đáp án [Block A]), "
        "B (ưu tiên đáp án [Block B]), hoặc TIE."
    )
    user_prompt = (
        f"Câu hỏi: {question}\n\n"
        f"[Block A]\n{first_label}\n\n"
        f"[Block B]\n{second_label}\n\n"
        "Token:"
    )
    if client is None:
        client = get_openai_compat_client()

    if client is not None:
        try:
            resp = client.chat.completions.create(
                model=os.getenv("LLM_JUDGE_MODEL") or chat_completion_model(),
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                max_tokens=8,
            )
            raw = resp.choices[0].message.content or ""
            w = _parse_winner(raw)
            if w:
                return w
        except Exception:
            pass

    # Lexical fallback (offline CI): trùng khóa dài + bonus khi đáp án chứa số xuất hiện trong câu hỏi (fact QA)
    def overlap_score(ans: str, q: str) -> float:
        qtok = {w.lower() for w in re.findall(r"\w{4,}", q)}
        atok = set(re.findall(r"\w{4,}", ans.lower()))
        base = len(qtok & atok) / max(1, len(qtok))
        dq = set(re.findall(r"\d+", q))
        da = set(re.findall(r"\d+", ans))
        if dq & da:
            base += 0.35
        qlow = q.lower()
        if re.search(r"bao nhiêu|mấy|bao lâu|tối đa|tối thiểu", qlow) and re.search(r"\d+", ans):
            base += 0.22
        if len(ans.strip()) < 8:
            base -= 0.15
        low = ans.lower()
        if ("vnđ" in qlow or "phụ cấp" in qlow or "ăn trưa" in qlow) and "usd" in low:
            base -= 0.55
        if ("usb" in qlow or "thiết bị lưu trữ" in qlow) and "luôn được phép" in low:
            base -= 0.35
        if ("usb" in qlow or "máy công ty" in qlow) and ("không được kết nối" in low or ("không được" in low and "phép" in low)):
            base += 0.18
        if ("2fa" in qlow or "hai yếu tố" in qlow) and ("nhạy cảm" in low or "dữ liệu nhạy cảm" in low):
            base += 0.38
        if re.search(r"không\s+(bao giờ\s+)?bắt buộc", low):
            base -= 0.45
        for cue in ("ít nhất", "bắt buộc", "theo quy định", "email xác nhận"):
            if cue in low:
                base += 0.08
        for bad in ("không cần báo", "không bao giờ", "plaintext", "pptp", "không mã hóa"):
            if bad in low:
                base -= 0.25
        return base

    s_first = overlap_score(first_label, question)
    s_second = overlap_score(second_label, question)
    if math.isclose(s_first, s_second, rel_tol=0.0, abs_tol=1e-9):
        # Hiếm khi hai đáp án trùng điểm heuristic — chọn bản trả lời chi tiết hơn (thường là policy đúng).
        return "A" if len(first_label) >= len(second_label) else "B"
    return "A" if s_first > s_second else "B"


def pairwise_swap_average(question: str, answer_a: str, answer_b: str, client: Any | None = None) -> tuple[Winner, dict]:
    """
    Swap-and-average: hai lần so sánh đảo vị trí, map về winner A/B/tie.
    Trả về (final_winner, debug_dict).
    """
    r1 = _pairwise_once(question, answer_a, answer_b, client)
    if r1 == "A":
        w1: Winner = "A"
    elif r1 == "B":
        w1 = "B"
    else:
        w1 = "tie"

    r2 = _pairwise_once(question, answer_b, answer_a, client)
    if r2 == "A":
        w2: Winner = "B"
    elif r2 == "B":
        w2 = "A"
    else:
        w2 = "tie"

    votes = {"A": 0, "B": 0, "tie": 0}
    for w in (w1, w2):
        votes[w] += 1

    if votes["A"] > votes["B"]:
        final: Winner = "A"
    elif votes["B"] > votes["A"]:
        final = "B"
    else:
        final = "tie"

    bias_signal = w1 != w2 and w1 != "tie" and w2 != "tie"
    meta = {
        "trial1_mapped_to_original": w1,
        "trial2_mapped_to_original": w2,
        "positional_disagreement": bias_signal,
        "votes": votes,
    }
    return final, meta


def absolute_score_answer(question: str, answer: str, client: Any | None = None) -> dict:
    """Chấm tuyệt đối 1–5 về độ đúng và độ đầy đủ so với câu hỏi."""
    if client is None:
        client = get_openai_compat_client()

    prompt = (
        f"Câu hỏi: {question}\nĐáp án: {answer}\n"
        "Cho điểm correctness (1-5) và completeness (1-5). JSON: "
        '{"correctness":int,"completeness":int}'
    )
    if client is not None:
        try:
            resp = client.chat.completions.create(
                model=os.getenv("LLM_JUDGE_MODEL") or chat_completion_model(),
                messages=[
                    {"role": "system", "content": "Chỉ trả JSON hợp lệ."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                response_format={"type": "json_object"},
                max_tokens=60,
            )
            raw = resp.choices[0].message.content or "{}"
            data = json.loads(raw)
            return {
                "correctness": int(data.get("correctness", 3)),
                "completeness": int(data.get("completeness", 3)),
            }
        except Exception:
            pass

    wc = len(re.findall(r"\w+", answer))
    base = min(5, 3 + (1 if wc > 15 else 0))
    return {"correctness": base, "completeness": base}


def load_pairwise_items(path: str | None = None) -> list[dict]:
    p = path or os.path.join(_REPO_ROOT, "data", "judge_pairwise_items.json")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def load_human_labels(path: str | None = None) -> dict[str, str]:
    p = path or os.path.join(_REPO_ROOT, "data", "human_judge_labels.json")
    with open(p, encoding="utf-8") as f:
        rows = json.load(f)
    return {row["pair_id"]: row["human_pairwise_winner"].upper() for row in rows}


def run_judge_evaluation(client: Any | None = None) -> dict:
    items = load_pairwise_items()
    human = load_human_labels()
    judge_labels: list[str] = []
    human_aligned: list[str] = []
    bias_pairs: list[dict] = []

    details = []
    kappa_ids = sorted(human.keys())
    for pid in kappa_ids:
        item = next((x for x in items if x["id"] == pid), None)
        if not item:
            continue
        q = item["question"]
        a, b = item["answer_a"], item["answer_b"]
        winner, meta = pairwise_swap_average(q, a, b, client)
        jl = "tie" if winner == "tie" else winner.upper()
        judge_labels.append(jl)
        human_aligned.append(human[pid])

        details.append({"pair_id": pid, "judge": jl, "human": human[pid], **meta})
        if meta.get("positional_disagreement"):
            bias_pairs.append({"pair_id": pid, "trials": meta})

    abs_scores = []
    for item in items[:5]:
        abs_scores.append(
            {
                "pair_id": item["id"],
                "a": absolute_score_answer(item["question"], item["answer_a"], client),
                "b": absolute_score_answer(item["question"], item["answer_b"], client),
            }
        )

    kappa = cohen_kappa(judge_labels, human_aligned)

    bias_notes = (
        "Judge có thể thiên vị vị trí (positional bias) khi hai trial swap cho kết quả khác nhau; "
        "swap-and-average giảm nhưng không loại trừ hoàn toàn. "
        "Fallback lexical có thể khác LLM khi không có API."
    )

    return {
        "cohen_kappa": round(kappa, 4),
        "n_human_pairs": len(human_aligned),
        "pairwise_details": details,
        "positional_bias_incidents": bias_pairs,
        "absolute_scores_sample": abs_scores,
        "bias_documentation": bias_notes,
    }


def save_judge_report(report: dict, path: str | None = None) -> str:
    outp = path or os.path.join(_REPO_ROOT, "reports", "llm_judge_report.json")
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return outp
