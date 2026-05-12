"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH, LAB_EVAL_LIMIT
from src.llm_client import ragas_chat_llm, ragas_embeddings_lc, use_local_llm


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    distribution: str | None = None


def load_test_set(path: str = TEST_SET_PATH, *, max_questions: int | None = None) -> list[dict]:
    """Load test set từ JSON. max_questions=None → áp dụng LAB_EVAL_LIMIT (env); >0 cắt bớt."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    cap = LAB_EVAL_LIMIT if max_questions is None else max_questions
    if cap and cap > 0:
        data = data[:cap]
    return data


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """
    Run RAGAS evaluation on 4 metrics:
    - Faithfulness: LLM answer grounded in context?
    - Answer Relevancy: Answer relevant to question?
    - Context Precision: Retrieved context relevant?
    - Context Recall: Context covers ground truth?
    """
    try:
        from ragas import evaluate
        from datasets import Dataset
        from ragas.metrics._faithfulness import faithfulness
        from ragas.metrics._context_precision import context_precision
        from ragas.metrics._context_recall import context_recall
        from ragas.metrics._answer_relevance import answer_relevancy

        llm = ragas_chat_llm()
        emb = ragas_embeddings_lc()

        metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

        # Fix answer_relevancy embeddings (RAGAS 0.4 compat)
        if emb is not None:
            try:
                answer_relevancy.embeddings = emb
            except Exception:
                pass

        # Build dataset
        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })

        eval_kw = {}
        if llm is not None:
            eval_kw["llm"] = llm
        if emb is not None:
            eval_kw["embeddings"] = emb

        # LM Studio / LLM cục bộ thường chỉ phục vụ tốt khi ít job song song (tránh "No models loaded").
        if use_local_llm():
            from ragas.run_config import RunConfig

            workers = int(os.getenv("RAGAS_MAX_WORKERS", "1"))
            tout = int(os.getenv("RAGAS_TIMEOUT", "420"))
            eval_kw["run_config"] = RunConfig(max_workers=max(1, workers), timeout=tout)

        # Run evaluation
        result = evaluate(dataset, metrics=metrics, **eval_kw)

        # Extract per-question results
        df = result.to_pandas()

        def _safe_float(val):
            if val is None:
                return 0.0
            if isinstance(val, list):
                return float(val[0]) if val else 0.0
            try:
                v = float(val)
                return 0.0 if v != v else v  # handle NaN
            except (TypeError, ValueError):
                return 0.0

        per_question = []
        for i, (_, row) in enumerate(df.iterrows()):
            q_cell = row.get("question")
            if q_cell is None or (isinstance(q_cell, str) and not str(q_cell).strip()):
                q_cell = row.get("user_input", "")
            q_str = str(q_cell).strip() if q_cell is not None and str(q_cell).strip() != "nan" else ""
            if not q_str and i < len(questions):
                q_str = questions[i]

            gt_cell = row.get("ground_truth", row.get("ground_truths", ""))
            gt_str = str(gt_cell) if gt_cell is not None else ""

            per_question.append(EvalResult(
                question=q_str,
                answer=str(row.get("answer", "")),
                contexts=list(row.get("contexts", [])),
                ground_truth=gt_str,
                faithfulness=_safe_float(row.get("faithfulness")),
                answer_relevancy=_safe_float(row.get("answer_relevancy")),
                context_precision=_safe_float(row.get("context_precision")),
                context_recall=_safe_float(row.get("context_recall")),
                distribution=None,
            ))

        def _agg(key):
            try:
                v = float(result[key])
                return 0.0 if v != v else v  # handle NaN
            except Exception:
                vals = [getattr(r, key, 0.0) for r in per_question]
                valid = [v for v in vals if v is not None and v == v]
                return sum(valid) / len(valid) if valid else 0.0

        return {
            "faithfulness": _agg("faithfulness"),
            "answer_relevancy": _agg("answer_relevancy"),
            "context_precision": _agg("context_precision"),
            "context_recall": _agg("context_recall"),
            "per_question": per_question,
        }

    except ImportError as e:
        print(f"  [WARN] RAGAS not available: {e}")
    except Exception as e:
        print(f"  [WARN] RAGAS evaluation error: {e}")

    # Fallback
    per_question = [
        EvalResult(question=q, answer=a, contexts=c, ground_truth=gt,
                   faithfulness=0.0, answer_relevancy=0.0,
                   context_precision=0.0, context_recall=0.0,
                   distribution=None)
        for q, a, c, gt in zip(questions, answers, contexts, ground_truths)
    ]
    return {"faithfulness": 0.0, "answer_relevancy": 0.0,
            "context_precision": 0.0, "context_recall": 0.0, "per_question": per_question}


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """
    Analyze bottom-N worst questions using Diagnostic Tree.

    Diagnostic mapping:
      faithfulness < 0.85     → "LLM hallucinating"       → "Tighten prompt, lower temperature"
      context_recall < 0.75   → "Missing relevant chunks"  → "Improve chunking or add BM25"
      context_precision < 0.75 → "Too many irrelevant chunks" → "Add reranking or metadata filter"
      answer_relevancy < 0.80 → "Answer doesn't match question" → "Improve prompt template"
    """
    if not eval_results:
        return []

    # 1. Compute avg score for each result
    def avg_score(r: EvalResult) -> float:
        scores = [r.faithfulness, r.answer_relevancy, r.context_precision, r.context_recall]
        valid = [s for s in scores if s is not None]
        return sum(valid) / len(valid) if valid else 0.0

    # 2. Sort by avg_score ascending → worst first
    sorted_results = sorted(eval_results, key=avg_score)
    bottom = sorted_results[:bottom_n]

    # 3. Diagnostic mapping
    DIAGNOSTICS = {
        "faithfulness": {
            "threshold": 0.85,
            "diagnosis": "LLM hallucinating — answer not grounded in context",
            "suggested_fix": "Tighten prompt with explicit grounding instruction, lower temperature"
        },
        "context_recall": {
            "threshold": 0.75,
            "diagnosis": "Missing relevant chunks — retrieval not finding key information",
            "suggested_fix": "Improve chunking strategy or add BM25 hybrid search"
        },
        "context_precision": {
            "threshold": 0.75,
            "diagnosis": "Too many irrelevant chunks — retrieval returning noise",
            "suggested_fix": "Add reranking or metadata filtering to improve precision"
        },
        "answer_relevancy": {
            "threshold": 0.80,
            "diagnosis": "Answer doesn't match question — response off-topic",
            "suggested_fix": "Improve prompt template to focus on answering the specific question"
        },
    }

    failures = []
    for result in bottom:
        scores = {
            "faithfulness": result.faithfulness,
            "answer_relevancy": result.answer_relevancy,
            "context_precision": result.context_precision,
            "context_recall": result.context_recall,
        }

        # Find worst metric (furthest below threshold)
        worst_metric = min(scores, key=lambda m: scores[m])
        worst_score = scores[worst_metric]

        diag = DIAGNOSTICS.get(worst_metric, {
            "diagnosis": "Unknown issue",
            "suggested_fix": "Review pipeline end-to-end"
        })

        failures.append({
            "question": result.question,
            "worst_metric": worst_metric,
            "score": worst_score,
            "avg_score": avg_score(result),
            "all_scores": scores,
            "diagnosis": diag["diagnosis"],
            "suggested_fix": diag["suggested_fix"],
        })

    return failures


def attach_distributions(per_question: list[EvalResult], test_set: list[dict]) -> None:
    """Gắn nhãn distribution từ test_set (cùng thứ tự câu hỏi)."""
    for i, r in enumerate(per_question):
        if i < len(test_set):
            d = test_set[i].get("distribution")
            if d:
                r.distribution = str(d)


def distribution_breakdown(per_question: list[EvalResult]) -> dict:
    """Trung bình 4 metric theo từng distribution (test stratified)."""
    buckets: dict[str, list[EvalResult]] = {}
    for r in per_question:
        key = r.distribution or "unknown"
        buckets.setdefault(key, []).append(r)

    def _avg(rows: list[EvalResult], metric: str) -> float:
        vals = [getattr(x, metric) for x in rows]
        vals = [float(v) for v in vals if v is not None and v == v]
        return sum(vals) / len(vals) if vals else 0.0

    out = {}
    for name, rows in sorted(buckets.items()):
        out[name] = {
            "count": len(rows),
            "faithfulness": _avg(rows, "faithfulness"),
            "answer_relevancy": _avg(rows, "answer_relevancy"),
            "context_precision": _avg(rows, "context_precision"),
            "context_recall": _avg(rows, "context_recall"),
        }
    return out


def failure_cluster_analysis(failures: list[dict]) -> dict:
    """
    Gom nhóm failure patterns (cluster) theo worst_metric và nhóm diagnosis.
    Dùng cho rubric: failure cluster analysis.
    """
    by_metric: dict[str, list[str]] = {}
    by_diagnosis: dict[str, list[str]] = {}
    for f in failures:
        m = f.get("worst_metric", "unknown")
        by_metric.setdefault(m, []).append(f.get("question", "")[:120])
        diag = f.get("diagnosis", "unknown")[:80]
        by_diagnosis.setdefault(diag, []).append(f.get("question", "")[:120])

    clusters = {
        "by_worst_metric": {k: {"count": len(v), "sample_questions": v[:5]} for k, v in by_metric.items()},
        "by_diagnosis_pattern": {k: {"count": len(v), "sample_questions": v[:3]} for k, v in by_diagnosis.items()},
        "summary": (
            f"{len(failures)} low-performing questions; dominant failure axes: "
            + ", ".join(sorted(by_metric.keys(), key=lambda x: -len(by_metric[x]))[:4])
        ),
    }
    return clusters


def save_report(
    results: dict,
    failures: list[dict],
    path: str = "ragas_report.json",
    *,
    failure_clusters: dict | None = None,
    distribution_breakdown_dict: dict | None = None,
):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
        "failure_clusters": failure_clusters or {},
        "distribution_breakdown": distribution_breakdown_dict or {},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
