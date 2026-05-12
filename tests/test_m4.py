"""Tests for Module 4: Evaluation."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.m4_eval import (
    load_test_set,
    evaluate_ragas,
    failure_analysis,
    failure_cluster_analysis,
    EvalResult,
)

def test_load_test_set():
    ts = load_test_set(max_questions=10**6)
    assert len(ts) >= 50 and "question" in ts[0] and "ground_truth" in ts[0]
    assert ts[0].get("distribution") in ("direct_factual", "procedural_synthesis", "edge_paraphrase")

def test_evaluate_returns_metrics():
    r = evaluate_ragas(["q"], ["a"], [["c"]], ["gt"])
    for k in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        assert k in r and isinstance(r[k], (int, float))

def test_failure_analysis_returns():
    results = [EvalResult("Q1", "A1", ["C1"], "GT1", 0.5, 0.6, 0.4, 0.3)]
    f = failure_analysis(results, bottom_n=1)
    assert len(f) == 1

def test_failure_has_diagnosis():
    results = [EvalResult("Q1", "A1", ["C1"], "GT1", 0.5, 0.6, 0.4, 0.3)]
    f = failure_analysis(results, bottom_n=1)
    if f:
        assert "diagnosis" in f[0] and "suggested_fix" in f[0]

def test_failure_cluster_analysis():
    f = [{"worst_metric": "faithfulness", "question": "q1", "diagnosis": "d1"}]
    c = failure_cluster_analysis(f)
    assert "by_worst_metric" in c and "faithfulness" in c["by_worst_metric"]
