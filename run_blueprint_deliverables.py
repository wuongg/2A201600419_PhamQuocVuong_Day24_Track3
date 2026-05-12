"""
Chạy deliverable Blueprint (Guard + Judge + báo cáo latency).

Usage:
    python run_blueprint_deliverables.py

Output:
    reports/guardrails_report.json
    reports/llm_judge_report.json
"""

from __future__ import annotations

import json
import os

from src.guardrails import benchmark_guard_latencies, evaluate_adversarial_prompts
from src.llm_judge import run_judge_evaluation, save_judge_report


def main():
    os.makedirs("reports", exist_ok=True)

    adv = evaluate_adversarial_prompts()
    lat = benchmark_guard_latencies(int(os.getenv("GUARD_BENCHMARK_ITERS", "40")))
    guard_report = {
        "adversarial_input_evaluation": adv,
        "guard_latency_ms": lat,
        "notes": "Output moderation dùng LLAMA_GUARD_BACKEND (xem BLUEPRINT.md). P95 đo trên full guard chain.",
    }
    gp = os.path.join("reports", "guardrails_report.json")
    with open(gp, "w", encoding="utf-8") as f:
        json.dump(guard_report, f, ensure_ascii=False, indent=2)
    print(f"Wrote {gp}")

    jr = run_judge_evaluation()
    jp = save_judge_report(jr)
    print(f"Wrote {jp}")
    print(f"Cohen kappa (judge vs human, n={jr['n_human_pairs']}): {jr['cohen_kappa']}")


if __name__ == "__main__":
    main()
