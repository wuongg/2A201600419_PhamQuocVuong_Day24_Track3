# CI/CD Integration Plan — RAGAS & Regression Eval

**Mục tiêu:** Tự động hóa đánh giá RAG sau mỗi thay đổi retrieval/generation/prompt, giữ ngưỡng chất lượng trước khi promote build.

## Pipeline đề xuất

1. **Trigger:** Pull request và nightly schedule trên nhánh `main`.
2. **Job `eval-regression`:**
   - Cài dependencies (`pip install -r requirements.txt`).
   - Inject secrets: `OPENAI_API_KEY` (GitHub Actions secrets / GitLab CI variables).
   - Chạy `pytest tests/ -q` (kiểm tra unit).
   - Chạy `python run_eval_quick.py` hoặc `python src/pipeline.py` để sinh `reports/ragas_report.json`.
3. **Quality gates (fail PR nếu vi phạm):**
   - `faithfulness ≥ 0.70` và `context_recall ≥ 0.70` (điều chỉnh theo SLO trong `BLUEPRINT.md`).
   - `num_questions == 54` (đồng bộ `test_set.json` stratified).
   - Không được làm trống `failure_clusters` khi có failures (artifact kiểm tra JSON schema).
4. **Artifacts:** upload `reports/ragas_report.json`, `reports/guardrails_report.json`, `reports/llm_judge_report.json` làm artifact 30 ngày.
5. **Cache:** cache HuggingFace models (`TRANSFORMERS_CACHE`) và pip để giảm thời gian job.

## Tách môi trường

- **PR:** eval trên subset deterministic (vd. 12 câu) để giữ chi phí API thấp — có thể thêm `test_set_ci.json`.
- **Release:** full `test_set.json` + snapshot embedding index checksum.

## Observability

- Gửi aggregate metrics tới Datadog/Grafana dưới dạng gauge (`ragas.faithfulness`, …).
- Alert khi Cohen κ judge vs human annotators giảm > 0.1 so với baseline tuần trước (drift chất lượng annotation hoặc judge prompt).

## Chi phí & giới hạn

- Giới hạn song song (max 2 concurrent eval jobs) để tránh spike token.
- Retry backoff khi rate-limit OpenAI.
