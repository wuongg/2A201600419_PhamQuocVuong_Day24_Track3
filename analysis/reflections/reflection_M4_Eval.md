# Individual Reflection — Lab 18

**Tên:** Phạm Quốc Vương - 2a202600419  
**Module phụ trách:** M4 — RAGAS Evaluation

---

## 1. Đóng góp kỹ thuật

- Module đã implement: `src/m4_eval.py` — RAGAS evaluation + failure analysis
- Các hàm/class chính đã viết:
  - `evaluate_ragas()` — Dataset.from_dict → ragas.evaluate → extract per-question scores
  - `failure_analysis()` — sort bottom-N, map worst metric → diagnosis + suggested_fix
  - `save_report()` — JSON output với aggregate + failures
- Số tests pass: **4/4**

## 2. Kiến thức học được

- **Khái niệm mới nhất:** 4 RAGAS metrics và ý nghĩa thực tế:
  - **Faithfulness**: LLM có bịa không? (grounding)
  - **Answer Relevancy**: Câu trả lời có đúng câu hỏi không? (focus)
  - **Context Precision**: Chunks retrieved có liên quan không? (precision)
  - **Context Recall**: Chunks retrieved có đủ thông tin không? (recall)
- **Điều bất ngờ nhất:** Faithfulness và Answer Relevancy đo 2 thứ khác nhau — có thể faithfulness cao (không bịa) nhưng answer relevancy thấp (trả lời đúng context nhưng không đúng câu hỏi).
- **Kết nối với bài giảng:** Slide "Diagnostic Tree" — sau khi implement failure_analysis mới thấy tree này thực sự hữu ích: biết metric nào thấp → biết fix ở đâu (chunking/search/reranking/prompt).

## 3. Khó khăn & Cách giải quyết

- **Khó khăn lớn nhất:** RAGAS API thay đổi giữa các version — `ragas.metrics` deprecated, cần dùng `ragas.metrics.collections`. Cũng cần `OPENAI_API_KEY` để chạy LLM-based metrics.
- **Cách giải quyết:** Wrap trong try/except, fallback về dummy scores 0.0 nếu RAGAS không available. Thêm deprecation warning handling.
- **Thời gian debug:** ~15 phút cho RAGAS API compatibility, ~10 phút cho per_question extraction từ DataFrame.

## 4. Nếu làm lại

- **Sẽ làm khác:** Thêm custom metrics phù hợp với tiếng Việt — RAGAS dùng LLM để judge, nếu LLM không hiểu tiếng Việt tốt thì scores không accurate.
- **Module nào muốn thử tiếp:** M5 Enrichment — muốn đo xem contextual prepend cải thiện context_recall bao nhiêu % so với raw chunks.

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | 5 |
| Code quality | 4 |
| Teamwork | 5 |
| Problem solving | 4 |
