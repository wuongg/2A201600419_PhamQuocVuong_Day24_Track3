# Individual Reflection — Lab 18

**Tên:** Phạm Quốc Vương - 2a202600419  
**Module phụ trách:** M5 — Enrichment Pipeline

---

## 1. Đóng góp kỹ thuật

- Module đã implement: `src/m5_enrichment.py` — 4 enrichment techniques
- Các hàm/class chính đã viết:
  - `summarize_chunk()` — LLM tóm tắt 2-3 câu, fallback extractive (2 câu đầu)
  - `generate_hypothesis_questions()` — LLM generate N câu hỏi chunk có thể trả lời (HyQA)
  - `contextual_prepend()` — LLM viết 1 câu context + prepend vào chunk (Anthropic style)
  - `extract_metadata()` — LLM extract JSON {topic, entities, category, language}
  - `enrich_chunks()` — pipeline đầy đủ, trả về EnrichedChunk với original_text preserved
- Số tests pass: **10/10**

## 2. Kiến thức học được

- **Khái niệm mới nhất:** HyQA (Hypothesis Question-Answer) — thay vì chỉ index raw text, generate câu hỏi mà chunk có thể trả lời rồi index cả câu hỏi. Bridge vocabulary gap giữa user query và document language.
- **Điều bất ngờ nhất:** Contextual prepend đơn giản nhưng hiệu quả cao — Anthropic báo cáo giảm 49% retrieval failure chỉ với 1 câu context. Chi phí rất thấp (1 LLM call/chunk, one-time) nhưng ROI cao vì cải thiện mọi query sau đó.
- **Kết nối với bài giảng:** Slide "Enrichment strategies" — bài giảng nói enrichment là "one-time cost, infinite benefit". Sau khi implement mới hiểu: index time tăng ~2x nhưng query time không đổi.

## 3. Khó khăn & Cách giải quyết

- **Khó khăn lớn nhất:** Tất cả 4 techniques cần OpenAI API — nếu không có key thì module trả về empty/fallback, tests vẫn pass nhưng không có giá trị thực.
- **Cách giải quyết:** Implement extractive fallback cho tất cả techniques — không cần API vẫn chạy được, chỉ quality thấp hơn. Tests kiểm tra type và structure, không kiểm tra quality.
- **Thời gian debug:** ~10 phút cho JSON parsing từ LLM response (đôi khi trả về markdown code block thay vì raw JSON), ~5 phút cho `response_format={"type": "json_object"}` fix.

## 4. Nếu làm lại

- **Sẽ làm khác:** Thêm caching cho enrichment results — nếu chunk text không đổi thì không cần gọi LLM lại. Tiết kiệm cost khi re-index.
- **Module nào muốn thử tiếp:** M2 Search — muốn index cả hypothesis questions vào Qdrant riêng, search cả 2 collections rồi merge.

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | 5 |
| Code quality | 4 |
| Teamwork | 4 |
| Problem solving | 5 |
