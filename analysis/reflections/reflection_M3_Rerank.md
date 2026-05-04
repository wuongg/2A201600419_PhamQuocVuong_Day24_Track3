# Individual Reflection — Lab 18

**Tên:** Phạm Quốc Vương - 2a202600419  
**Module phụ trách:** M3 — Reranking

---

## 1. Đóng góp kỹ thuật

- Module đã implement: `src/m3_rerank.py` — Cross-encoder reranking + latency benchmark
- Các hàm/class chính đã viết:
  - `CrossEncoderReranker._load_model()` — thử FlagReranker trước, fallback sang CrossEncoder
  - `CrossEncoderReranker.rerank()` — predict scores → sort → top-k RerankResult
  - `FlashrankReranker.rerank()` — lightweight alternative
  - `benchmark_reranker()` — đo avg/min/max latency qua n_runs
- Số tests pass: **5/5**

## 2. Kiến thức học được

- **Khái niệm mới nhất:** Cross-encoder vs Bi-encoder — bi-encoder encode query và document riêng (fast, dùng cho retrieval), cross-encoder encode cả cặp (slow, accurate, dùng cho reranking). Đây là lý do reranking chỉ áp dụng cho top-20, không phải toàn bộ corpus.
- **Điều bất ngờ nhất:** bge-reranker-v2-m3 hiểu tiếng Việt tốt hơn mong đợi — query "nghỉ phép bao nhiêu ngày" → rank đúng chunk "12 ngày/năm" lên #1, đẩy chunk "30 ngày không lương" xuống dù BM25 score của 2 chunk gần nhau.
- **Kết nối với bài giảng:** Slide "Reranking pipeline" — bài giảng nói latency là trade-off chính. Benchmark cho thấy first load ~3-5s (model download), subsequent calls ~200-500ms — acceptable cho production.

## 3. Khó khăn & Cách giải quyết

- **Khó khăn lớn nhất:** FlagEmbedding và sentence_transformers có API khác nhau (`compute_score` vs `predict`), cần handle cả 2 cases.
- **Cách giải quyết:** Try/except cho từng import, lưu `_model_type` để biết dùng API nào. Fallback cuối là sort theo original score nếu không có model nào.
- **Thời gian debug:** ~10 phút cho model loading logic, ~5 phút cho scores type handling (scalar vs list).

## 4. Nếu làm lại

- **Sẽ làm khác:** Cache model ở module level thay vì instance level — tránh reload model mỗi lần tạo CrossEncoderReranker mới.
- **Module nào muốn thử tiếp:** M4 Evaluation — muốn thấy con số cụ thể reranking cải thiện context_precision bao nhiêu.

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | 4 |
| Code quality | 5 |
| Teamwork | 4 |
| Problem solving | 4 |
