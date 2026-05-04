# Individual Reflection — Lab 18

**Tên:** Phạm Quốc Vương - 2a202600419  
**Module phụ trách:** M2 — Hybrid Search

---

## 1. Đóng góp kỹ thuật

- Module đã implement: `src/m2_search.py` — BM25 + Dense + RRF fusion
- Các hàm/class chính đã viết:
  - `segment_vietnamese()` — underthesea word_tokenize, fallback về raw text
  - `BM25Search.index()` + `.search()` — BM25Okapi trên text đã segment
  - `DenseSearch.index()` + `.search()` — bge-m3 + Qdrant vector DB
  - `reciprocal_rank_fusion()` — merge rankings, score = Σ 1/(k+rank)
- Số tests pass: **5/5**

## 2. Kiến thức học được

- **Khái niệm mới nhất:** RRF (Reciprocal Rank Fusion) — không cần tune weight giữa BM25 và dense, chỉ cần rank position. Robust hơn weighted sum vì không bị ảnh hưởng bởi score scale khác nhau.
- **Điều bất ngờ nhất:** Vietnamese segmentation quan trọng hơn tưởng — "nghỉ phép" nếu không segment sẽ bị tách thành "nghỉ" + "phép", BM25 score sai hoàn toàn. Underthesea giải quyết được nhưng cần download model ~100MB.
- **Kết nối với bài giảng:** Slide "Hybrid Search" — bài giảng nói BM25 tốt cho exact match, dense tốt cho semantic. Sau khi implement mới thấy rõ: query "nghỉ phép năm" → BM25 rank #1 đúng chunk, dense rank #3 vì embedding gần với nhiều chunks khác.

## 3. Khó khăn & Cách giải quyết

- **Khó khăn lớn nhất:** Qdrant connection — nếu Docker chưa chạy thì DenseSearch crash toàn bộ pipeline.
- **Cách giải quyết:** Wrap tất cả Qdrant calls trong try/except, print warning thay vì raise exception. Pipeline vẫn chạy với BM25-only khi Qdrant không available.
- **Thời gian debug:** ~15 phút cho RRF logic (nhầm 0-indexed vs 1-indexed rank), ~10 phút cho Qdrant error handling.

## 4. Nếu làm lại

- **Sẽ làm khác:** Thêm sparse vector support trong Qdrant (SPLADE) thay vì BM25 riêng — sẽ gọn hơn và Qdrant hỗ trợ native hybrid search.
- **Module nào muốn thử tiếp:** M3 Reranking — muốn xem cross-encoder cải thiện precision bao nhiêu sau khi hybrid search đã có recall tốt.

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | 5 |
| Code quality | 4 |
| Teamwork | 4 |
| Problem solving | 4 |
