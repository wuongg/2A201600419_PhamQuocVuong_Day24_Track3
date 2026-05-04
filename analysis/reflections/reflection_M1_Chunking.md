# Individual Reflection — Lab 18

**Tên:** Phạm Quốc Vương - 2a202600419  
**Module phụ trách:** M1 — Advanced Chunking Strategies

---

## 1. Đóng góp kỹ thuật

- Module đã implement: `src/m1_chunking.py` — 3 advanced chunking strategies
- Các hàm/class chính đã viết:
  - `chunk_semantic()` — encode câu bằng `all-MiniLM-L6-v2`, nhóm theo cosine similarity
  - `chunk_hierarchical()` — parent (2048 chars) + child (256 chars), mỗi child có `parent_id`
  - `chunk_structure_aware()` — regex split markdown headers, mỗi section thành 1 chunk
  - `compare_strategies()` — chạy cả 4 strategies, in bảng so sánh stats
- Số tests pass: **13/13**

## 2. Kiến thức học được

- **Khái niệm mới nhất:** Hierarchical chunking — index child nhỏ để embedding chính xác, nhưng trả về parent lớn cho LLM để có đủ context. Đây là pattern production thực tế, không chỉ là lý thuyết.
- **Điều bất ngờ nhất:** Semantic chunking với threshold 0.85 tạo ra ít chunks hơn basic chunking nhưng mỗi chunk coherent hơn nhiều — giảm noise cho embedding.
- **Kết nối với bài giảng:** Slide về "Chunking strategies" — bài giảng nói hierarchical là "production default", sau khi implement mới hiểu tại sao: precision từ child + context từ parent.

## 3. Khó khăn & Cách giải quyết

- **Khó khăn lớn nhất:** `chunk_hierarchical()` — logic gom paragraphs vào parent rồi slide window tạo children, dễ bị off-by-one và tạo ra children rỗng.
- **Cách giải quyết:** Thêm `if child_text.strip()` để lọc children rỗng; test với `parent_size=200, child_size=80` để dễ debug hơn với text ngắn.
- **Thời gian debug:** ~20 phút cho hierarchical, ~10 phút cho structure-aware (regex split).

## 4. Nếu làm lại

- **Sẽ làm khác:** Implement semantic chunking với sliding window thay vì chỉ compare consecutive sentences — sẽ nhóm tốt hơn khi topic shift dần dần.
- **Module nào muốn thử tiếp:** M5 Enrichment — contextual prepend nghe rất thú vị, muốn thử xem nó cải thiện retrieval bao nhiêu % so với raw chunks.

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | 4 |
| Code quality | 4 |
| Teamwork | 5 |
| Problem solving | 4 |
