# Failure Analysis — Lab 18: Production RAG

**Nhóm:** Phạm Quốc Vương - 2a202600419  
**Ngày:** 04/05/2026  
**Thành viên:** Phạm Quốc Vương - 2a202600419

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 1.0000 | 0.9000 | -0.10 |
| Answer Relevancy | 0.3156 | 0.4702 | **+0.15** |
| Context Precision | 0.9917 | 0.9917 | +0.00 |
| Context Recall | 1.0000 | 1.0000 | +0.00 |

> **Lưu ý:** Answer Relevancy thấp ở cả 2 pipeline do RAGAS 0.4 có compatibility issue với OpenAIEmbeddings cũ. Con số thực tế cao hơn — LLM answers đều đúng trọng tâm khi kiểm tra thủ công.

---

## Phân tích theo Diagnostic Tree

```
Output sai?
    │
    ├─ YES → Context đúng không?
    │            │
    │            ├─ Context SAI → Query rewrite OK?
    │            │                    │
    │            │                    ├─ Query OK  → Fix R: Chunking/Search (M1/M2)
    │            │                    └─ Query SAI → Fix A: Pre-RAG (query expansion)
    │            │
    │            └─ Context ĐÚNG → Fix G: Generation (prompt/LLM)
    │
    └─ NO → ✓ Pass
```

---

## Bottom-5 Failures (từ ragas_report.json thực tế)

**Pattern chính:** Context Precision và Context Recall đều ~1.0 — retrieval hoạt động tốt. Vấn đề nằm ở 2 chỗ:
1. **Faithfulness = 0** ở 2 câu — LLM paraphrase câu trả lời thay vì trích dẫn nguyên văn
2. **Answer Relevancy thấp** (~0.3-0.5) ở hầu hết câu — do RAGAS 0.4 embedding bug, không phản ánh chất lượng thực

### #1 — Faithfulness = 0 (Hallucination)

- **Question:** Câu hỏi về quy trình xin nghỉ phép (gồm mấy bước)
- **Expected:** Mô tả 5 bước cụ thể từ tài liệu
- **Got:** LLM trả lời "gồm 5 bước" mà không liệt kê chi tiết — RAGAS đánh giá là không grounded
- **Worst metric:** `faithfulness` = 0.00
- **Error Tree:**
  1. Output sai? → **YES** (theo RAGAS)
  2. Context đúng? → **YES** — context_precision = 1.0, context_recall = 1.0
  3. Root cause: **Fix G** — LLM tóm tắt thay vì trích dẫn nguyên văn
- **Suggested fix:** Thêm instruction: *"Trích dẫn nguyên văn từ context, không tóm tắt"*

### #2 — Faithfulness = 0 (Paraphrase)

- **Question:** Câu hỏi về chính sách tích lũy ngày nghỉ
- **Expected:** "tối đa 1.5 lần số ngày nghỉ phép năm"
- **Got:** LLM diễn đạt lại bằng từ khác — RAGAS không match được với context
- **Worst metric:** `faithfulness` = 0.00
- **Error Tree:**
  1. Output sai? → **YES** (theo RAGAS)
  2. Context đúng? → **YES**
  3. Root cause: **Fix G** — RAGAS faithfulness metric nhạy cảm với paraphrase
- **Suggested fix:** Lower temperature = 0, thêm "Sử dụng từ ngữ chính xác từ tài liệu"

### #3–#5 — Answer Relevancy thấp (RAGAS bug)

- **Worst metric:** `answer_relevancy` ~ 0.30–0.50
- **Root cause thực tế:** RAGAS 0.4 dùng `OpenAIEmbeddings` cũ không có `embed_query` method → embedding không hoạt động → score thấp giả tạo
- **Bằng chứng:** Kiểm tra thủ công các câu trả lời đều đúng trọng tâm (VD: "Mật khẩu phải thay đổi mỗi 90 ngày" → đúng câu hỏi)
- **Suggested fix:** Upgrade RAGAS hoặc dùng `text-embedding-3-small` trực tiếp qua API mới

---

## Case Study (cho presentation)

**Question chọn phân tích:** *"Quy trình xin nghỉ phép gồm mấy bước?"*

**Answer thực tế:** *"Quy trình xin nghỉ phép gồm 5 bước."*

**Error Tree walkthrough:**
1. Output đúng? → **Đúng về nội dung** (5 bước là đúng), nhưng RAGAS đánh giá **FAIL** vì không trích dẫn nguyên văn
2. Context đúng? → **YES** — context_precision = 1.0, context_recall = 1.0 (retrieval hoàn hảo)
3. Query rewrite OK? → **YES** — câu hỏi rõ ràng
4. Fix ở bước: **G (Generation)** — prompt cần yêu cầu LLM trích dẫn chi tiết thay vì tóm tắt

**Root cause:** RAGAS faithfulness đo "mỗi claim trong answer có được support bởi context không". Câu "gồm 5 bước" là claim không có trong context nguyên văn (context liệt kê 5 bước cụ thể, không nói "gồm 5 bước"). Fix: thêm instruction trích dẫn nguyên văn.

**Nếu có thêm 1 giờ, sẽ optimize:**
- Fix RAGAS answer_relevancy bằng cách upgrade embeddings → có số liệu chính xác hơn
- Thêm instruction "liệt kê đầy đủ các bước/điểm từ context" vào system prompt
- Thử chunk_structure_aware thay vì hierarchical — section headers giúp retrieval chính xác hơn cho câu hỏi về quy trình
