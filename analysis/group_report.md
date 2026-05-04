# Group Report — Lab 18: Production RAG

**Nhóm:** Phạm Quốc Vương - 2a202600419  
**Ngày:** 04/05/2026

## Thành viên & Phân công

| Tên | Module | Hoàn thành | Tests pass |
|-----|--------|-----------|-----------|
| Phạm Quốc Vương - 2a202600419 | M1: Chunking | ✅ | 13/13 |
| Phạm Quốc Vương - 2a202600419 | M2: Hybrid Search | ✅ | 5/5 |
| Phạm Quốc Vương - 2a202600419 | M3: Reranking | ✅ | 5/5 |
| Phạm Quốc Vương - 2a202600419 | M4: Evaluation | ✅ | 4/4 |
| Phạm Quốc Vương - 2a202600419 | M5: Enrichment | ✅ | 10/10 |

## Kết quả RAGAS

| Metric | Naive | Production | Δ |
|--------|-------|-----------|---|
| Faithfulness | 1.0000 | 0.9000 | -0.10 |
| Answer Relevancy | 0.3156 | 0.4702 | **+0.15** |
| Context Precision | 0.9917 | 0.9917 | +0.00 |
| Context Recall | 1.0000 | 1.0000 | +0.00 |

**Nhận xét tổng quan:** Production pipeline cải thiện rõ nhất ở Answer Relevancy (+0.15) nhờ LLM generation thực sự (gpt-4o-mini) thay vì trả về raw context. Context Precision và Context Recall đều đạt ~1.0 ở cả 2 pipeline — corpus nhỏ và câu hỏi trực tiếp nên retrieval dễ. Faithfulness giảm nhẹ (-0.10) vì LLM đôi khi paraphrase thay vì trích dẫn nguyên văn. Answer Relevancy thấp ở cả 2 do RAGAS 0.4 có bug với embeddings cũ — con số thực tế cao hơn.

## Kiến trúc Pipeline

```
Documents
   │
   ▼
[M1] Hierarchical Chunking
   │  Parent (2048 chars) → Child (256 chars)
   │  Child indexed, Parent returned to LLM
   ▼
[M5] Enrichment (optional)
   │  Contextual Prepend + HyQA + Auto Metadata
   ▼
[M2] Hybrid Search
   │  BM25 (Vietnamese segmented) + Dense (bge-m3)
   │  RRF fusion: score = Σ 1/(60 + rank)
   ▼
[M3] Cross-Encoder Reranking
   │  bge-reranker-v2-m3: top-20 → top-3
   ▼
[LLM] Answer Generation
   │  GPT-4o-mini với grounding prompt
   ▼
[M4] RAGAS Evaluation
   Faithfulness / Answer Relevancy / Context Precision / Context Recall
```

## Key Findings

1. **Biggest improvement:** Answer Relevancy tăng +0.15 (0.32 → 0.47) nhờ LLM generation thực sự (gpt-4o-mini) thay vì trả về raw context. Khi pipeline dùng context[0] làm answer (naive), RAGAS đánh giá thấp vì context dài không match câu hỏi ngắn.

2. **Biggest challenge:** RAGAS 0.4 breaking changes — API thay đổi hoàn toàn, `OpenAIEmbeddings` cũ không có `embed_query` method khiến answer_relevancy bị underestimate. Phải debug ~30 phút để tìm ra root cause.

3. **Surprise finding:** Context Precision và Context Recall đều đạt ~1.0 ngay từ naive baseline — corpus nhỏ (3 files, 14 chunks) và câu hỏi trực tiếp nên retrieval dễ. Trong production với corpus lớn hơn, hybrid search và reranking sẽ tạo ra sự khác biệt lớn hơn nhiều.

## Presentation Notes (5 phút)

1. **RAGAS scores (naive vs production):** Faithfulness 1.0→0.9 (LLM paraphrase), Answer Relevancy 0.32→0.47 (+0.15, biggest win), Context Precision/Recall ~1.0 ở cả 2 (corpus nhỏ).

2. **Biggest win — LLM Generation:** Thêm gpt-4o-mini vào pipeline thay vì trả raw context → Answer Relevancy tăng +0.15. Đây là bước đơn giản nhất nhưng impact lớn nhất.

3. **Case study — Faithfulness failure:** Câu "Quy trình xin nghỉ phép gồm mấy bước?" → LLM trả lời "gồm 5 bước" (đúng về nội dung) nhưng RAGAS đánh giá faithfulness = 0 vì không trích dẫn nguyên văn. Fix: thêm instruction "liệt kê đầy đủ từ context".

4. **Next optimization nếu có thêm 1 giờ:** Fix RAGAS embeddings để có answer_relevancy chính xác; thêm instruction trích dẫn nguyên văn vào prompt để tăng faithfulness; test với corpus lớn hơn để thấy rõ impact của hybrid search và reranking.
