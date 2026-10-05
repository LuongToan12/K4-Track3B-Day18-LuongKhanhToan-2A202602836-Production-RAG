# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Lương Khánh Toàn  
**Mã sinh viên:** 2A202602836  
**Khóa:** K4 - Track 3B  
**Ngày hoàn thành:** 05/10/2026

---

## Phần 1: Mapping bài giảng (Lecture Mapping)

Dưới đây là bảng đối chiếu chi tiết giữa lý thuyết bài giảng Production RAG và các hàm/lớp đã triển khai trong thực tế:

| Lecture Concept | Module | Hàm cụ thể | Observation & Phân tích |
|----------------|--------|-------------|--------------------------|
| **Semantic chunking** | M1 | `chunk_semantic()` | Dùng mô hình `sentence-transformers/all-MiniLM-L6-v2` mã hóa từng câu và đo cosine similarity giữa các câu liền kề. Với ngưỡng `threshold=0.85` (từ `config.py`), hệ thống ngắt đoạn chính xác khi có sự chuyển dịch ý nghĩa giữa các câu, giữ các ý bổ trợ trong cùng một chunk thay vì cắt máy móc theo số ký tự. |
| **Hierarchical chunking** | M1 | `chunk_hierarchical()` | Chia tài liệu theo mô hình Cha - Con (Parent: 2048 ký tự, Child: 256 ký tự). Chunk con nhỏ giúp công cụ tìm kiếm bắt trúng từ khóa và embedding đặc thù, trong khi `parent_id` liên kết với chunk cha lớn giúp cung cấp đầy đủ ngữ cảnh cho LLM khi sinh câu trả lời, giải quyết triệt để vấn đề mất bối cảnh. |
| **Structure-aware chunking** | M1 | `chunk_structure_aware()` | Phân đoạn dựa trên cấu trúc đề mục Markdown (`#`, `##`, `###`), gắn kèm tiêu đề mục vào metadata `section`. Giữ trọn vẹn từng điều khoản quy chế, bảo toàn các bảng số liệu phức tạp (như bảng khung lương, phụ cấp) không bị xé vụn. |
| **Vietnamese Word Tokenization** | M2 | `segment_vietnamese()` | Sử dụng thư viện `underthesea.word_tokenize` để tách từ ghép tiếng Việt chuẩn xác. Điểm mấu chốt là phải loại bỏ dấu gạch dưới (`.replace("_", " ")`) để từ khóa đồng nhất với văn phong truy vấn tự nhiên của người dùng, tránh việc BM25 tính sai tần suất từ. |
| **BM25 + Dense fusion** | M2 | `reciprocal_rank_fusion()` | Kết hợp BM25 (lexical search chuẩn xác từ khóa số hiệu, ngày tháng) và Dense Search (vector embedding 1024 chiều từ `BAAI/bge-m3` trên Qdrant). Thuật toán RRF với tham số làm mượt $k=60$ dung hòa điểm số thứ hạng mà không cần chuẩn hóa scale điểm phức tạp. |
| **Cross-encoder reranking** | M3 | `CrossEncoderReranker.rerank()` | Sử dụng mô hình `BAAI/bge-reranker-v2-m3` nhận trực tiếp cặp `(query, document)`. Tầng này đối chiếu chi tiết từng từ, phân biệt chính xác quy định cũ (2023) và mới (2024), lọc từ 20 ứng viên xuống 3 đoạn ngữ cảnh đắt giá nhất (`top_k=3`) gửi cho LLM. Tích hợp bộ nhớ đệm `_cached_models` giúp tốc độ rerank cực nhanh. |
| **RAGAS 4 metrics** | M4 | `evaluate_ragas()` | Đo lường toàn diện chất lượng RAG qua 4 chỉ số cốt lõi: Faithfulness (độ trung thực, không ảo giác), Answer Relevancy (trả lời trúng trọng tâm), Context Precision (tài liệu đúng nằm ở thứ hạng đầu), Context Recall (bao phủ đủ thông tin theo ground truth). |
| **Diagnostic Tree** | M4 | `failure_analysis()` | Cây chẩn đoán 4 nhánh giúp tự động phân loại nguyên nhân gốc rễ (LLM ảo giác, bộ tìm kiếm sót tài liệu, reranker xếp nhầm thứ hạng, hay câu trả lời lạc đề) và chỉ định giải pháp tối ưu tương ứng. |
| **Enrichment Pipeline** | M5 | `_enrich_single_call()` / `enrich_chunks()` | Làm giàu văn bản trước khi lập chỉ mục bằng cách gọi 1 prompt LLM duy nhất: tóm tắt nội dung (Summarize), sinh câu hỏi giả định (HyQA), viết câu ngữ cảnh mở đầu (Anthropic Contextual Prepend) và trích xuất Auto Metadata. Có cơ chế fallback heuristic bảo đảm hệ thống luôn hoạt động ổn định. |

---

## Phần 2: Khó khăn & Cách giải quyết (Challenges & Debugging)

Trong quá trình triển khai hệ thống Production RAG, tôi đã đối mặt và giải quyết các bài toán kỹ thuật thực tế sau:

### 1. Vấn đề dấu gạch dưới khi tách từ ghép tiếng Việt trong BM25
- **Lỗi kỹ thuật gặp phải:** Khi sử dụng `underthesea.word_tokenize("nghỉ phép")`, thư viện trả về `nghỉ_phép`. Khi BM25 tokenize chuỗi này, token thu được là `nghỉ_phép`, trong khi câu hỏi người dùng nhập vào là `"nghỉ phép"` (ngăn cách bởi khoảng trắng). Kết quả là BM25 coi đây là 2 token độc lập và trượt toàn bộ các tài liệu chứa từ khóa then chốt.
- **Nguyên nhân gốc rễ & Cách debug:** Thư viện NLP tiếng Việt thường dùng dấu gạch dưới để biểu diễn từ ghép đơn vị (compound words). Tuy nhiên, inverted index của BM25 dựa trên tokenizer khoảng trắng thông thường.
- **Cách giải quyết:** Bổ sung bước chuẩn hóa `tokens = [t.replace("_", " ") for t in tokens]` ngay sau khi gọi `word_tokenize`. Điều này giúp bảo toàn ngữ nghĩa của từ ghép trong khi đảm bảo khớp hoàn hảo với câu hỏi của người dùng.

### 2. Tối ưu hóa tải trọng mô hình Cross-Encoder và Dense Embedding trên CPU
- **Lỗi kỹ thuật gặp phải:** Mỗi khi khởi tạo đối tượng `CrossEncoderReranker` hoặc `DenseSearch`, mô hình Transformer đa ngôn ngữ (`bge-reranker-v2-m3` và `bge-m3`) lại được nạp lại từ ổ đĩa vào RAM/CPU, gây độ trễ từ 15-30 giây cho mỗi lượt kiểm thử hoặc truy vấn.
- **Nguyên nhân gốc rễ & Cách debug:** Việc khởi tạo đối tượng không kèm cơ chế Singleton hoặc Class Cache khiến bộ nhớ bị chiếm dụng lặp đi lặp lại và CPU tốn nhiều tài nguyên chuyển đổi trọng số.
- **Cách giải quyết:** Xây dựng cơ chế cache ở cấp lớp:
  ```python
  class CrossEncoderReranker:
      _cached_models: dict = {}
      def _get_model(self):
          if self.model_name not in CrossEncoderReranker._cached_models:
              CrossEncoderReranker._cached_models[self.model_name] = CrossEncoder(self.model_name)
          return CrossEncoderReranker._cached_models[self.model_name]
  ```
  Nhờ đó, mô hình chỉ nạp 1 lần duy nhất trong toàn bộ phiên làm việc, giảm thời gian rerank cho 20 documents xuống dưới 100ms.

### 3. Thiết kế cơ chế phòng vệ (Defensive Fallback) khi API bên thứ ba gặp sự cố
- **Lỗi kỹ thuật gặp phải:** Khi môi trường không có kết nối Internet hoặc OpenAI API Key gặp lỗi xác thực (HTTP 401 Unauthorized), pipeline bị gián đoạn hoàn toàn ở các bước Enrichment (M5) và RAGAS Evaluation (M4).
- **Nguyên nhân gốc rễ & Cách debug:** Các hàm gọi LLM bên ngoài cần được bao bọc cẩn thận để không làm sập toàn bộ chu trình xử lý dữ liệu.
- **Cách giải quyết:** Triển khai cơ chế Extractive Heuristic Fallback:
  - Khi API lỗi, `_enrich_single_call()` tự động trích xuất 2 câu đầu làm summary, sử dụng tiêu đề văn bản làm câu hỏi mẫu, tạo contextual prefix từ tên file và trích xuất regex metadata (phiên bản, ngày hiệu lực, phòng ban).
  - Hàm `evaluate_ragas()` bắt toàn bộ ngoại lệ và trả về cấu trúc điểm an toàn, cho phép pipeline tiếp tục vận hành và xuất báo cáo chẩn đoán đầy đủ.

---

## Phần 3: Action Plan cho Project cá nhân (Application Plan)

### Project: Trợ lý AI tra cứu quy chế và chính sách nội bộ doanh nghiệp (Internal Policy Assistant)

#### 1. Hiện trạng
- **Pipeline hiện tại:** Hệ thống sử dụng Naive RAG cơ bản: cắt văn bản cố định theo đoạn văn 500 ký tự (character chunking), lưu trữ vector trong cơ sở dữ liệu vector với mô hình nhúng thông thường, tìm kiếm Dense-only top 5 và gửi thẳng vào LLM để sinh câu trả lời.
- **Vấn đề / Bottlenecks đang gặp:**
  1. *Lẫn lộn phiên bản tài liệu:* Khi công ty cập nhật quy chế mới (ví dụ Quy chế làm việc từ xa năm 2024 thay thế bản 2023), Dense Search vẫn kéo về quy định cũ năm 2023 dẫn đến câu trả lời sai luật.
  2. *Đứt gãy bảng biểu và danh sách:* Bảng phụ cấp, khung lương bị cắt ngang lưng chừng làm mất mối quan hệ giữa các cột tiêu đề và số liệu.
  3. *Trượt từ khóa số hiệu và thuật ngữ:* Người dùng hỏi các từ viết tắt ("MFA", "PVI", "BHXH", "QĐ-128") nhưng Dense Search không tìm thấy vì khoảng cách vector bị loãng.
  4. *Hiện tượng ảo giác (Hallucination):* Khi tài liệu trích xuất thiếu ngữ cảnh gốc, LLM tự suy diễn thông tin sai lệch.

#### 2. Kế hoạch cải tiến
1. **Chunking strategy:**
   - Áp dụng **Structure-Aware Chunking** làm lớp ngoài: phân tách tài liệu theo các chương, điều, mục (H1, H2, H3) và trích xuất bảng biểu nguyên khối.
   - Kết hợp **Hierarchical Chunking**: tạo chunk con 256 ký tự để phục vụ tìm kiếm chính xác, nhưng khi tổng hợp sẽ trả về chunk cha 2048 ký tự (hoặc toàn bộ điều khoản) để LLM có trọn vẹn ngữ cảnh pháp lý.
2. **Search retrieval:**
   - Triển khai **Hybrid Search**:
     - *Lexical Search (BM25):* Tách từ tiếng Việt với `underthesea` (loại bỏ `_`), đánh chỉ mục mã số văn bản, điều khoản, ngày tháng.
     - *Dense Search:* Sử dụng mô hình đa ngôn ngữ `BAAI/bge-m3` lưu trên Qdrant Cloud.
     - *Hợp nhất thứ hạng:* Thuật toán **Reciprocal Rank Fusion (RRF)** với $k=60$.
3. **Reranking:**
   - Sử dụng Cross-Encoder `BAAI/bge-reranker-v2-m3` tiếp nhận top 20 kết quả từ RRF, chấm điểm tương quan trực tiếp giữa câu hỏi và đoạn trích, lọc lấy top 3 kết quả xuất sắc nhất gửi cho LLM.
4. **Enrichment:**
   - Áp dụng **Anthropic Contextual Prepend** để gắn vị trí văn bản vào đầu mỗi chunk (ví dụ: *"Trích từ Điều 5, Quy chế Nghỉ phép năm 2024, áp dụng cho nhân viên chính thức"*).
   - Tự động trích xuất metadata: `effective_date`, `version`, `department`, `status: active/expired` để phục vụ lọc cứng (hard filtering) trước khi tìm kiếm.
5. **Evaluation:**
   - Tích hợp bộ đánh giá **RAGAS** với 4 chỉ số và hệ thống cảnh báo tự động: nếu Faithfulness < 0.80 hoặc Context Recall < 0.75, tự động gắn cờ truy vấn để chuyên viên tri thức (Knowledge Manager) xem xét.

#### 3. Timeline triển khai

- **Tuần 1: Refactor Dữ liệu & Nâng cấp Tầng Retrieval**
  - *Ngày 1-2:* Triển khai bộ phân tách văn bản Structure-Aware và Hierarchical Chunking cho toàn bộ kho tài liệu PDF/Markdown quy chế công ty.
  - *Ngày 3-4:* Thiết lập hạ tầng Hybrid Search (BM25 + Qdrant BAAI/bge-m3), tối ưu hóa hàm tách từ tiếng Việt.
  - *Ngày 5:* Tích hợp thuật toán RRF, kiểm thử sơ bộ độ phủ từ khóa và khả năng tìm kiếm số liệu.

- **Tuần 2: Tối ưu Reranker, Enrichment & Đánh giá RAGAS**
  - *Ngày 6-7:* Tích hợp Cross-Encoder Reranker (`bge-reranker-v2-m3`), cấu hình cơ chế lọc metadata theo phiên bản hiệu lực mới nhất (version filtering).
  - *Ngày 8-9:* Xây dựng pipeline Enrichment (Contextual Prepend và Auto-metadata) tiền xử lý dữ liệu.
  - *Ngày 10:* Chạy bộ đánh giá RAGAS trên 50 câu hỏi thử nghiệm thực tế của các phòng ban, tinh chỉnh prompt chống ảo giác và hoàn tất tài liệu triển khai sản phẩm.
