# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Lương Khánh Toàn  
**Mã sinh viên:** 2A202602836  
**Khóa:** K4 - Track 3B  

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.4500 | 0.8200 | +0.3700 |
| Answer Relevancy | 0.5500 | 0.7900 | +0.2400 |
| Context Precision | 0.4200 | 0.8400 | +0.4200 |
| Context Recall | 0.5100 | 0.7800 | +0.2700 |

> **Nhận xét tổng quan:**  
> Hệ thống Production RAG mang lại bước nhảy vọt toàn diện so với Naive Baseline:
> - **Context Precision** tăng mạnh nhất (+0.4200), đạt 0.8400 nhờ sự kết hợp giữa Hybrid Search (BM25 + Dense) và tầng Cross-Encoder Reranker lọc đúng 3 đoạn đắt giá nhất.
> - **Faithfulness** tăng từ 0.4500 lên 0.8200 (+0.3700), giải quyết triệt để vấn đề ảo giác và cắt đứt văn bản nhờ kỹ thuật Hierarchical Chunking (bảo toàn ngữ cảnh cha 2048 ký tự).
> - **Answer Relevancy** và **Context Recall** đều vượt ngưỡng tiêu chuẩn sản xuất (> 0.75).

---

## Bottom-5 Failures

### #1
- **Question:** Nhân viên được nghỉ bao nhiêu ngày phép năm?
- **Expected:** Theo chính sách hiện hành (v2024), nhân viên được nghỉ 15 ngày phép năm có lương. Chính sách cũ (v2023) là 12 ngày nhưng đã bị thay thế.
- **Got:** Trích xuất tài liệu `nghi_phep_nam_v2023.md` và trả lời "Nhân viên chính thức được nghỉ 12 ngày phép năm".
- **Worst metric:** `context_precision` (0.35)
- **Error Tree:** Output sai → Context sai (chứa văn bản cũ hết hiệu lực) → Query thiếu mốc thời gian → Reranker chưa phân biệt được văn bản active vs expired.
- **Root cause:** Xung đột phiên bản tài liệu (Version Conflict). Kho dữ liệu có cả 2 phiên bản 2023 và 2024. Cả hai đều có độ tương đồng ngữ nghĩa rất cao với câu hỏi của người dùng, nên nếu không có cơ chế lọc siêu dữ liệu theo trạng thái hiệu lực, tài liệu cũ vẫn có thể lọt vào top 3.
- **Suggested fix:** Bổ sung metadata hard-filtering (`status: "active"`, `effective_year: 2024`) trước khi retrieval; hoặc dùng Contextual Prepend đánh dấu rõ `[TÀI LIỆU HẾT HIỆU LỰC]` vào chunk 2023.

---

### #2
- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Expected:** Theo chính sách v2024: 15 ngày cơ bản + 3 ngày thâm niên (9 ÷ 3 = 3) = 18 ngày phép. Lương Senior (P3-P4): 20-35 triệu VNĐ/tháng.
- **Got:** Trả lời đúng khung lương 20-35 triệu VNĐ/tháng nhưng tính sai số ngày phép (16 ngày vì áp dụng công thức cũ 5 năm + 1 ngày của bản 2023).
- **Worst metric:** `faithfulness` (0.40)
- **Error Tree:** Output sai một phần → Context bị loãng giữa 2 chủ đề độc lập → Query phức hợp 2 vế → Cần phân rã câu hỏi (Query Decomposition).
- **Root cause:** Câu hỏi đa chặng (multi-hop) đòi hỏi tổng hợp thông tin từ 2 văn bản khác nhau (`bang_luong_2024.md` và `nghi_phep_nam_v2024.md`). Top-3 context bị chiếm phần lớn bởi văn bản bảng lương, khiến mảnh ghép về thâm niên nghỉ phép bị thiếu hoặc bị lẫn với phiên bản cũ.
- **Suggested fix:** Áp dụng kỹ thuật Query Decomposition (tách thành Sub-query 1: "Lương Senior" và Sub-query 2: "Thâm niên 9 năm được bao nhiêu ngày phép năm v2024"), thực hiện tìm kiếm song song rồi gộp ngữ cảnh lại.

---

### #3
- **Question:** Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?
- **Expected:** Thời hạn thanh toán là 15 ngày. Quá hạn 5 ngày, bị tính phí 2%/tháng trên 15.000.000 VNĐ = 300.000 VNĐ/tháng (tính pro-rata khoảng 50.000 VNĐ cho 5 ngày).
- **Got:** Trả lời "Thời hạn thanh toán tạm ứng là 15 ngày, quá hạn sẽ bị trừ vào lương" nhưng không nêu được mức phạt phần trăm cụ thể.
- **Worst metric:** `context_recall` (0.45)
- **Error Tree:** Output thiếu ý → Context bị cắt giữa điều khoản thời hạn và điều khoản mức phạt → Query OK → Chunking làm tách rời các điều liên quan.
- **Root cause:** Kích thước đoạn con quá nhỏ hoặc ranh giới cắt đoạn nằm ngay giữa Điều 3 (Thời hạn thanh toán) và Điều 4 (Biện pháp xử lý quá hạn), khiến ngữ cảnh gửi cho LLM chỉ có phần đầu của quy định.
- **Suggested fix:** Khi gửi cho LLM, luôn sử dụng Parent Chunk (kích thước 2048 ký tự) thay vì Child Chunk đơn lẻ để bảo đảm bao hàm trọn vẹn cả thời hạn và chế tài xử lý.

---

### #4
- **Question:** Thông tin lương thuộc cấp độ phân loại dữ liệu nào?
- **Expected:** Theo quy chế chi trả lương, thông tin lương được phân loại là dữ liệu Bí mật, cấm chia sẻ với đồng nghiệp. Theo chính sách phân loại dữ liệu, dữ liệu Bí mật (cấp 3) phải mã hóa khi truyền và hạn chế truy cập theo need-to-know.
- **Got:** Trả lời "Thông tin lương là dữ liệu Nội bộ (Cấp 2)" do trích nhầm định nghĩa dữ liệu chung trong sổ tay nhân viên.
- **Worst metric:** `context_precision` (0.50)
- **Error Tree:** Output sai cấp độ → Context trích nhầm tài liệu đại trà thay vì tài liệu chuyên sâu → Query ngắn → Cần làm giàu bằng HyQA.
- **Root cause:** Từ khóa "thông tin lương" xuất hiện với tần suất dày đặc ở nhiều tài liệu (`bang_luong_2024.md`, `ky_luong.md`), làm loãng điểm số của tài liệu cốt lõi `phan_loai_du_lieu.md`.
- **Suggested fix:** Áp dụng HyQA (Hypothetical Question Answering) tại khâu Enrichment: sinh sẵn các câu hỏi giả định như "Bảng lương thuộc cấp độ bảo mật nào?" gắn vào chunk phân loại dữ liệu để tăng mức độ khớp khi truy vấn.

---

### #5
- **Question:** Nếu cần mua một chiếc laptop 30 triệu cho nhân viên mới, ai phê duyệt và cần gì từ phòng CNTT?
- **Expected:** Laptop 30 triệu nằm trong khoảng 5-50 triệu nên cần Giám đốc phòng ban (Director) phê duyệt. Ngoài ra, mua sắm thiết bị CNTT cần có xác nhận cấu hình kỹ thuật từ phòng CNTT trước khi đề xuất. Cần đính kèm ít nhất 3 báo giá vì trên 10 triệu.
- **Got:** Trả lời cần Giám đốc phòng ban phê duyệt nhưng bỏ sót điều kiện "cần xác nhận cấu hình kỹ thuật từ phòng CNTT" và "đính kèm ít nhất 3 báo giá".
- **Worst metric:** `context_recall` (0.55)
- **Error Tree:** Output thiếu các điều kiện con → Context có thông tin nhưng bị chia nhỏ → Prompt sinh câu trả lời chưa ép liệt kê toàn bộ điều kiện.
- **Root cause:** Tài liệu `mua_sam.md` có nhiều quy định lồng nhau (phân cấp hạn mức, quy định riêng cho thiết bị CNTT, quy định số lượng báo giá). Mô hình LLM khi nhận context chưa trích xuất triệt để mọi điều kiện ràng buộc.
- **Suggested fix:** Cải thiện System Prompt: "Liệt kê đầy đủ tất cả các bên phê duyệt, thủ tục kỹ thuật bắt buộc và hồ sơ chứng từ cần đính kèm", đồng thời dùng Structure-Aware Chunking để giữ nguyên khối toàn bộ mục "Quy trình mua sắm thiết bị IT".

---

## Case Study (cho presentation)

**Question chọn phân tích:**  
`"Nhân viên được nghỉ bao nhiêu ngày phép năm?"` (Xung đột phiên bản v2023 vs v2024)

**Error Tree walkthrough:**
1. **Output đúng?** → **KHÔNG**. Hệ thống trả về 12 ngày (quy định cũ đã hết hiệu lực từ ngày 01/01/2024), trong khi đáp án chính xác là 15 ngày.
2. **Context đúng?** → **KHÔNG**. Top 3 context trích xuất có chứa chunk từ `nghi_phep_nam_v2023.md` xếp trên `nghi_phep_nam_v2024.md`.
3. **Query rewrite OK?** → Câu hỏi của người dùng không đề cập rõ năm cụ thể ("...được nghỉ bao nhiêu ngày phép năm?"), hệ thống cần tự động hiểu là đang hỏi chính sách hiện hành có hiệu lực tại thời điểm hiện tại.
4. **Fix ở bước:**  
   - **Tầng Tiền xử lý (Enrichment):** Trích xuất metadata `version` và `status` (hiệu lực / hết hiệu lực).
   - **Tầng Retrieval:** Bổ sung bộ lọc cứng (Metadata Filtering) ưu tiên các tài liệu đang còn hiệu lực (`status == "active"`).
   - **Tầng Reranking:** Bổ sung tín hiệu về ngày hiệu lực (Recency Boost) vào Cross-Encoder để các chính sách mới luôn được ưu tiên xếp hạng cao hơn.

**Nếu có thêm 1 giờ, sẽ optimize:**
- **Triển khai Temporal / Recency Reranking:** Tự động phát hiện phiên bản tài liệu từ metadata (`v2023` vs `v2024`, ngày ban hành) để cộng điểm ưu tiên (score boost) cho văn bản mới nhất.
- **Tích hợp Query Decomposition & Routing:** Đối với các câu hỏi phức hợp nhiều ý (như Case #2), tự động phân rã câu hỏi thành nhiều truy vấn con độc lập và tổng hợp đa nguồn.
- **Parent-Child Retrieval Assembly:** Mở rộng ngữ cảnh gửi tới LLM bằng toàn bộ chunk cha (Parent Chunk) thay vì chỉ gửi chunk con, loại bỏ hiện tượng đứt gãy câu và số liệu.
