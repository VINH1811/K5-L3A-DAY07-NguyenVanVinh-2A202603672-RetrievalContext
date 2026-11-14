# Báo cáo Day 7 — 2A202603672 Nguyễn Văn Vinh

Mọi số liệu dưới đây lấy từ file trong [results/](../../results/) và có thể chạy lại theo [src/README.md](../../src/README.md).

## Dữ liệu và benchmark

- **Nguồn, quyền sử dụng, số trang và số chunk:** hai báo cáo World Bank *Taking Stock* về Việt Nam (08/2023 "Making Public Investment Work for Growth" và 03/2025 "Electrifying Journeys"), tải công khai từ documents.worldbank.org, bản quyền World Bank, chỉ dùng cho bài học và không đưa PDF lên repo (URL và sha256 trong `data/manifest.json` của repo Day 6). 88 trang nội dung, 370 chunk (`by_paragraph`, tối đa 800 ký tự) — index B của Day 6.
- **Vector store, embedding model, phiên bản và số chiều:** Chroma `PersistentClient` 1.5.9 của Day 6, collection `day6_B`, cosine, HNSW (xấp xỉ). Embedding `google/embeddinggemma-2` chạy local bằng sentence-transformers 6.3.0, 768 chiều, chuẩn hóa L2, query prefix `task: search result | query: ` giống lúc tạo index. Dense của Day 7 cho lại đúng số Day 6 trên dev (Hit@1 11/15, Hit@3 15/15), xác nhận query embedding tương thích index.
- **Agent model, cách chạy và phân chia dev/test:** `qwen2.5:7b` qua Ollama 0.40.16, `temperature=0`, `seed=7`, `num_ctx=8192`, chạy CPU (`num_gpu=0`, vì runner GPU của Ollama báo lỗi CUDA trên máy này). Chi phí dịch vụ 0. Benchmark 24 câu trong [data/questions.json](../../data/questions.json): dev 15 câu có đáp án (q01–q15) + 2 câu thiếu bằng chứng (u01, u02); test 5 câu có đáp án (q16–q20) + 2 câu thiếu bằng chứng (u03, u04). 20 câu có đáp án và cách chia lấy nguyên từ Day 6 (chia trước khi chạy retrieval).
- **Cách xác định câu có/thiếu evidence và tiêu chí chấm:**
  - *Hit:* chunk đúng `source_id`, đúng trang và chứa trích dẫn evidence sau khi gộp khoảng trắng và bỏ phân biệt hoa thường (cùng định nghĩa Day 6; khác `hit` của starter vì text PDF xuống dòng giữa câu). Các phần tử `gold` là evidence thay thế. Cả 20 câu đều có evidence nằm trọn trong một chunk (kiểm tra bằng `src/export_day6.py`).
  - *Câu thiếu bằng chứng:* câu gần chủ đề; kiểm tra bằng regex trên toàn bộ 370 chunk, mẫu regex và số chunk khớp ghi trong `absence_reason` của từng câu (ví dụ `vin\s*fast|vingroup` → 0 chunk). Không tính vào mẫu số retrieval.
  - *Chấm câu trả lời:* 0/1/2 theo BENCHMARK.md; citation hợp lệ khi chunk được trích có trong context **và** nêu đúng điều câu trả lời khẳng định. Nhãn và lý do từng câu ở `results/<lượt chạy>/reviews.json`.
  - *Về bộ test:* Day 6 đã chạy dense trên 5 câu test này một lần sau khi chốt chiến lược bằng dev (kết quả 5/5), không dùng test để chỉnh. Ở Day 7 mọi lựa chọn (hybrid, context policy, prompt) đều làm trên dev; test chỉ chạy một lần sau khi chốt. Giới hạn: kết quả retrieval dense trên test đã biết trước từ Day 6.

## Retrieval và thử nghiệm

Cùng corpus, index `day6_B`, câu hỏi gốc (không viết lại), top-3. Thời gian là median mỗi query trên CPU; chi phí dịch vụ 0 cho mọi cấu hình.

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---|---|---|---|
| Baseline | Dense (Chroma Day 6), top-3, không filter, không rerank, 3000 ký tự | Mốc so sánh | dev | 11/15 (73%) | 15/15 (100%) | 15/15 (100%) | 104 ms / 0 USD |
| Phương pháp 2 | BM25 (k1 = 1,5; b = 0,75), top-3, 3000 ký tự | Khớp từ khóa và con số chính xác tốt hơn dense | dev | 10/15 (67%) | 14/15 (93%) | 14/15 (93%) | 1,3 ms / 0 |
| Phương pháp 3 | Hybrid: dense + BM25 gộp bằng RRF (k = 60, 20 candidate mỗi bên), top-3, 3000 ký tự | Hai retriever sai ở các câu khác nhau nên gộp theo thứ hạng sẽ sửa Hit@1 | dev | 12/15 (80%) | 15/15 (100%) | 15/15 (100%) | 96 ms / 0 |
| Tham chiếu | TF-IDF của starter, top-3, 3000 ký tự | Baseline lexical có sẵn | dev | 8/15 (53%) | 14/15 (93%) | 14/15 (93%) | 1,4 ms / 0 |
| Cải thiện 1 | Context budget 800 → 3000 ký tự (áp dụng cho mọi phương pháp) | Budget 800 của starter nhỏ hơn một chunk (tối đa 800 ký tự + dòng nguồn) nên context gần như rỗng | dev | không đổi | không đổi | dense 2/15 → 15/15; hybrid 2/15 → 15/15; BM25 2/15 → 14/15 | context trung bình ~340 → ~2550 ký tự |
| Cải thiện 2 | Hybrid + filter `year` do người dùng cung cấp (3 câu có filter) | q14 không nêu năm nên lẫn chunk của báo cáo sai năm | dev | 13/15 (87%); dense + filter vẫn 11/15 | 15/15 | 15/15 | 77 ms / 0 |
| Cải thiện 3 (mức agent) | Hybrid + context policy `interleave` thay cho `first_come` | Với `first_come`, 3 chunk của lượt tìm đầu chiếm hết budget nên lượt tìm sau không thêm được gì | dev | — | — | agent: 14/15 → 15/15 | 34 s/câu (agent) |
| Cấu hình đã chọn | Hybrid RRF, top-3, 3000 ký tự, không filter, `interleave` | Chốt trên dev theo tiêu chí công bố trước; chạy test một lần | test | 5/5 | 5/5 | 5/5 (retrieval một lượt); 5/5 (context cuối của agent) | 219 ms / 0 |
| Baseline | Dense, top-3, 3000 ký tự, không filter, `first_come` | Mốc so sánh trên cùng bộ test | test | 5/5 | 5/5 | 5/5; 5/5 | 202 ms / 0 |

Hybrid vừa là phương pháp thứ ba vừa là một cấu hình cải thiện so với baseline dense. Filter không nằm trong cấu hình đã chọn vì chỉ 3/20 câu có filter và chưa đo ở mức agent. Context Evidence Rate của retrieval tính trên top-3 của một lượt tìm bằng câu hỏi gốc; ở mức agent tính trên context cuối gửi model (query do model viết, có thể nhiều lượt).

**Chấm context trên test (5 câu có đáp án, 0/1/2):** baseline 5 câu mức 2; cấu hình đã chọn 5 câu mức 2 (0 câu mức 0 hoặc 1). Lý do từng câu ở trường `context_reason` trong `results/agent-test-*/reviews.json`.

**Hai lỗi dev, nguyên nhân, cách thử:**

1. **q09 — lượt tìm sau không vào được context** ("How many electric vehicle chargers are estimated to be needed by 2030?"). Với hybrid + `first_come`, agent tìm 3 lượt bằng 3 query khác nhau nhưng `context_ids` giống hệt nhau ở cả ba lượt, rồi từ chối (trace `results/agent-dev-hybrid/traces.json`). Nguyên nhân: `pack_context` của starter xếp theo thứ tự đến, 3 chunk đầu (~850 ký tự mỗi chunk) đã dùng hết 3000 ký tự. Thử: `interleave` xếp theo thứ hạng trong từng lượt, lượt mới trước. Kết quả: chunk tr.52 chứa "800,000 units by 2030" vào context ở lượt 2, agent trả lời đúng; Context Evidence Rate 14/15 → 15/15, từ chối nhầm 1/15 → 0/15, điểm trung bình 1,67 → 1,80.
2. **q14 — câu hỏi không nêu năm** ("How did the banking system's non-performing loan ratio change?"). BM25 trả 3 chunk của báo cáo 2025 (trượt Hit@3); dense tìm thấy evidence ở hạng 3, hybrid ở hạng 2. Agent dense từ chối sau 1 lượt. Thử: (a) hybrid — agent có evidence trong context và trả lời đúng; (b) filter `year: 2023` ở retrieval — Hit@1 hybrid 12/15 → 13/15.

Lỗi thứ ba chưa sửa được: **q03** — hybrid trả lời "36 percent" thay vì 71,2 vì model đọc nhầm số chú thích "36" đứng ngay trước "71.2 percent" trong chunk ("...Committee, 36 71.2 percent of surveyed enterprises..."). Dense trả lời đúng câu này dù trích cùng chunk; khác biệt nằm ở các chunk còn lại trong context. Nguyên nhân gốc là text PDF chưa lọc số chú thích (đã ghi ở phần "nếu làm lại" của Day 6).

**Tiêu chí chọn (chốt trước khi chạy test):** Context Evidence Rate của agent trên dev cao nhất; hòa thì ít từ chối nhầm hơn; hòa nữa thì điểm trả lời trung bình; cuối cùng là thời gian. Hybrid + `interleave` đứng đầu ở cả ba tiêu chí đầu (15/15; 0/15; 1,80) với thời gian tương đương.

## Agent và đánh giá

- **Tool, số lượt gọi, điều kiện dừng, cách chọn context và xử lý lỗi:**
  - Một tool `search(query)`; retriever đứng sau thay đổi theo cấu hình. Tối đa 3 lượt tìm, top-3 mỗi lượt, context budget 3000 ký tự (gồm cả dòng nguồn).
  - Lượt đầu dùng tool calling của Ollama. Từ lượt hai, model trả JSON theo schema `{passages_contain_answer, answer, next_query}`. Lý do: khi thử trên dev với tool calling thuần, `qwen2.5:7b` luôn gọi `search` đủ 3 lượt dù đoạn văn đã có đáp án (70–90 s/câu); với quyết định JSON, 15/17 câu dev dừng sau 1 lượt (~35 s/câu). Trace của lần thử tool calling thuần không được lưu lại.
  - Điều kiện dừng: model báo đoạn văn chứa đáp án → trả lời; hết lượt tìm, model không đưa query mới hoặc lặp lại query cũ → từ chối. Câu trả lời không có citation hợp lệ (ID phải nằm trong context và xuất hiện trong câu trả lời) cũng bị chuyển thành từ chối.
  - Model thấy nhãn ngắn `[1]`, `[2]`; controller đổi lại thành chunk ID thật. Nội dung đoạn văn được ghi rõ trong prompt là dữ liệu, không phải chỉ dẫn. Gold không vào agent hay retriever; chỉ dùng để chấm sau khi chạy.
  - Lỗi model hoặc vector store được ghi `status: error`, không thay bằng kết quả giả. Không có lỗi hệ thống nào trong các lượt chạy báo cáo. Nếu model không gọi tool ở lượt đầu, controller tìm bằng câu hỏi gốc và ghi `fallback` trong trace (xảy ra 1/17 câu ở dense dev, 0 ở các lượt khác).

| Cấu hình / split | Tool use | Số lượt gọi | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|
| Dense, `first_come` / dev | 17/17 | 1 lượt: 16 câu; 2 lượt: 1 | 2 / 1 / 12 — 1,67 | 13/13 | 13/13 | 2/2 | 2/15 |
| Hybrid, `first_come` / dev | 17/17 | 1: 15; 2: 1; 3: 1 | 2 / 1 / 12 — 1,67 | 13/15 | 14/14 | 2/2 | 1/15 |
| Hybrid, `interleave` / dev | 17/17 | 1: 15; 2: 1; 3: 1 | 1 / 1 / 13 — 1,80 | 14/16 | 15/15 | 2/2 | 0/15 |
| Baseline: dense, `first_come` / test | 7/7 | 1: 7 | 1 / 0 / 4 — 1,60 | 4/4 | 4/4 | 2/2 | 1/5 |
| Đã chọn: hybrid, `interleave` / test | 7/7 | 1: 6; 2: 1 | 1 / 0 / 4 — 1,60 | 4/4 | 4/4 | 2/2 | 1/5 |

Đã đọc toàn bộ 7 câu test của cả hai cấu hình (5 có đáp án, 2 thiếu bằng chứng) và toàn bộ 17 câu dev của ba cấu hình.

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| q17 (test, đã chọn) | Có đáp án | 1 lượt `search("transport modes moving passengers and freight extra power generation needed for EV charging cut")`; context có 2025-03 tr.52 | "A modal shift ... at a scale of 35 percent by 2050 could reduce the required additional generation supply by 9–11 percent ..." `[wb-taking-stock-2025-03:p0052:para800:001]` | 2; citation hợp lệ: chunk nêu đúng 9–11 percent và điều kiện 35 percent đến 2050 |
| q19 (test, đã chọn) | Có đáp án, không nêu năm | 2 lượt: "credit growth rate Viet Nam" rồi "credit growth rate in viet nam 2023-2024"; context có 2025-03 tr.31 | "Credit growth reached the government's target of 15 percent by the end of 2024" `[...2025-03:p0031:para800:000]` | 2; citation hợp lệ. Baseline dense từ chối câu này dù context có evidence (0) |
| q18 (test, đã chọn) | Có đáp án (dạng có/không, đáp án "không") | 1 lượt; context có 2023-08 tr.51 chứa "there is no regulatory requirement..." | Từ chối | 0; từ chối nhầm: model coi câu phủ định là "không chứa đáp án". Baseline dense trả lời đúng (2) |
| u03 (test, đã chọn) | Thiếu bằng chứng | 1 lượt `search("Long Thanh International Airport opening year")`; 3 chunk trả về, không chunk nào nhắc Long Thành | "Không đủ bằng chứng để trả lời." | Từ chối đúng, không thêm khẳng định |
| q03 (dev, đã chọn) | Có đáp án | 1 lượt; context có 2023-08 tr.24 | "36 percent of surveyed enterprises reported a cut of at least 5 percent in labor" `[...p0024:para800:001]` | 0; citation không hợp lệ: chunk ghi 71,2 percent, "36" là số chú thích |

- **Đường dẫn trace, nhãn, lệnh chạy, thời gian và chi phí:** mỗi lượt chạy là một thư mục trong [results/](../../results/): `traces.json` (từng tool call, query, kết quả, context, câu trả lời, citation, từng lần gọi model, token, thời gian), `summary.json`, `cases.json`, `review_sheet.json`, `reviews.json` (nhãn và lý do), `metrics.json` (tổng hợp bằng `starter/agent_metrics.py`). Lệnh chạy ở [src/README.md](../../src/README.md). Thời gian trên CPU, đo một lần:

| Lượt chạy | Median / câu | Tổng | Token prompt / output | Chi phí |
|---|---|---|---|---|
| `agent-dev-dense` (17 câu) | 36,1 s | 627 s | 29.152 / 1.367 | 0 USD |
| `agent-dev-hybrid` (17 câu) | 34,3 s | 651 s | 30.796 / 1.477 | 0 USD |
| `agent-dev-hybrid-interleave` (17 câu) | 34,1 s | 644 s | 30.864 / 1.479 | 0 USD |
| `agent-test-dense` (7 câu) | 34,6 s | 252 s | 11.540 / 561 | 0 USD |
| `agent-test-hybrid-interleave` (7 câu) | 37,9 s | 283 s | 12.420 / 591 | 0 USD |

- **Kết quả trước/sau, ví dụ lỗi, đánh đổi và giới hạn:**
  - *Trước/sau trên dev:* dense → hybrid + `interleave`: Context Evidence Rate 12/15 → 15/15, từ chối nhầm 2/15 → 0/15, điểm trả lời 1,67 → 1,80. Đổi lại citation hợp lệ giảm 13/13 → 14/16 (q03, q04).
  - *Trên test không có cải thiện:* hai cấu hình bằng nhau (1,60/2; 4/4 citation; 2/2 abstention; 1/5 từ chối nhầm), mỗi bên từ chối nhầm một câu khác nhau (dense: q19; đã chọn: q18). Cả hai lần context đều đã có evidence, nên điểm yếu trên test nằm ở bước model quyết định "đoạn văn có chứa đáp án không", không nằm ở retrieval. Với 5 câu, chênh lệch 1 câu không đủ để kết luận cấu hình nào tốt hơn.
  - *Đánh đổi:* hybrid thêm BM25 gần như không tốn thời gian (retrieval ~0,1 s so với ~35 s gọi model). Thời gian do model quyết định: mỗi lượt tìm thêm tốn khoảng 20–25 s trên CPU, nên `interleave` chỉ đáng giá ở những câu cần tìm lại. Budget 3000 ký tự (~3 chunk) tăng token prompt nhưng là mức tối thiểu để chunk 800 ký tự vào được context.
  - *Giới hạn:* mẫu nhỏ (dev 17, test 7), hai tài liệu cùng một loạt báo cáo, câu hỏi tiếng Anh. Dev và test dùng chung tài liệu. Kết quả dense trên test đã biết từ Day 6. Model 7B chạy CPU, đo thời gian một lần. Chroma dùng HNSW xấp xỉ. Nhãn chấm do một người đọc, chưa có người chấm thứ hai. Câu thiếu bằng chứng chỉ được kiểm tra bằng regex trên chunk, không đọc lại toàn bộ PDF; phần không được index (bìa, mục lục, tài liệu tham khảo) không được kiểm tra. Filter `year` chưa đo ở mức agent. Chưa thử rerank, viết lại query có kiểm soát hay model khác.
  - *Nếu làm tiếp:* lọc số chú thích khi chunk (q03); thêm bước kiểm tra lại khi model định từ chối mà còn lượt tìm (q18, q19); thử rerank để sửa Hit@1 của q02, q11.

## Thử thách nâng cao

- **Tài liệu, nguồn, quyền sử dụng, số trang và phiên bản:** World Bank Group, *Vietnam Country Climate and Development Report*, 07/2022, 100 trang PDF (trang in 1–87 là trang vật lý 11–97; tr.98–99 trống). Tải công khai từ [documents.worldbank.org](https://documents1.worldbank.org/curated/en/099355507062229876/pdf/P17724103036ee07909701028a237b6d18b.pdf), sha256 `bff12549…316f89`. Bản quyền World Bank; chỉ dùng cho bài học, không đưa PDF, ảnh trang hay bản trích xuất lên repo (tái tạo theo [src/advanced/README.md](../../src/advanced/README.md)). Tài liệu có 22 hình (biểu đồ, bản đồ, sơ đồ), 6 bảng đánh số và bảng phụ lục. Lưu ý: tài liệu vừa đúng 100 trang PDF, khoảng 87 trang nội dung.
- **Cách xử lý text, bảng, ảnh, biểu đồ; phạm vi index và phần chưa xử lý:**
  - Render 100 trang thành ảnh PNG 110 dpi (PyMuPDF). Mỗi trang gửi ảnh và text layer (pypdf) cho model đọc ảnh, nhận về JSON các phần tử `text` / `table` / `figure`: bảng chép theo hàng, cột, đơn vị; hình gồm tiêu đề, trục, chú giải, mọi số và nhãn in trên hình ("Printed values"), số ước lượng tách riêng có dấu ~, bản đồ ghi vùng được tô đậm và địa danh.
  - Model trích xuất: `gemini-3-flash-preview` 13 trang, `gemini-2.5-flash` 15 trang, `gemini-3.1-flash-lite` 72 trang. Lý do: gói miễn phí giới hạn 20 request/ngày cho mỗi model flash, script tự chuyển model khi hết hạn mức. Kết quả: 298 phần tử text, 19 bảng, 32 hình; 426 nghìn token, khoảng 40 phút.
  - Hai index trong Chroma, cùng embedding `embeddinggemma-2`: `pypdf` (text layer, 362 chunk, 96 trang có text) và `mm` (453 chunk: 402 text, 19 bảng, 32 hình). Mỗi bảng/hình là một chunk riêng, mở đầu bằng nhãn và tiêu đề; mỗi chunk giữ trang vật lý, trang in, loại phần tử, nhãn. Citation trỏ tới đúng vùng trang (ví dụ `ccdr:p051:mm-figure:00` = Figure 13, trang in 41).
  - Chưa xử lý tốt: Figure 12 (biểu đồ tròn) không có chú giải màu trong bản gốc, model tự gán tên nhiên liệu cho các phần ngoài phần than (59% khớp text). Mô tả Figure 11 (bản đồ ngập lụt) không nêu vùng đậm nhất. Ảnh trang trí đầu chương chỉ được ghi một dòng. Text trích xuất chỉ được kiểm tra trên các trang có trong benchmark (33 câu có bảng/hình đều có đủ số then chốt), không phải toàn bộ 100 trang.
- **Cách chia benchmark, schema evidence, model và cấu hình đã chốt trên dev:**
  - [data/advanced/questions.json](../../data/advanced/questions.json), sinh từ [build_questions.py](../../data/advanced/build_questions.py), chia trước khi chạy retrieval. Dev 22 câu (6 bảng, 2 ảnh, 5 biểu đồ, 2 multi-hop, 4 text, 3 thiếu bằng chứng); test 30 câu (6 bảng, 5 ảnh/sơ đồ, 5 biểu đồ, 4 multi-hop, 4 text, 6 thiếu bằng chứng). Câu dev và test không trùng đáp án.
  - Mỗi câu có nhóm chính, đáp án, `evidence_groups` (trang vật lý, trang in, phần tử như "Table 4"/"Figure 13", mô tả ô hoặc vùng), phép tính nếu có. Multi-hop có hai nhóm evidence ở hai trang khác nhau. Đáp án đọc từ ảnh trang gốc; mọi trích dẫn text đã được kiểm tra khớp nguyên văn trang. Câu thiếu bằng chứng ghi mẫu regex và số trang khớp; một câu dự kiến (doanh số xe điện) bị loại khỏi nhóm này vì tài liệu có số liệu (tr.58), chuyển thành câu dev có đáp án.
  - Agent: cùng vòng lặp với phần cơ bản (`interleave`, tối đa 3 lượt tìm), controller gọi `gemini-3.1-flash-lite` (model duy nhất còn đủ hạn mức). Kết quả dev (đạt/không đạt), file trong [results/advanced/](../../results/advanced/):

| Cấu hình dev | Đạt | Bảng | Ảnh | Biểu đồ | Multi-hop | Text | Thiếu evidence |
|---|---|---|---|---|---|---|---|
| pypdf + hybrid (text-only) | 16/22 | 6/6 | 1/2 | 2/5 | 0/2 | 4/4 | 3/3 |
| mm + hybrid | 18/22 | 6/6 | 2/2 | 3/5 | 0/2 | 4/4 | 3/3 |
| mm + hybrid + ảnh trang | 19/22 | 5/6 | 2/2 | 4/5 | 1/2 | 4/4 | 3/3 |
| mm + hybrid + ảnh trang + quy tắc nâng cao, top-k 8, budget 24.000 (chốt) | 22/22 | 6/6 | 2/2 | 5/5 | 2/2 | 4/4 | 3/3 |

  - Ba lỗi dev dẫn tới cấu hình chốt: ad14 ước lượng "khoảng 150 MtCO2e" từ Figure 4 thay vì số 109,13 trong bảng phụ lục; ad12 nêu "lệch 1,1 điểm" không có chiều; ad06 không lấy được bảng phụ lục (bảng dài ~8.000 ký tự chiếm gần hết budget 12.000). Sửa: thêm quy tắc "trả lời đủ mọi vế, ưu tiên số in trong bảng/text hơn số ước lượng từ biểu đồ, nêu chiều thay đổi", top-k 5 → 8, budget 12.000 → 24.000 ký tự. "Ảnh trang" nghĩa là ở bước quyết định, model nhận thêm ảnh của tối đa 3 trang có trong context.
  - Tiêu chí chọn (ghi trong [SELECTION.md](../../data/advanced/SELECTION.md) trước khi chạy test): tỷ lệ đạt dev cao nhất; hòa thì ít lượt gọi model hơn. Test chạy một lần, không chỉnh sau đó.

| Nhóm test | Đạt / tổng (cấu hình chốt) | Tỷ lệ | Baseline text-only | Lỗi / nhận xét |
|---|---|---|---|---|
| Bảng | 5/6 | 83,3% | 6/6 | at04: chọn nhầm hàng "competitiveness gains" (−0,6) thay vì "energy efficiency gains" (+0,2) dù trích đúng Table 4 |
| Ảnh, sơ đồ | 4/5 | 80,0% | 4/5 | at09: chọn Cần Thơ thay vì Đà Nẵng trên bản đồ Figure 11 dù có ảnh trang |
| Biểu đồ | 5/5 | 100% | 2/5 | Baseline đọc nhầm số (246 thay vì 276 TWh; 28,1 là nhãn trục) hoặc từ chối |
| Multi-hop | 3/4 | 75,0% | 1/4 | at18: vế GDP lấy −0,6% (NZP không cải cách) thay vì +1,7% (Table 4) |
| Text | 4/4 | 100% | 4/4 | |
| Thiếu bằng chứng | 6/6 | 100% | 6/6 | ut03 ở ranh giới (xem dưới) |
| **Toàn bộ** | **27/30** | **90,0%** | **23/30 (76,7%)** | 0 lỗi hệ thống, 0 timeout |

- **Trace, nhãn và evidence từng câu; thời gian, chi phí, lỗi hệ thống:** `results/advanced/test-final/` gồm `traces.json` (từng lượt gọi model, query, chunk trả về, context, trang ảnh đã gửi, câu trả lời, citation, token, thời gian), `review_sheet.json` (câu hỏi, đáp án, evidence, câu trả lời, nội dung từng citation), `reviews.json` (đạt/không đạt và lý do từng câu), `score.json`. Cấu hình chốt: 67 lượt gọi model, 255.501 token prompt (phần lớn là ảnh trang) / 3.191 token output, median 7,7 s/câu, tổng 789 s (gồm thời gian chờ khi API báo 429/503 và tự thử lại). Baseline: 79 lượt gọi, 167.181 / 3.591 token, median 3,3 s, tổng 287 s. Chi phí 0 USD (gói miễn phí). Không có lỗi hệ thống trong hai lượt test; các lỗi 429 (hạn mức theo phút) và 503 (quá tải) đều được thử lại thành công.
- **Giới hạn, cách tái lập và đường dẫn benchmark:**
  - *Kết quả nằm đúng ngưỡng:* 27/30 = 90,0%, và có 4 câu được chấm "đạt" theo cách nới (ghi rõ trong `reviews.json`): at08 chỉ nêu "ven biển miền Trung" trong khi dải tô màu kéo từ ven biển phía Bắc; at17 và at19 nêu đủ hai giá trị nhưng không nêu tỷ lệ; ut03 không dùng câu từ chối mà bác bỏ tiền đề bằng evidence ("hệ thống trading chưa triển khai", tr.22), không đưa ra giá nào. Nếu người chấm đánh "không đạt" bất kỳ câu nào trong 4 câu này, kết quả là 26/30 và không đạt ngưỡng.
  - Mẫu nhỏ (dev 22, test 30) trên một tài liệu; benchmark do một người soạn và chấm, chưa có người chấm thứ hai. Cấu hình chốt đạt 22/22 trên dev nhưng 27/30 trên test, cho thấy đã khớp dev một phần. Model trích xuất khác nhau giữa các trang. Model agent là bản "lite" vì giới hạn hạn mức miễn phí. Câu hỏi bằng tiếng Anh.
  - Tái lập: [src/advanced/README.md](../../src/advanced/README.md). Benchmark: [data/advanced/questions.json](../../data/advanced/questions.json).
