# Code Day 7 — cách chạy

Mọi lệnh chạy từ thư mục gốc của repo, Python 3.11+ (đã chạy với 3.13.5 trên Windows 11). Trên Windows luôn thêm `-X utf8`.

## 1. Chuẩn bị

| Thành phần | Giá trị đã dùng |
|---|---|
| Dữ liệu | 2 báo cáo World Bank *Taking Stock* (08/2023 và 03/2025), 88 trang nội dung, 370 chunk (`by_paragraph`, tối đa 800 ký tự) |
| Vector store | Chroma `PersistentClient` của Day 6, collection `day6_B`, cosine, HNSW |
| Embedding | `google/embeddinggemma-2`, 768 chiều, chuẩn hóa L2, query prefix `task: search result \| query: `, chạy local bằng `sentence-transformers` |
| Model agent | `qwen2.5:7b` qua Ollama 0.40.16, `temperature=0`, `seed=7`, `num_ctx=8192`, chạy CPU (`num_gpu=0`) |

1. Tạo lại index Day 6 theo README của repo `K5-L3A-DAY06-NguyenVanVinh-2A202603672-DataIngestion` (tải 2 PDF theo `data/manifest.json`, chạy `python -m src.run`). PDF và index không được đưa lên Git vì bản quyền thuộc World Bank.
2. Đặt repo Day 6 cạnh repo này, hoặc đặt biến môi trường `DAY6_DIR` (thư mục repo Day 6) hay `DAY6_CHROMA_PATH` (thư mục `chroma_db`).
3. Cài thư viện: `pip install -r requirements.txt` (cùng bộ thư viện với Day 6; có thể dùng lại venv Day 6).
4. Cài [Ollama](https://ollama.com) và tải model: `ollama pull qwen2.5:7b`. Nếu GPU của bạn chạy được Ollama, thêm `--num-gpu -1` khi chạy agent.

## 2. Xuất dữ liệu Day 6 sang schema Day 7

```bash
python -X utf8 src/export_day6.py
```

Tạo `starter/data/day7/chunks.json` (370 chunk, không đưa lên Git) và cập nhật 20 câu có đáp án trong `data/questions.json`; các câu thiếu bằng chứng viết tay (`u01`–`u04`) được giữ nguyên. Script dừng nếu có evidence không nằm trọn trong một chunk đúng nguồn và trang.

## 3. Đo retrieval

```bash
python -X utf8 src/eval_retrieval.py --method hybrid --split dev --budget 3000 --output starter/results/dev-hybrid-b3000
```

`--method`: `tfidf` (baseline của starter), `bm25`, `dense`, `hybrid` (dense + BM25, RRF). Thêm `--use-filter` để dùng filter `year` do người dùng cung cấp trong câu hỏi. Kết quả: `summary.json` và `traces.json`.

## 4. Chạy agent

```bash
python -X utf8 src/run_agent.py --method hybrid --context-policy interleave --split dev --output starter/results/agent-dev-hybrid-interleave
```

Tham số: `--top-k` (3), `--budget` (3000 ký tự), `--max-searches` (3), `--context-policy` (`first_come` hoặc `interleave`), `--model`, `--num-gpu`, `--ids` (chỉ chạy một số câu và gộp vào kết quả cũ cùng cấu hình). Kết quả: `traces.json`, `cases.json`, `summary.json` và `review_sheet.json` (phiếu để người chấm điền nhãn).

Script ghi vào `starter/results/` (không được Git theo dõi). Các lượt chạy dùng trong báo cáo đã được sao chép nguyên vào [results/](../results/) để nộp; mỗi lượt agent có thêm `reviews.json` (nhãn người chấm, kèm lý do) và `metrics.json`.

## 5. Tổng hợp nhãn đã chấm

```bash
python -X utf8 starter/agent_metrics.py --cases results/agent-test-hybrid-interleave/cases.json --traces results/agent-test-hybrid-interleave/traces.json --reviews results/agent-test-hybrid-interleave/reviews.json --split test --output results/agent-test-hybrid-interleave/metrics.json
```

## 6. Test code

```bash
python -X utf8 -m unittest discover -s tests -v
```

Test dùng mock, chỉ kiểm tra code, không phải benchmark.

## File

| File | Nội dung |
|---|---|
| `common.py` | Đường dẫn, đọc/ghi JSON, định nghĩa hit (so khớp sau khi gộp khoảng trắng và bỏ phân biệt hoa thường, giống Day 6) |
| `export_day6.py` | Xuất chunk từ Chroma và chuyển câu hỏi Day 6 |
| `dense.py` | Dense retrieval trên Chroma Day 6 |
| `lexical.py` | BM25 (k1 = 1,5; b = 0,75), thư viện chuẩn |
| `hybrid.py` | Gộp dense và BM25 bằng Reciprocal Rank Fusion (k = 60, 20 candidate mỗi bên) |
| `eval_retrieval.py` | Hit@1, Evidence Hit@3, Context Evidence Rate |
| `controller.py` | Controller gọi model qua Ollama: lượt đầu tool calling `search`, các lượt sau quyết định dạng JSON |
| `agent.py` | Vòng lặp agent, mở rộng từ `starter/agentic.py` với `context_policy` |
| `run_agent.py` | Chạy agent trên một split và lưu trace |
