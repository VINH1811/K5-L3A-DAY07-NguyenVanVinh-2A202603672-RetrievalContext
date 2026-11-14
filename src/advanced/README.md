# Thử thách nâng cao — cách chạy

Chạy từ thư mục gốc repo trên Windows, luôn thêm `-X utf8`. Cần:

- Python có `pypdf`, `chromadb`, `sentence-transformers` (venv của Day 6 dùng được) và một venv nhỏ có `pymupdf==1.26.3` để render trang.
- API key Gemini trong file `.env` ở thư mục gốc: `GEMINI_API_KEY=...` (file đã được gitignore, không commit). Gói miễn phí giới hạn số request mỗi ngày cho từng model; script tự chuyển model khi hết hạn mức ngày.

## 1. Tài liệu

Tải PDF về `starter/data/advanced/vn-ccdr-2022.pdf` (thư mục không được Git theo dõi):

```bash
curl -L -o starter/data/advanced/vn-ccdr-2022.pdf "https://documents1.worldbank.org/curated/en/099355507062229876/pdf/P17724103036ee07909701028a237b6d18b.pdf"
```

sha256 mong đợi: `bff12549ea7f81fdf8c2d5e0df5a418485c1fd2734e131ad902de89969316f89`.

## 2. Render trang và text layer

```bash
python src/advanced/render_pages.py --pdf starter/data/advanced/vn-ccdr-2022.pdf --out starter/data/advanced/pages --dpi 110
```

Text layer: lưu `pypdf` text từng trang vào `starter/data/advanced/pypdf_pages.json` (mảng `{page, text}`), ví dụ:

```bash
python -X utf8 -c "import pypdf,json;r=pypdf.PdfReader('starter/data/advanced/vn-ccdr-2022.pdf');json.dump([{'page':i,'text':p.extract_text() or ''} for i,p in enumerate(r.pages,1)],open('starter/data/advanced/pypdf_pages.json','w',encoding='utf-8'),ensure_ascii=False)"
```

## 3. Trích xuất bảng, hình, text bằng model đọc ảnh

```bash
python -X utf8 src/advanced/extract.py --pages starter/data/advanced/pages --pypdf starter/data/advanced/pypdf_pages.json --out starter/data/advanced/extracted --model gemini-2.5-flash gemini-3.1-flash-lite gemini-2.5-flash-lite
```

Mỗi trang lưu một JSON, chạy lại sẽ bỏ qua trang đã có. Lần chạy báo cáo dùng `gemini-3-flash-preview` (13 trang), `gemini-2.5-flash` (15), `gemini-3.1-flash-lite` (72); thống kê ở `results/advanced/index_stats.json`.

## 4. Build index

```bash
python -X utf8 src/advanced/build_index.py
```

Tạo `chunks_pypdf.json`, `chunks_mm.json` và hai collection Chroma `adv_pypdf`, `adv_mm` trong `starter/data/advanced/chroma`.

## 5. Đo retrieval và chạy agent

```bash
python -X utf8 src/advanced/run_advanced.py --chunks mm --method hybrid --split dev --retrieval-only --output starter/results/adv-ret-dev-mm-hybrid
```

```bash
python -X utf8 src/advanced/run_advanced.py --chunks mm --method hybrid --split test --page-images --model gemini-3.1-flash-lite --output starter/results/adv-test-final
```

Cấu hình chốt: `--chunks mm --method hybrid --page-images`, mặc định `--top-k 8 --budget 24000 --max-searches 3`, context policy `interleave` (xem [data/advanced/SELECTION.md](../../data/advanced/SELECTION.md)). Baseline: `--chunks pypdf --method hybrid`.

## 6. Chấm và tổng hợp

Người chấm điền `reviews.json` (`{query_id, pass, reason}` cho mọi câu) trong thư mục lượt chạy, đối chiếu `review_sheet.json` với ảnh trang gốc, rồi:

```bash
python -X utf8 src/advanced/score.py --run results/advanced/test-final
```

## File

| File | Nội dung |
|---|---|
| `gemini.py` | Client Gemini (thư viện chuẩn), thử lại khi 429/503, dừng ngay khi hết hạn mức ngày |
| `render_pages.py` | Render trang PDF thành PNG |
| `extract.py` | Trích xuất text/bảng/hình từng trang thành JSON |
| `build_index.py` | Tạo chunk `pypdf` và `mm`, embed và lưu Chroma |
| `run_advanced.py` | Đo retrieval hoặc chạy agent (controller Gemini, tùy chọn gửi ảnh trang) |
| `score.py` | Tỷ lệ đạt theo nhóm từ nhãn của người chấm |
