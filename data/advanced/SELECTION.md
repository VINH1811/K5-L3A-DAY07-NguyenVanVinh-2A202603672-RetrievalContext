# Cấu hình chốt cho test (ghi trước khi chạy test, 2026-11-15)

Tiêu chí: tỷ lệ đạt cao nhất trên dev (22 câu, chấm đạt/không đạt như ADVANCED.md); hòa thì ít lượt gọi model hơn.

| Lượt dev | Đạt |
|---|---|
| pypdf + hybrid, không ảnh | 16/22 |
| mm + hybrid, không ảnh | 18/22 |
| mm + hybrid + ảnh trang | 19/22 |
| mm + hybrid + ảnh trang + quy tắc nâng cao, top-k 8, budget 24000 (v2) | 22/22 |

Chốt: `--chunks mm --method hybrid --page-images --top-k 8 --budget 24000 --max-searches 3`, model `gemini-3.1-flash-lite`, context policy `interleave`.
Baseline chạy cùng test để so sánh: `--chunks pypdf --method hybrid` (không ảnh), cùng quy tắc, top-k, budget và model.
Test chỉ chạy một lần; không chỉnh gì sau khi xem kết quả test.
