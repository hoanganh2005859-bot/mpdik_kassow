# Hướng dẫn upload và chạy trên Kaggle — Kassow KR810 Tier 0-4

Bộ phát hành này gồm hai artifact chính:

- `KR810_Tier0_Tier4_Kaggle_Dataset_v1.0.0.zip` — toàn bộ source code + benchmark + trajectory
  cần thiết để chạy pipeline (không gồm `notebooks/`).
- `KR810_Tier0_Tier4_Kaggle_Template_v1.0.0.ipynb` — notebook Kaggle để chạy pipeline và hiển
  thị kết quả.

Có hai cách để đưa Dataset lên Kaggle. Cách A (Dataset đã giải nén) được khuyến nghị vì đơn giản
hơn khi kiểm tra nội dung trên giao diện Kaggle; Cách B (một file ZIP) tiện hơn khi upload thủ
công nhanh. Notebook hỗ trợ cả hai mà không cần chỉnh sửa gì thêm.

## CÁCH A — KHUYẾN NGHỊ: Dataset đã giải nén

1. Giải nén `KR810_Tier0_Tier4_Kaggle_Dataset_v1.0.0.zip` trên máy của bạn.
2. Trên Kaggle, tạo **New Dataset**.
3. Upload toàn bộ nội dung đã giải nén (không upload thêm một thư mục cha bao ngoài) --
   `DATASET_MANIFEST.json` phải nằm ngay ở gốc (root) của Dataset.
4. Tạo notebook mới (hoặc import) và dùng nội dung của
   `KR810_Tier0_Tier4_Kaggle_Template_v1.0.0.ipynb`.
5. **Attach Dataset** vừa tạo vào notebook (Add Input).
6. Chỉnh Cell 2 (User Configuration) theo phần "Cấu hình Cell 2" bên dưới.
7. Chạy **smoke** trước (mặc định trong Cell 2).
8. Sau khi smoke pass, đổi sang **full** và chạy lại.
9. Tải file ZIP kết quả từ `/kaggle/working` (Output tab của Kaggle).

## CÁCH B — DÙNG MỘT FILE ZIP

1. Trên Kaggle, tạo **New Dataset**.
2. Upload trực tiếp file `KR810_Tier0_Tier4_Kaggle_Dataset_v1.0.0.zip` (không giải nén trước).
3. Attach Dataset này vào notebook (Add Input) — notebook dùng cùng
   `KR810_Tier0_Tier4_Kaggle_Template_v1.0.0.ipynb` như Cách A.
4. Notebook (Cell 4 "Locate dataset") sẽ **tự động phát hiện** ZIP dưới
   `/kaggle/input/<dataset>/*.zip`, kiểm tra nội dung hợp lệ, rồi **giải nén an toàn** (chống
   path traversal, absolute path, symlink, và giới hạn kích thước/số file) vào
   `/kaggle/working/_kr810_dataset_extracted`. Bạn không cần tự giải nén.
5. Chạy **smoke** trước.
6. Sau khi pass, chạy **full**.

Nếu Kaggle báo có nhiều hơn một Dataset/ZIP hợp lệ được attach cùng lúc, notebook sẽ liệt kê các
candidate và yêu cầu bạn đặt `DATASET_ROOT_OVERRIDE` hoặc `DATASET_ZIP_OVERRIDE` trong Cell 2.

## Cấu hình Cell 2

Chạy smoke trước:

```python
PRESET = "smoke"
RUN_NAME = "kr810_smoke"
OVERWRITE = True
RESUME = False
```

Sau khi smoke pass, đổi sang full:

```python
PRESET = "full"
RUN_NAME = "kr810_full"
```

Lưu ý:

- **Không bật `OVERWRITE = True` và `RESUME = True` cùng lúc** -- notebook sẽ báo lỗi validation
  ở Cell 2 nếu cả hai đều True.
- CPU (không cần GPU) là đủ để chạy Tier 0-4 -- pipeline không dùng và không yêu cầu GPU.
- Chạy `full` tốn nhiều thời gian và CPU hơn `smoke` nhiều lần (toàn bộ 1200 điểm IK, 8
  trajectory, cả hai method warm/cold-start); cân nhắc dùng `RESUME = True` cho các lần chạy
  tiếp theo nếu phiên Kaggle bị ngắt giữa chừng.
- File ZIP kết quả (`KR810_Tier0_Tier4_<RUN_NAME>.zip`) được tạo trong `/kaggle/working`, không
  phải trong Dataset input (input luôn chỉ đọc).
- Các ngưỡng chấp nhận (position/orientation) trong notebook là **tiêu chí tự định nghĩa của dự
  án** (`configs/evaluation_config.json`), **không phải chứng nhận ISO 9283**.
- Đây là **kinematic evaluation** (Tier 0-4: FK/Jacobian/DLS và các chỉ số bám quỹ đạo/khả thi
  khớp), **không có dynamics, không có PPO/MPDIK/MAPPO**.
