# Báo cáo MVP kiểm tra hư hỏng ngoại thất xe

Ngày chốt bằng chứng: **16/10/2026**

## 1. Bài toán và phạm vi

MVP hỗ trợ bước tiếp nhận xe bằng cách nhận một ảnh ngoại thất, phát hiện hư hỏng và gắn hư hỏng với bộ phận xe tương ứng. Đầu ra gồm ảnh đã đánh dấu, loại hư hỏng, độ tin cậy, bộ phận, mức độ ước lượng và khuyến nghị kiểm tra.

Đây là công cụ **hỗ trợ kỹ thuật viên**, không thay thế kết luận sửa chữa. Mức độ hư hỏng hiện được suy ra bằng quy tắc diện tích vùng phát hiện, chưa phải nhãn được chuyên gia thẩm định.

## 2. Kiến trúc chạy local

```text
Ảnh tải lên
    ├─ YOLO26n detect: dent / scratch / broken_or_cracked
    └─ YOLO26n-seg: 21 bộ phận ngoại thất
             ↓
 Ghép box hư hỏng với mask có độ phủ lớn nhất
             ↓
 FastAPI + giao diện web: ảnh đánh dấu và phiếu chẩn đoán
```

- Damage model: `runs/damage/weights/best.pt`.
- Panel model: `runs/panel/weights/best.pt`.
- Ứng dụng không cần Internet và đã chạy trên NVIDIA RTX 3060 12 GB.
- Qwen3-VL 2B qua Ollama đã được chuẩn bị cho phần diễn giải mở rộng, nhưng không nằm trên đường suy luận chính của bản demo để giảm độ trễ và tránh phụ thuộc dịch vụ.

## 3. Dữ liệu

| Nhóm dữ liệu | Mục đích | Số ảnh | Chia tập |
|---|---|---:|---|
| AutoDamageIQ chọn lọc | Bổ sung damage detection | 5.633 | Đưa vào bộ damage chung |
| Jijo đã làm sạch | Dent/scratch detection | 2.159 | 1.727 / 216 / 216 |
| **Damage tổng** | 3 nhóm hư hỏng | **7.792** | **6.233 / 780 / 779** |
| CarDamagePanel | 21 lớp panel segmentation | **560** | **415 / 103 / 42** |

Bộ dữ liệu đã được kiểm tra ảnh lỗi, nhãn YOLO, trùng chính xác và gần trùng. Năm bản sao rò rỉ ở test của CarDamagePanel đã bị loại cùng nhãn; sau làm sạch không còn trùng được xác nhận giữa các split. Chi tiết nguồn, giấy phép và phân bố lớp nằm trong [DATA_AUDIT.md](DATA_AUDIT.md).

## 4. Cấu hình và thời gian train

| Model | Kích thước ảnh | Batch | Giới hạn epoch | Early stopping | Epoch tốt nhất | Tổng thời gian |
|---|---:|---:|---:|---:|---:|---:|
| YOLO26n damage | 640 | 16 | 60 | 12 | 44 | 52,5 phút |
| YOLO26n-seg panel | 640 | 8 | 100 | 20 | 70 | 14,7 phút |

Hai model được train tuần tự bằng Python 3.11, PyTorch 2.6.0 + CUDA 12.4. Tổng thời gian ghi nhận khoảng **67,2 phút**. Cấu hình tái lập và lệnh chạy nằm trong [TRAINING.md](TRAINING.md).

## 5. Kết quả định lượng

### 5.1 Validation trong quá trình train

| Model | Precision | Recall | mAP50 | mAP50–95 |
|---|---:|---:|---:|---:|
| Damage box, epoch 44 | 0,513 | 0,403 | 0,392 | 0,233 |
| Panel mask, epoch 70 | 0,869 | 0,848 | 0,884 | 0,671 |

### 5.2 Đánh giá lại trên test giữ riêng

| Model | Precision | Recall | mAP50 | mAP50–95 |
|---|---:|---:|---:|---:|
| Damage box, 779 ảnh / 1.952 instance | 0,504 | 0,379 | 0,383 | 0,222 |
| Panel mask, 42 ảnh / 801 instance | 0,750 | 0,389 | 0,416 | 0,274 |

Damage model giữ kết quả khá sát validation. Panel model giảm mạnh trên test, đặc biệt ở recall; điều này cho thấy split nhỏ và khác biệt miền ảnh, nên không được dùng metric validation cao như bằng chứng duy nhất về khả năng tổng quát hóa.

## 6. Kiểm thử demo trên 10 ảnh ngoài tập train

Ngưỡng confidence dùng để tạo bằng chứng là **0,20**. Hệ thống trả về **8 vùng phát hiện trên 6/10 ảnh**.

| Case | Kích thước | Kết quả mô hình |
|---:|---:|---|
| 01 | 1640×922 | Móp @ nắp capo, 0,257 |
| 02 | 600×400 | Trầy xước @ cản trước, 0,744 |
| 03 | 415×739 | 2 vùng trầy xước @ bánh sau, 0,259 và 0,209 |
| 04 | 640×480 | Trầy xước, chưa xác định panel, 0,295 |
| 05 | 424×724 | Không phát hiện |
| 06 | 415×739 | Móp @ cản sau, 0,308 |
| 07 | 479×640 | Không phát hiện |
| 08 | 1200×848 | Không phát hiện |
| 09 | 1280×960 | Không phát hiện |
| 10 | 800×1067 | Trầy xước @ cản sau, 0,757; móp @ vè sau, 0,633 |

Mười ảnh này chưa có ground truth do người gán nhãn, vì vậy bảng trên là **bằng chứng hệ thống chạy end-to-end**, không phải phép đo accuracy. Mỗi ảnh được nhận diện bằng SHA-256 trong manifest để kết quả có thể đối chiếu chính xác mà không phải đưa ảnh người dùng lên Git.

## 7. Gói bằng chứng và tái lập

Các bằng chứng có thể đưa lên repository:

- [reports/evidence/results.json](reports/evidence/results.json): cấu hình, metric train/test và kết quả chi tiết từng ảnh.
- [reports/evidence/manifest.csv](reports/evidence/manifest.csv): bảng kiểm toán gọn gồm tên, kích thước, SHA-256 và phát hiện.
- [reports/evidence/README.md](reports/evidence/README.md): giải thích phạm vi và cách kiểm chứng.
- [scripts/generate_evidence.py](scripts/generate_evidence.py): chạy lại toàn bộ phép đánh giá.

Tái tạo bằng chứng:

```powershell
conda activate vinfast-damage
python scripts/generate_evidence.py
```

Chạy demo:

```powershell
uvicorn demo_app.main:app --host 127.0.0.1 --port 8000
```

## 8. Giới hạn và bước tiếp theo

- Recall damage test 0,379 còn thấp; mô hình có thể bỏ sót lỗi nhỏ, phản sáng hoặc góc chụp khó.
- Panel test chỉ có 42 ảnh và chênh lệch lớn so với validation; cần bổ sung ảnh cùng quy trình tiếp nhận thực tế.
- Dữ liệu không được huấn luyện riêng cho từng dòng VinFast, nên cần fine-tune bằng ảnh được cho phép sử dụng tại xưởng.
- Một số lớp panel chưa phân biệt trái/phải; việc ghép theo độ phủ có thể nhầm ở vùng chồng lấn.
- Severity hiện là heuristic. Bản sau nên có nhãn kích thước thực, độ sâu và xác nhận của kỹ thuật viên.
- Với sản phẩm thực tế, nên thêm hướng dẫn chụp 6–8 góc cố định, kiểm tra ảnh mờ và cơ chế “không chắc chắn — cần chụp lại”.

## 9. Kết luận

MVP đã hoàn thành chuỗi local từ tải ảnh, chạy hai model, ghép damage–panel đến sinh phiếu chẩn đoán. Kết quả đủ để trình diễn ý tưởng và quy trình kỹ thuật trong thời gian ngắn; số liệu test cũng chỉ ra rõ phần cần ưu tiên tiếp theo là tăng recall damage và khả năng tổng quát hóa của panel segmentation.
