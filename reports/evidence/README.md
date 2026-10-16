# Evidence package

Thư mục này chứa bằng chứng có thể tái lập cho báo cáo MVP.

## File được lưu trên Git

- `results.json`: nguồn dữ liệu đầy đủ, gồm cấu hình inference, epoch tốt nhất, metric trên test và kết quả từng test case.
- `manifest.csv`: bản tóm tắt dễ đọc; SHA-256 liên kết mỗi kết quả với đúng file đầu vào.

## File chỉ lưu local

- `annotated/`: ảnh đã vẽ box và nhãn dự đoán.
- `ultralytics/`: output do Ultralytics tạo khi đánh giá model.
- `../../test_cases/`: ảnh đầu vào do người dùng cung cấp.

Ba nhóm trên được `.gitignore` loại khỏi Git để không tải dữ liệu, ảnh người dùng hoặc artifact sinh tự động lên repository. Model weights và toàn bộ dataset cũng chỉ lưu local.

## Tạo lại bằng chứng

Từ thư mục gốc của project:

```powershell
conda activate vinfast-damage
python scripts/generate_evidence.py
```

Chỉ chạy lại 10 ảnh demo, bỏ qua đánh giá dataset:

```powershell
python scripts/generate_evidence.py --skip-dataset-eval
```

Ngưỡng mặc định là `0.20`; có thể đổi bằng `--confidence`. Việc thay ngưỡng sẽ làm thay đổi kết quả và phải được ghi rõ khi so sánh.

## Cách diễn giải

- Metric trong `evaluation` được tính trực tiếp trên split `test` bằng `best.pt`.
- Mười ảnh trong `test_cases` không có ground truth được kiểm duyệt, nên chỉ chứng minh pipeline chạy end-to-end; không dùng chúng để tuyên bố accuracy.
- `annotated_local` trong JSON là đường dẫn tương đối đến ảnh bằng chứng local. Ảnh này có thể được mở khi demo nhưng không được commit.
