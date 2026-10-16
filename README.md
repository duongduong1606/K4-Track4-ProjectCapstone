# K4 Track 4 Project Capstone — Vin Perception Hackathon

Trang đề bài hackathon cuối khóa Track 4 (Computer Vision & Robotics): 2 giờ build, 10 phút demo mỗi team, đề tài gắn với hệ sinh thái Vin.

- `index.html`: trang đề bài (một file, không cần build). Mở trực tiếp bằng trình duyệt.
- `DATA_AUDIT.md`: kết quả kiểm tra, làm sạch và lựa chọn dữ liệu hư hỏng ngoại thất.
- `TRAINING.md`: môi trường và lệnh train cho stack YOLO26n + YOLO26n-seg + Qwen3-VL.
- `demo_app/`: giao diện FastAPI local để tải ảnh, chạy hai model và xem chẩn đoán.
- `REPORT.md`: báo cáo kỹ thuật, kết quả và liên kết đến bằng chứng kiểm thử.
- `reports/evidence/`: metric test, manifest SHA-256 và hướng dẫn tái lập bằng chứng.

Nội dung gồm hướng đề tài gợi ý, bốn yêu cầu của mỗi demo, bảy trụ cột Track 4, công cụ gợi ý, thang điểm, lịch trình, checklist trước demo và FAQ. Ngày giờ, địa điểm, giám khảo, giải thưởng sẽ được ban tổ chức cập nhật.

## Chạy demo local

Đặt hai model đã train tại `runs/damage/weights/best.pt` và `runs/panel/weights/best.pt`, sau đó chạy:

```powershell
conda env create -f environment.yml   # chỉ cần chạy lần đầu
conda activate vinfast-damage
uvicorn demo_app.main:app --host 127.0.0.1 --port 8000
```

Mở <http://127.0.0.1:8000>, tải ảnh ngoại thất lên và bấm **Phân tích hư hỏng**. Model weights không được lưu trên Git; xem thêm cách train và tái tạo model trong [TRAINING.md](TRAINING.md).
