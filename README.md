# Document Scanner & Perspective Correction (Xử lý ảnh tài liệu & Nắn thẳng góc nhìn)

Dự án nghiên cứu và phát triển hệ thống quét tài liệu tự động bằng các thuật toán Thị giác máy tính (Computer Vision), giải quyết bài toán chụp tài liệu bị nghiêng, bóng đổ, sai hướng và mất nét.

---

## 📁 Cấu trúc thư mục dự án

```text
Document_Scanner_Perspective_Correction/
├── modules/                         # Các module thuật toán lõi
│   ├── __init__.py                  # Package exports
│   ├── edge_detector.py             # Module 1: Tiền xử lý & Dò biên (Gaussian + Canny + Morphology)
│   ├── perspective_warp.py          # Module 2: Trích xuất 4 góc & Nắn phẳng (Homography Transform)
│   ├── orientation_corrector.py     # Module 3: Tự động xoay thẳng văn bản (ORB + Hamming + RANSAC)
│   └── image_enhancer.py            # Module 4: Tăng cường chất lượng ảnh, nhị phân hóa & làm nét
├── tools/                           # Scripts kiểm thử và benchmark tự động
│   ├── test_real_image.py           # Thử nghiệm Module 1 trên ảnh thực tế
│   ├── test_pipeline_week2.py       # Benchmark tích hợp Module 1 & 2 (100% trên 27 ảnh)
│   ├── test_module3_orientation.py  # Thử nghiệm đối sánh xoay ORB RANSAC & Deskew
│   ├── week2_experiments.py         # So sánh các bộ lọc nhị phân (Otsu, Adaptive, CLAHE)
│   ├── week3_sharpening.py          # Thử nghiệm làm nét văn bản (Sharpening)
│   └── generate_sample_image.py     # Sinh ảnh mẫu tổng hợp
├── docs/                            # Tài liệu lý thuyết chi tiết & câu hỏi vấn đáp
│   ├── theory_module1_canny.md      # Lý thuyết đạo hàm ảnh & thuật toán Canny (TV1)
│   ├── theory_module2_warp.md       # Lý thuyết ma trận Homography & Warping ngược (TV2)
│   ├── theory_module3_orientation.md# Lý thuyết ORB, Hamming & giải thuật RANSAC (TV3)
│   └── theory_module4_enhancer.md   # Lý thuyết cân bằng ánh sáng, nhị phân hóa (TV4)
├── dataset/
│   ├── raw/                         # Bộ dữ liệu 27 ảnh chụp thực nghiệm (TV5)
│   └── output/                      # Kết quả trực quan qua các tuần thực nghiệm
├── main_app.py                      # Giao diện ứng dụng chính (Giai đoạn 2)
├── requirements.txt                 # Danh sách thư viện phụ thuộc
└── README.md
```

---

## ⚙️ Cài đặt môi trường

Yêu cầu Python >= 3.8:

```bash
pip install -r requirements.txt
```

---

## 🚀 Hướng dẫn chạy thử nghiệm các Module

### 1. Kiểm thử Module 1 & 2 (Dò biên & Nắn thẳng phối cảnh)
```bash
python tools/test_pipeline_week2.py
```
*Tỷ lệ thành công: 27/27 ảnh (100%), thời gian trung bình ~12 ms/ảnh.*

### 2. Kiểm thử Module 3 (Xoay thẳng văn bản tự động)
```bash
python tools/test_module3_orientation.py
```

### 3. Kiểm thử Module 4 (So sánh bộ lọc nhị phân & Làm nét)
```bash
python tools/week2_experiments.py
python tools/week3_sharpening.py
```

---

## 👥 Phân công nhiệm vụ (Giai đoạn 1)

- **TV1 (Trưởng nhóm):** Nghiên cứu & phát triển Module 1 (Dò biên Canny, lọc Gauss, hình thái học).
- **TV2:** Nghiên cứu & phát triển Module 2 (Trích xuất 4 góc đa giác, ma trận Homography, Warping ngược).
- **TV3:** Nghiên cứu & phát triển Module 3 (ORB Feature Detector, Hamming Matcher, ước lượng xoay RANSAC, Deskew).
- **TV4:** Nghiên cứu & phát triển Module 4 (Lọc nhị phân Otsu, Adaptive Gaussian, CLAHE, Sharpening nét chữ).
- **TV5:** Thu thập & xây dựng bộ dữ liệu thực nghiệm 27 ảnh đa dạng chủ đề và góc nghiêng.