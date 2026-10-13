# TÀI LIỆU LÝ THUYẾT & NỘI DUNG SLIDE THUYẾT TRÌNH (MODULE 1)
**Người phụ trách:** TV1 (Trưởng nhóm)  
**Chủ đề:** Tiền xử lý, Đạo hàm ảnh & Thuật toán Dò biên Canny (Canny Edge Detection)

---

## PHẦN 1: CƠ SỞ TOÁN HỌC & LÝ THUYẾT CỐT LÕI

### 1. Khử nhiễu bằng bộ lọc Gaussian (Gaussian Blur)
* **Bản chất toán học:** Tích chập ảnh xám $I(x, y)$ với ma trận nhân Gauss 2 chiều (Gaussian Kernel):
  $$G(x, y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2 + y^2}{2\sigma^2}}$$
* **Mục đích:** 
  * Đạo hàm rất nhạy cảm với nhiễu tần số cao (nhiễu hạt, vân bề mặt).
  * Lọc Gaussian đóng vai trò là bộ lọc thông thấp (Low-pass filter), làm mượt ảnh và triệt tiêu nhiễu trước khi tính đạo hàm.

---

### 2. Đạo hàm ảnh & Gradient không gian (Sobel Operator)
* **Biên (Edge) là gì:** Biên là các vị trí có sự thay đổi đột ngột về cường độ sáng (Intensity discontinuity).
* **Đạo hàm bậc 1:**
  * Theo trục ngang $X$: $G_x = \frac{\partial I}{\partial x} \approx I * S_x$
  * Theo trục dọc $Y$: $G_y = \frac{\partial I}{\partial y} \approx I * S_y$
  * Kernel Sobel $3 \times 3$:
    $$S_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}, \quad S_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix}$$
* **Độ lớn Gradient (Gradient Magnitude):**
  $$G = \sqrt{G_x^2 + G_y^2} \quad \text{hoặc xấp xỉ nhanh:} \quad |G_x| + |G_y|$$
* **Hướng của Gradient (Gradient Direction):**
  $$\theta = \arctan\left(\frac{G_y}{G_x}\right)$$
  *(Hướng Gradient luôn vuông góc với tiếp tuyến của đường biên).*

---

### 3. Thuật toán Canny Edge Detection (4 Bước chuẩn)
Thuật toán do John F. Canny đề xuất năm 1986, tối ưu hóa 3 tiêu chí: (1) Tỷ lệ lỗi thấp, (2) Định vị biên chính xác, (3) Đáp ứng đơn điểm (một biên chỉ có 1 vạch).

1. **Bước 1: Làm mịn ảnh (Gaussian Smoothing):** Khử nhiễu.
2. **Bước 2: Tìm Gradient cường độ:** Tính $G$ và $\theta$ bằng Sobel.
3. **Bước 3: Triệt tiêu không cực đại (Non-Maximum Suppression - NMS):**
   * Làm mỏng đường biên dày thành đường biên mảnh đúng 1 pixel.
   * Làm tròn hướng gradient $\theta$ về 4 hướng chính: $0^\circ, 45^\circ, 90^\circ, 135^\circ$.
   * So sánh độ lớn pixel hiện tại với 2 pixel lân cận theo hướng gradient:
     * Nếu pixel hiện tại là cực đại cục bộ $\rightarrow$ Giữ lại.
     * Ngược lại $\rightarrow$ Đặt về 0 (loại bỏ).
4. **Bước 4: Phân ngưỡng Hysteresis (Ngưỡng kép):**
   * Sử dụng 2 ngưỡng: $T_{\text{low}}$ và $T_{\text{high}}$ (thường tỉ lệ $1:2$ hoặc $1:3$).
   * $G \ge T_{\text{high}}$: Biên mạnh (Strong Edge) $\rightarrow$ Giữ lại chắc chắn.
   * $G < T_{\text{low}}$: Nhiễu $\rightarrow$ Loại bỏ hoàn toàn.
   * $T_{\text{low}} \le G < T_{\text{high}}$: Biên yếu (Weak Edge) $\rightarrow$ Chỉ giữ lại nếu nó có liên kết kết nối (8-connectivity) với một pixel biên mạnh.

---

### 4. Nối liền biên mép giấy bằng Toán tử hình thái học (Morphological Operations)
* Trong thực tế chụp mép giấy, ánh sáng loang hoặc bóng đổ làm Canny bị đứt đoạn.
* **Giải pháp của TV1:**
  * **Dilation (Giãn nở):** $A \oplus B$ làm dày đường biên và kết nối các khoảng đứt gãy nhỏ.
  * **Closing (Phép Đóng = Dilation rồi Erosion):** Hàn gắn các khe hở và làm mượt biên viền trước khi chuyển sang Module 2 tìm Contour.

---

## PHẦN 2: BỘ CÂU HỎI VẤN ĐÁP DÀNH RIÊNG CHO TV1

| STT | Câu hỏi của Giảng viên | Câu trả lời chuẩn xác |
| :---: | :--- | :--- |
| **1** | *Tại sao phải làm mịn Gauss trước khi chạy Canny?* | Vì toán tử đạo hàm bản chất là bộ lọc thông cao, khuếch đại nhiễu hạt rất mạnh. Lọc Gauss giúp khử nhiễu trước để tránh sinh ra các biên giả do nhiễu. |
| **2** | *Bản chất của Non-Maximum Suppression (NMS) trong Canny là gì?* | Là kỹ thuật làm mỏng biên (Edge Thinning). Nó chỉ giữ lại các điểm có độ lớn gradient đạt cực đại cục bộ theo phương vuông góc với đường biên, triệt tiêu các điểm lân cận về 0 để biên chỉ có độ dày 1 pixel. |
| **3** | *Tại sao Canny lại dùng 2 ngưỡng (Hysteresis) thay vì 1 ngưỡng đơn như thông thường?* | Ngưỡng đơn dễ gặp tình trạng: nếu chọn cao thì biên bị đứt khúc, nếu chọn thấp thì nhiễu xuất hiện tràn lan. Ngưỡng kép vừa loại được nhiễu ($< T_{\text{low}}$), vừa truy vết giữ lại được các đoạn mép mờ nhạt nếu chúng nối liền với biên mạnh. |
| **4** | *Hàm `auto_canny` trong code của bạn hoạt động trên nguyên lý nào?* | Hàm dựa trên giá trị trung vị (Median) của ảnh xám: $T_{\text{low}} = \max(0, (1 - \sigma) \times \text{median})$ và $T_{\text{high}} = \min(255, (1 + \sigma) \times \text{median})$ với $\sigma = 0.33$. Nó tự động thích ứng theo ảnh sáng hay tối mà không cần người dùng nhập tay. |

---

## PHẦN 3: DÀN Ý SLIDE THUYẾT TRÌNH (MẪU 5 SLIDE CHO TV1)

* **Slide 1: Giới thiệu Module 1:** Tiền xử lý ảnh và Tầm quan trọng của dò biên trong bài toán quét tài liệu.
* **Slide 2: Khử nhiễu & Đạo hàm không gian:** Công thức bộ lọc Gauss và ma trận Sobel tính độ lớn/hướng gradient.
* **Slide 3: Thuật toán Canny:** Sơ đồ khối 4 bước, cơ chế làm mảnh NMS và phân ngưỡng Hysteresis.
* **Slide 4: Cải tiến thực tế:** Thuật toán Auto-Canny theo Median và phép đóng hình thái học (Morphological Closing) để chống đứt mép giấy.
* **Slide 5: Kết quả thực nghiệm:** Hình ảnh so sánh ảnh gốc, ảnh biên Canny thô và ảnh biên hoàn thiện sau xử lý.
