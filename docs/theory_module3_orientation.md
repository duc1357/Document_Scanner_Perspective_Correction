# TÀI LIỆU LÝ THUYẾT & NỘI DUNG SLIDE THUYẾT TRÌNH (MODULE 3)
**Người phụ trách:** TV3 (Pháp)  
**Chủ đề:** Hiệu chỉnh hướng văn bản tự động (ORB Feature Detector, Đối sánh Hamming & Ước lượng RANSAC)

---

## PHẦN 1: CƠ SỞ TOÁN HỌC & LÝ THUYẾT CỐT LÕI

### 1. Thuật toán phát hiện đặc trưng ORB (Oriented FAST and Rotated BRIEF)
ORB được Ethan Rublee và cộng sự đề xuất năm 2011 như một giải pháp thay thế mã nguồn mở, hiệu năng cao và miễn phí bản quyền cho SIFT và SURF.
ORB kết hợp 2 kỹ thuật chính:
1. **Phát hiện điểm đặc trưng (Keypoint Detection - oFAST):**
   * Sử dụng thuật toán FAST (Features from Accelerated Segment Test) để tìm điểm góc nhanh chóng bằng cách so sánh cường độ sáng của pixel trung tâm $p$ với 16 pixel trên đường tròn bán kính 3 quanh nó.
   * Để đạt tính bất biến với phép xoay (Rotation Invariance), ORB tính hướng của điểm đặc trưng thông qua trọng tâm cường độ sáng (Intensity Centroid):
     * Các mô-men cường độ (Moments):
       $$m_{pq} = \sum_{x, y} x^p y^q I(x, y)$$
     * Tọa độ trọng tâm góc cục bộ:
       $$C = \left( \frac{m_{10}}{m_{00}}, \frac{m_{01}}{m_{00}} \right)$$
     * Góc định hướng $\theta$ của keypoint:
       $$\theta = \arctan2(m_{01}, m_{10})$$

2. **Mô tả điểm đặc trưng (Descriptor - rBRIEF):**
   * BRIEF biểu diễn đặc trưng dưới dạng chuỗi bit nhị phân (Binary String) thông qua so sánh cường độ sáng giữa các cặp điểm:
     $$\tau(p; x, y) = \begin{cases} 1 & \text{nếu } I(x) < I(y) \\ 0 & \text{ngược lại} \end{cases}$$
   * Vector mô tả gồm $n$ bit (thường là 256 bits = 32 bytes):
     $$f_n(p) = \sum_{1 \le i \le n} 2^{i-1} \tau(p; x_i, y_i)$$
   * rBRIEF xoay ma trận tọa độ kiểm thử theo góc $\theta$ đã tính ở bước oFAST để đảm bảo vector đặc trưng bất biến khi ảnh bị xoay.

---

### 2. Đối sánh đặc trưng bằng Khoảng cách Hamming (Hamming Distance Matcher)
* Với các bộ mô tả nhị phân như ORB, khoảng cách giữa hai vector đặc trưng $A$ và $B$ không đo bằng khoảng cách Euclid $\ell_2$ mà đo bằng **khoảng cách Hamming**:
  $$D_{\text{Hamming}}(A, B) = \text{popcount}(A \oplus B)$$
  *(Đếm số bit khác nhau thông qua phép XOR bitwise).*
* **Tối ưu tốc độ phần cứng:** Phép toán XOR và đếm bit (`POPCNT`) được vi xử lý hiện đại (CPU) hỗ trợ ở cấp độ lệnh hợp ngữ máy, cho tốc độ đối sánh nhanh gấp hàng chục lần so với khoảng cách thực của SIFT/SURF.
* **Lọc tương quan bằng Lowe's Ratio Test (KNN Matcher với $k=2$):**
  $$D(m_1) < \text{ratio} \times D(m_2) \quad (\text{với ratio } \approx 0.75)$$
  * $m_1$: Điểm khớp gần nhất (khoảng cách nhỏ nhất).
  * $m_2$: Điểm khớp nhì.
  * Nếu tỷ lệ này nhỏ hơn ngưỡng, điểm khớp được coi là tin cậy và giữ lại; ngược lại bị loại bỏ do đặc trưng lặp lại hoặc mơ hồ.

---

### 3. Thuật toán RANSAC ước lượng Ma trận Affine (Random Sample Consensus)
* **Vấn đề Outliers:** Sau bước đối sánh, luôn tồn tại các cặp điểm khớp sai (Mismatch / Outliers) do cấu trúc lặp lại của các ký tự chữ cái trong văn bản.
* **Mô hình Affine cục bộ:**
  $$\begin{bmatrix} x' \\ y' \end{bmatrix} = \begin{bmatrix} a & -b \\ b & a \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} + \begin{bmatrix} t_x \\ t_y \end{bmatrix}$$
  * Trong đó ma trận biến đổi gồm phép xoay $\theta$, tỷ lệ $s$ và dịch chuyển $(t_x, t_y)$.
  * Góc xoay được trích xuất từ phần tử ma trận: $\theta = \arctan2(M_{10}, M_{00})$.
* **Các bước của RANSAC:**
  1. Chọn ngẫu nhiên tập con tối thiểu các cặp điểm (2 cặp điểm cho Partial Affine 2D).
  2. Tính toán ma trận mô hình ứng viên.
  3. Áp dụng ma trận cho toàn bộ các cặp điểm còn lại và đếm số lượng điểm phù hợp (Inliers) có sai số tái chiếu $e < T_{\text{threshold}}$.
  4. Lặp lại $N$ lần, chọn mô hình có số Inliers lớn nhất và tinh chỉnh lại bằng bình phương tối thiểu (Least Squares).

---

### 4. Cơ chế Deskew độc lập (Standalone Deskew) & Rotate Fallback
* Khi không có ảnh mẫu đối sánh (Reference Image):
  * Sử dụng phân ngưỡng nhị phân Otsu để tách biệt các nét chữ.
  * Tìm tập hợp tọa độ các pixel nét chữ và xác định hình chữ nhật xoay diện tích tối thiểu `cv2.minAreaRect(coords)`.
  * Trích xuất góc nghiêng trục dòng văn bản và xoay ảnh bù trừ (`cv2.getRotationMatrix2D` + `cv2.warpAffine`).
* **Rotate Fallback:** Kiểm tra tỷ lệ khung hình $(W > H)$. Vì tài liệu chuẩn (A4, hóa đơn, bài tập) luôn có chiều dọc lớn hơn chiều ngang, nếu $W > H$ hệ thống tự động xoay $90^\circ$ theo chiều kim đồng hồ để đưa văn bản về hướng đứng chuẩn đọc.

---

## PHẦN 2: BỘ CÂU HỎI VẤN ĐÁP DÀNH CHO TV3

| STT | Câu hỏi vấn đáp | Gợi ý trả lời cốt lõi |
| :---: | :--- | :--- |
| **1** | *Tại sao lại chọn ORB thay vì SIFT hoặc SURF cho bài toán này?* | ORB hoàn toàn miễn phí bản quyền (SIFT từng có bằng sáng chế), sử dụng vector nhị phân 256-bit kết hợp khoảng cách Hamming tính bằng lệnh XOR phần cứng nên tốc độ nhanh gấp 10-100 lần, cực kỳ phù hợp cho hệ thống quét thời gian thực (Real-time Scanner). |
| **2** | *Khoảng cách Hamming khác khoảng cách Euclidean thế nào?* | Khoảng cách Euclidean tính căn bậc hai tổng bình phương sai lệch giữa các số thực. Khoảng cách Hamming tính số lượng bit khác nhau giữa hai chuỗi nhị phân bằng phép `XOR` và đếm bit `1`, tiêu tốn rất ít chu kỳ CPU. |
| **3** | *RANSAC đóng vai trò gì trong việc tính góc xoay của văn bản?* | Các ký tự văn bản có tính lặp lại rất cao (nhiều chữ 'e', 'a', 'o' giống nhau) tạo ra rất nhiều điểm đối sánh sai (Outliers). RANSAC giúp loại bỏ các điểm nhiễu này và chỉ giữ lại các Inliers thống nhất theo cùng một phép biến đổi hình học chuẩn xác. |
| **4** | *Khi người dùng quét một tài liệu mới hoàn toàn không có ảnh mẫu đối chứng thì Module 3 làm thế nào?* | Hệ thống kích hoạt cơ chế Standalone Deskew: dùng `minAreaRect` trên tọa độ các pixel nét chữ sau nhị phân để khử góc nghiêng dòng chữ, kết hợp kiểm tra tỷ lệ khung hình $W/H$ để tự động xoay đứng tài liệu nếu tài liệu bị đặt nằm ngang. |
