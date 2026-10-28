# TÀI LIỆU LÝ THUYẾT & NỘI DUNG SLIDE THUYẾT TRÌNH (MODULE 2)
**Người phụ trách:** TV2 (nTrường) & Trưởng nhóm TV1 (Đức)  
**Chủ đề:** Trích xuất 4 góc đa giác & Biến đổi phối cảnh (Perspective Transformation / Homography)

---

## PHẦN 1: CƠ SỞ TOÁN HỌC & LÝ THUYẾT CỐT LÕI

### 1. Bản chất hình học của Biến đổi phối cảnh (Perspective Transformation)
Khi chụp tài liệu từ góc nghiêng, khoảng cách từ các phần khác nhau của tờ giấy tới cảm biến máy ảnh là không đồng đều. Các đường thẳng song song trong không gian thực (ví dụ 2 mép giấy) sẽ hội tụ về một điểm ảo (Vanishing Point) trong ảnh 2D. Hiện tượng này làm biến dạng tỷ lệ khung hình và chữ viết.

Phép chiếu phối cảnh phẳng (Planar Homography) là một ánh xạ xạ ảnh (Projective Mapping) từ mặt phẳng đối tượng sang mặt phẳng cảm biến:
$$\begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} \sim \mathbf{H} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix} = \begin{bmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & h_{33} \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$

* Ma trận $\mathbf{H}$ có kích thước $3 \times 3$ gồm 9 phần tử.
* Do tính chất tỷ lệ đồng nhất (Scale Invariant trong hệ tọa độ thuần nhất - Homogeneous Coordinates), ta có thể chuẩn hóa bằng cách chia cho $h_{33} = 1$. Vì vậy, ma trận $\mathbf{H}$ có đúng **8 bậc tự do (8 Degrees of Freedom - DOF)**.
* Mỗi cặp điểm tương ứng $(x, y) \leftrightarrow (x', y')$ cung cấp 2 phương trình đại số độc lập:
  $$x' = \frac{h_{11}x + h_{12}y + h_{13}}{h_{31}x + h_{32}y + 1}, \quad y' = \frac{h_{21}x + h_{22}y + h_{23}}{h_{31}x + h_{32}y + 1}$$
* Để giải hệ phương trình 8 ẩn số này, ta cần tối thiểu **4 cặp điểm không thẳng hàng** (chính là 4 góc của tờ giấy tài liệu: Top-Left, Top-Right, Bottom-Right, Bottom-Left).
* Trong OpenCV, hàm `cv2.getPerspectiveTransform(src, dst)` sử dụng phương pháp phân tích giá trị kỳ dị (SVD - Singular Value Decomposition) để giải hệ phương trình tuyến tính thuần nhất trên.

---

### 2. Warping xuôi (Forward Warping) vs. Warping ngược (Inverse Warping)

| Tiêu chí so sánh | Warping xuôi (Forward Warping) | Warping ngược (Inverse Warping) |
| :--- | :--- | :--- |
| **Cách tiếp cận** | Duyệt qua từng pixel nguồn $(x, y)$, dùng ma trận $\mathbf{H}$ để tính tọa độ đích $(x', y')$. | Duyệt qua từng pixel của ảnh đích $(u, v)$, dùng ma trận nghịch đảo $\mathbf{H}^{-1}$ để tìm tọa độ nguồn $(x, y)$. |
| **Hiện tượng lỗ hổng (Holes/Gaps)** | **Rất dễ xảy ra:** Do phép biến dạng kéo giãn, các pixel nguồn sau khi làm tròn số nguyên có thể không phủ kín toàn bộ lưới ảnh đích, tạo nên các chấm đen li ti không có giá trị màu. | **Tuyệt đối không xảy ra:** Mọi pixel $(u, v)$ trên ảnh đích đều được gán giá trị nội suy từ ảnh nguồn. |
| **Hiện tượng chồng lấn (Overlaps)** | Nhiều pixel nguồn có thể rơi vào cùng 1 tọa độ đích, gây lãng phí tính toán. | Không xảy ra. |
| **Ứng dụng thực tế** | Hiếm khi dùng trong xử lý ảnh thực tế. | **Là chuẩn mực** trong OpenCV (`cv2.warpPerspective`) và đồ họa máy tính. |

---

### 3. Phép nội suy song tuyến (Bilinear Interpolation)
Khi áp dụng ma trận nghịch đảo $\mathbf{H}^{-1}$, tọa độ nguồn $(x, y)$ tìm được là các số thực (Floating-point) không trùng khít với lưới pixel nguyên:
$$x = i + \alpha, \quad y = j + \beta \quad (0 \le \alpha, \beta < 1)$$

Nội suy song tuyến tính giá trị màu $I(x, y)$ bằng cách bình quân gia quyền dựa trên khoảng cách tới 4 pixel nguyên lân cận: $(i, j), (i+1, j), (i, j+1), (i+1, j+1)$:
$$I(x, y) = (1 - \alpha)(1 - \beta) I(i, j) + \alpha(1 - \beta) I(i+1, j) + (1 - \alpha)\beta I(i, j+1) + \alpha\beta I(i+1, j+1)$$

* **Ưu điểm:** Loại bỏ hiện tượng răng cưa (Jaggies/Aliasing), giúp văn bản sau khi nắn phẳng giữ được nét chữ mượt mà, không bị vỡ hạt.

---

### 4. Thuật toán Sắp xếp 4 góc chuẩn chiều kim đồng hồ (`order_points`)
Trước khi tính Homography, 4 điểm góc phải được sắp xếp theo đúng thứ tự:
1. `rect[0]`: Top-Left (Góc trên - trái)
2. `rect[1]`: Top-Right (Góc trên - phải)
3. `rect[2]`: Bottom-Right (Góc dưới - phải)
4. `rect[3]`: Bottom-Left (Góc dưới - trái)

* **Toán học áp dụng:**
  * **Tổng tọa độ $(x + y)$:**
    * Điểm Top-Left gần gốc tọa độ $(0, 0)$ nhất $\rightarrow$ Có tổng $(x + y)$ **nhỏ nhất**.
    * Điểm Bottom-Right xa gốc tọa độ nhất $\rightarrow$ Có tổng $(x + y)$ **lớn nhất**.
  * **Hiệu tọa độ $(y - x)$:**
    * Điểm Top-Right có hoành độ $x$ lớn, tung độ $y$ nhỏ $\rightarrow$ Có hiệu $(y - x)$ **nhỏ nhất** (hoặc $(x - y)$ lớn nhất).
    * Điểm Bottom-Left có hoành độ $x$ nhỏ, tung độ $y$ lớn $\rightarrow$ Có hiệu $(y - x)$ **lớn nhất** (hoặc $(x - y)$ nhỏ nhất).

---

### 5. Thuật toán Ramer-Douglas-Peucker xấp xỉ đa giác (`cv2.approxPolyDP`)
* **Nguyên lý:** Đơn giản hóa đường gấp khúc với số lượng đỉnh nhỏ nhất mà khoảng cách Hausdorff lớn nhất giữa đường bao gốc và đa giác xấp xỉ không vượt quá ngưỡng sai số $\varepsilon$ (Epsilon):
  $$\varepsilon = \text{factor} \times \text{chu vi} (\text{Arc Length})$$
* Trong bài toán tài liệu: Ta quét dải $\text{factor} \in [0.015, 0.08]$ để tìm đa giác lồi 4 đỉnh (`len(approx) == 4` và `cv2.isContourConvex`).

---

## PHẦN 2: BỘ CÂU HỎI VẤN ĐÁP DÀNH RIÊNG CHO MODULE 2 (TV2 & NHÓM)

| STT | Câu hỏi của Giảng viên | Câu trả lời chuẩn xác |
| :---: | :--- | :--- |
| **1** | *Tại sao ma trận Homography có kích thước 3x3 (9 phần tử) nhưng lại chỉ có 8 bậc tự do?* | Vì trong hệ tọa độ thuần nhất, vector tọa độ $[sx, sy, s]^T$ tỷ lệ với $[x, y, 1]^T$ bất kể hệ số tỷ lệ $s \ne 0$. Ta có thể chia tất cả các phần tử cho $h_{33}$ để cố định $h_{33} = 1$, do đó chỉ còn lại 8 biến số độc lập. |
| **2** | *Tại sao phải dùng ít nhất 4 điểm để nắn thẳng tài liệu mà không phải 3 điểm như phép biến đổi Affine?* | Biến đổi Affine (6 bậc tự do) bảo toàn tính song song, chỉ xử lý được tịnh tiến, quay, co giãn và trượt (shear) nên chỉ cần 3 điểm. Biến đổi phối cảnh (Perspective/Homography) có 8 bậc tự do và không bảo toàn tính song song (2 mép giấy nghiêng hội tụ vào điểm ảo), mỗi điểm cho 2 phương trình nên cần tối thiểu 4 điểm $(4 \times 2 = 8)$ để giải hệ phương trình. |
| **3** | *Tại sao hàm `warpPerspective` của OpenCV lại áp dụng Warping ngược thay vì Warping xuôi?* | Nếu dùng Warping xuôi, các pixel nguồn khi nhân với $H$ sẽ ra tọa độ thực, khi làm tròn sang số nguyên sẽ có pixel bị trùng nhau và có ô trống không có pixel nào rơi vào, gây ra các lỗ thủng đen (Holes) trên ảnh đích. Warping ngược duyệt qua từng pixel của ảnh phẳng đích, dùng $H^{-1}$ chiếu ngược về ảnh nguồn rồi nội suy màu song tuyến, đảm bảo ảnh kết quả liên tục 100% không tì vết. |
| **4** | *Nếu tờ giấy bị cong nhẹ ở mép hoặc bị ngón tay che một góc khiến `approxPolyDP` không ra 4 đỉnh thì hệ thống xử lý thế nào?* | Nhóm xây dựng cơ chế Fallback vững chắc 3 tầng: Tầng 1 quét dải epsilon từ 0.015 đến 0.08; Tầng 2 dùng Convex Hull để loại bỏ vết lồi lõm do tay che; Tầng 3 dùng `cv2.minAreaRect` để trích xuất 4 góc của hình chữ nhật xoay bao phủ diện tích nhỏ nhất của tài liệu. Kết hợp phân ngưỡng Otsu giúp tỷ lệ thành công đạt 100% trên toàn bộ tập test. |
| **5** | *Tại sao nhóm lại thu nhỏ ảnh (Resize) khi tìm góc rồi mới nhân tỷ lệ ánh xạ ngược về ảnh gốc?* | Có 2 lý do kỹ thuật quan trọng: (1) Tối ưu tốc độ xử lý gấp nhiều lần (chỉ mất ~8.5 ms/ảnh thay vì hàng trăm ms); (2) Thu nhỏ ảnh đóng vai trò như một bộ lọc thông thấp tự nhiên, làm triệt tiêu các nét chữ li ti và nhiễu viền bên trong tài liệu, giúp đường viền mép giấy nổi bật và dễ phát hiện hơn rất nhiều. |

---

## PHẦN 3: DÀN Ý SLIDE THUYẾT TRÌNH (MẪU 5 SLIDE CHO TV2)

* **Slide 1: Giới thiệu Module 2:** Mục tiêu nắn thẳng tài liệu từ góc chụp nghiêng về bản scan phẳng trực diện chuẩn kích thước.
* **Slide 2: Trích xuất 4 góc:** Tìm Contour lớn nhất, giải thuật xấp xỉ đa giác Ramer-Douglas-Peucker và thuật toán sắp xếp 4 góc chuẩn chiều kim đồng hồ.
* **Slide 3: Cơ sở toán học Homography:** Ma trận biến đổi phối cảnh 8 bậc tự do, hệ phương trình xác định từ 4 cặp điểm ảnh.
* **Slide 4: Kỹ thuật Warping ngược & Nội suy song tuyến:** So sánh cơ chế Forward vs Inverse Warping, công thức toán học nội suy màu chống răng cưa.
* **Slide 5: Kết quả thực nghiệm Tuần 2:** Biểu đồ tỷ lệ phát hiện 100% (27/27 ảnh), thời gian xử lý siêu tốc 8.5 ms (~117 FPS), hình ảnh minh chứng trước và sau khi nắn phẳng.
