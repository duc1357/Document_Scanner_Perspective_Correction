# TÀI LIỆU LÝ THUYẾT & NỘI DUNG SLIDE THUYẾT TRÌNH (MODULE 4)
**Người phụ trách: TV4**
**Chủ đề**: Tăng cường chất lượng ảnh tài liệu (Image Enhancement)

----
## PHẦN 1: CƠ SỞ TOÁN HỌC & LÝ THUYẾT CỐT LÕI
### 1. Cân bằng ánh sáng (Illumination Correction)
- Bản chất: Ảnh chụp tài liệu thường bị bóng đổ hoặc ánh sáng không đồng đều, khiến chữ khó đọc.
- Phương pháp: Sử dụng Gaussian Blur để ước lượng nền sáng, sau đó điều chỉnh độ sáng của ảnh.
* Công thức:
\[
B(x,y)=G_\sigma * I(x,y)
\]

\[
I_c(x,y)=\frac{I(x,y)}{\max(B(x,y),1)}\cdot\overline{B}
\]

* Trong đó:
- \(I(x,y)\): Ảnh xám đầu vào.
- \(B(x,y)\): Nền sáng ước lượng.
- \(I_c(x,y)\): Ảnh sau cân bằng ánh sáng.
* Mục đích: Giảm bóng đổ, cân bằng độ sáng và làm rõ nội dung văn bản.
### 2. Làm nét văn bản (Image Sharpening)
- Bản chất: Tăng độ tương phản tại các đường viền để nét chữ rõ hơn.
- Phương pháp: Sử dụng kernel tích chập để tăng sự khác biệt giữa pixel trung tâm và các pixel lân cận.
* Kernel làm nét:
\[
K=
\begin{bmatrix}
0 & -\alpha & 0\\
-\alpha & 1+4\alpha & -\alpha\\
0 & -\alpha & 0
\end{bmatrix}
\]

* Công thức:
\[
I_s(x,y)=I_c(x,y)*K
\]

* Trong đó \(\alpha\) là cường độ làm nét.
* Mục đích: Tăng độ rõ của nét chữ trước khi chuyển ảnh sang đen trắng.
### 3. Nhị phân hóa thích nghi (Adaptive Thresholding)
- Bản chất: Chuyển ảnh xám thành ảnh hai mức đen và trắng.
- Vấn đề: Ngưỡng cố định khó xử lý ảnh có vùng sáng và tối khác nhau.
- Giải pháp: Sử dụng cv2.adaptiveThreshold() để tính ngưỡng riêng cho từng vùng ảnh.
* Công thức:
\[
T(x,y)=\operatorname{GaussianMean}(x,y)-C
\]

\[
I_b(x,y)=
\begin{cases}
255, & I(x,y)>T(x,y)\\
0, & I(x,y)\leq T(x,y)
\end{cases}
\]

* Tham số chính:
- block_size = 31: Kích thước vùng lân cận, phải là số lẻ lớn hơn 1.
- C = 10: Hằng số điều chỉnh ngưỡng.
* Ưu điểm: Phù hợp với ảnh có ánh sáng không đồng đều, giúp tách chữ khỏi nền giấy.
### 4. Quy trình xử lý của Module 4
1. Illumination Correction: Cân bằng ánh sáng.
2. Image Sharpening: Làm nét văn bản.
3. Adaptive Thresholding: Tạo ảnh đen trắng.
Đầu ra gồm ba ảnh: ảnh cân bằng sáng, ảnh làm nét và ảnh nhị phân.
* Vai trò trong hệ thống: Module 4 nhận ảnh sau khi Module 2 nắn thẳng và Module 3 điều chỉnh hướng, sau đó tăng cường chất lượng để dễ đọc và phục vụ xuất tài liệu.

## PHẦN 2: BỘ CÂU HỎI VẤN ĐÁP DÀNH RIÊNG CHO TV4
STT	Câu hỏi	Câu trả lời
1	Module 4 giải quyết vấn đề gì?	Cải thiện chất lượng ảnh tài liệu bằng cách cân bằng ánh sáng, làm nét và nhị phân hóa.
2	Tại sao cần cân bằng ánh sáng?	Để giảm ảnh hưởng của bóng đổ và độ sáng không đồng đều.
3	Gaussian Blur có tác dụng gì?	Làm mượt ảnh và ước lượng nền sáng để hiệu chỉnh ánh sáng.
4	Làm nét ảnh hoạt động như thế nào?	Dùng kernel tích chập để tăng độ tương phản tại đường viền chữ.
5	Adaptive Threshold là gì?	Phương pháp nhị phân hóa sử dụng ngưỡng cục bộ thay vì một ngưỡng cố định cho toàn ảnh.
6	block_size có tác dụng gì?	Xác định kích thước vùng lân cận dùng để tính ngưỡng.
7	Tham số C dùng để làm gì?	Điều chỉnh ngưỡng cục bộ bằng cách trừ một hằng số.
8	Tại sao không dùng ngưỡng cố định?	Vì độ sáng của ảnh có thể thay đổi giữa các vùng.
9	Module 4 trả về ba ảnh để làm gì?	Để lựa chọn ảnh phù hợp cho hiển thị hoặc xuất PDF.
10	Hạn chế của Module 4 là gì?	Có thể làm nổi nhiễu, mất nét chữ mảnh hoặc chưa xử lý tốt bóng đổ phức tạp.
## PHẦN 3: DÀN Ý SLIDE THUYẾT TRÌNH (5 SLIDE)
* **Slide 1 – Giới thiệu Module 4:** Mục tiêu tăng cường chất lượng ảnh và ba kỹ thuật chính.
* **Slide 2 – Cân bằng ánh sáng:** Gaussian Blur, ước lượng nền sáng và công thức hiệu chỉnh.
* **Slide 3 – Làm nét và nhị phân hóa:** Kernel Sharpening, Adaptive Threshold và các tham số.
* **Slide 4 – Pipeline xử lý:** Minh họa ba bước xử lý và vai trò của từng hàm trong code.
* **Slide 5 – Kết quả thực nghiệm:** So sánh ảnh gốc, ảnh cân bằng sáng, ảnh làm nét và ảnh nhị phân.
