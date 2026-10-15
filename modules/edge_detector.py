"""
Module 1: Tiền xử lý & Dò biên (Edge Detection)
Thành viên phụ trách: TV1 (Trưởng nhóm)
Mục tiêu:
    - Chuyển đổi ảnh sang ảnh xám (Grayscale).
    - Khử nhiễu bằng bộ lọc Gaussian Blur.
    - Cài đặt thuật toán dò biên Canny với ngưỡng tùy chỉnh linh hoạt và chế độ Auto-Canny.
    - Xử lý hình thái học (Morphological Operations) để kết nối mép tài liệu bị đứt đoạn.
"""

from typing import Tuple, Optional
import cv2
import numpy as np


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Chuyển đổi ảnh màu BGR sang ảnh mức xám (Grayscale).
    Nếu ảnh đã là ảnh xám (1 kênh), trả về bản sao của ảnh.

    Args:
        image (np.ndarray): Mảng numpy của ảnh đầu vào.

    Returns:
        np.ndarray: Ảnh mức xám (1 kênh màu, uint8).
    """
    if image is None:
        raise ValueError("Ảnh đầu vào là None. Vui lòng kiểm tra đường dẫn hoặc dữ liệu ảnh.")

    if len(image.shape) == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif len(image.shape) == 3 and image.shape[2] == 4:
        # Hỗ trợ ảnh có kênh Alpha (BGRA)
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    elif len(image.shape) == 2:
        return image.copy()
    else:
        raise ValueError(f"Định dạng kênh màu không hợp lệ: shape={image.shape}")


def apply_gaussian_blur(
    image: np.ndarray,
    kernel_size: Tuple[int, int] = (5, 5),
    sigma_x: float = 0.0
) -> np.ndarray:
    """
    Khử nhiễu bằng bộ lọc Gaussian Blur.
    Toán học: Tích chập ảnh với ma trận trọng số Gauss 2 chiều để làm mượt các chi tiết tần số cao (nhiễu hạt).

    Args:
        image (np.ndarray): Ảnh đầu vào (ảnh xám hoặc ảnh màu).
        kernel_size (Tuple[int, int]): Kích thước cửa sổ lọc Gauss (phải là các số nguyên dương lẻ, ví dụ (3, 3), (5, 5)).
        sigma_x (float): Độ lệch chuẩn theo trục X. Nếu = 0, OpenCV tự tính theo kernel_size: sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8.

    Returns:
        np.ndarray: Ảnh đã được làm mịn khử nhiễu.
    """
    # Đảm bảo kernel_size là các số lẻ
    kw, kh = kernel_size
    kw = kw if kw % 2 != 0 else kw + 1
    kh = kh if kh % 2 != 0 else kh + 1

    return cv2.GaussianBlur(image, (kw, kh), sigmaX=sigma_x)


def auto_canny(image: np.ndarray, sigma: float = 0.33) -> np.ndarray:
    """
    Tự động tính toán hai ngưỡng low_threshold và high_threshold của Canny dựa trên giá trị trung vị (Median) của ảnh.
    Công thức:
        lower = max(0, (1.0 - sigma) * median)
        upper = min(255, (1.0 + sigma) * median)

    Args:
        image (np.ndarray): Ảnh xám đã làm mịn.
        sigma (float): Hệ số co giãn ngưỡng quanh trung vị (thường chọn 0.33).

    Returns:
        np.ndarray: Ảnh nhị phân chứa biên Canny.
    """
    gray = to_grayscale(image)
    median_val = np.median(gray)

    low_threshold = int(max(0, (1.0 - sigma) * median_val))
    high_threshold = int(min(255, (1.0 + sigma) * median_val))

    return cv2.Canny(gray, low_threshold, high_threshold)


def detect_canny_edges(
    image: np.ndarray,
    low_threshold: int = 50,
    high_threshold: int = 150,
    aperture_size: int = 3,
    l2_gradient: bool = False
) -> np.ndarray:
    """
    Thực hiện dò biên Canny với các ngưỡng tùy chỉnh linh hoạt.
    Các bước cốt lõi của Canny:
        1. Tính đạo hàm không gian bằng toán tử Sobel (Gx, Gy).
        2. Triệt tiêu không cực đại (Non-Maximum Suppression - NMS) để làm mảnh biên thành 1 pixel.
        3. Phân ngưỡng Hysteresis (Ngưỡng kép) với low_threshold và high_threshold để giữ lại biên mạnh và biên yếu liên tục.

    Args:
        image (np.ndarray): Ảnh đầu vào (xám hoặc màu).
        low_threshold (int): Ngưỡng dưới để xác định biên yếu tiềm năng.
        high_threshold (int): Ngưỡng trên để xác định biên chắc chắn.
        aperture_size (int): Kích thước kernel Sobel tính gradient (3, 5, hoặc 7).
        l2_gradient (bool): Nếu True, tính độ lớn gradient chính xác bằng sqrt(Gx^2 + Gy^2),
                           nếu False, xấp xỉ nhanh bằng |Gx| + |Gy|.

    Returns:
        np.ndarray: Ảnh nhị phân các cạnh biên (255 là biên, 0 là nền).
    """
    gray = to_grayscale(image)
    return cv2.Canny(
        gray,
        threshold1=low_threshold,
        threshold2=high_threshold,
        apertureSize=aperture_size,
        L2gradient=l2_gradient
    )


def connect_edges(
    edges: np.ndarray,
    kernel_size: Tuple[int, int] = (5, 5),
    iterations: int = 1
) -> np.ndarray:
    """
    Sử dụng toán tử hình thái học (Morphological Operations) để hàn gắn các đường biên bị đứt gãy.
    Quy trình:
        - Giãn nở (Dilation) để làm dày biên và kết nối các đoạn đứt ngắn.
        - Phép Đóng (Closing = Dilate -> Erode) để lấp các lỗ thủng nhỏ dọc mép giấy.

    Args:
        edges (np.ndarray): Ảnh biên nhị phân thu được từ Canny.
        kernel_size (Tuple[int, int]): Kích thước cấu trúc phần tử (Structuring Element).
        iterations (int): Số lần lặp giãn nở.

    Returns:
        np.ndarray: Ảnh biên nhị phân đã được làm liền mạch.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    dilated = cv2.dilate(edges, kernel, iterations=iterations)
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
    return closed


def preprocess_and_detect_edges(
    image: np.ndarray,
    blur_kernel: Tuple[int, int] = (5, 5),
    low_threshold: int = 50,
    high_threshold: int = 150,
    use_auto_canny: bool = False,
    apply_morphology: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Hàm quy trình (Pipeline) hoàn chỉnh cho Module 1:
    Ảnh đầu vào -> Ảnh xám khử nhiễu -> Biên Canny thô -> Biên Canny liền mạch.

    Args:
        image (np.ndarray): Ảnh gốc (BGR).
        blur_kernel (Tuple[int, int]): Cửa sổ Gaussian Blur.
        low_threshold (int): Ngưỡng dưới Canny.
        high_threshold (int): Ngưỡng trên Canny.
        use_auto_canny (bool): Có dùng thuật toán Auto-Canny hay không.
        apply_morphology (bool): Có áp dụng hàn gắn biên hay không.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            (gray_blurred, raw_edges, connected_edges)
    """
    # 1. Chuyển ảnh xám
    gray = to_grayscale(image)

    # 2. Khử nhiễu Gaussian
    gray_blurred = apply_gaussian_blur(gray, kernel_size=blur_kernel)

    # 3. Dò biên Canny
    if use_auto_canny:
        raw_edges = auto_canny(gray_blurred)
    else:
        raw_edges = detect_canny_edges(
            gray_blurred,
            low_threshold=low_threshold,
            high_threshold=high_threshold
        )

    # 4. Hàn gắn biên hình thái học
    if apply_morphology:
        connected_edges = connect_edges(raw_edges)
    else:
        connected_edges = raw_edges.copy()

    return gray_blurred, raw_edges, connected_edges


def resize_image_with_ratio(
    image: np.ndarray,
    target_height: int = 600
) -> Tuple[np.ndarray, float]:
    """
    Chuẩn hóa kích thước ảnh về một chiều cao xác định để tăng tốc độ xử lý
    và giảm nhiễu hạt/chữ viết li ti khi dò biên mép giấy.

    Args:
        image (np.ndarray): Ảnh gốc kích thước bất kỳ.
        target_height (int): Chiều cao mục tiêu sau khi thu nhỏ (mặc định 600px).

    Returns:
        Tuple[np.ndarray, float]:
            - resized_image: Ảnh đã đổi kích thước.
            - ratio: Tỷ lệ co giãn (original_height / target_height) để ánh xạ
                     tọa độ phát hiện được trở lại kích thước ảnh gốc.
    """
    h, w = image.shape[:2]
    if h <= target_height:
        return image.copy(), 1.0

    ratio = h / float(target_height)
    target_width = int(w / ratio)
    resized = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
    return resized, ratio


def robust_edge_detection(
    image: np.ndarray,
    blur_kernel: Tuple[int, int] = (5, 5)
) -> np.ndarray:
    """
    Quy trình dò biên vững chắc (Robust Edge Pipeline) cho tài liệu chụp thực tế:
    Kết hợp Canny nhạy bén + Phép đóng hình thái học mạnh + Ranh giới phân ngưỡng Otsu
    để giải quyết triệt để tình trạng mép giấy bị đứt đoạn do bóng đổ, ánh sáng loang hoặc vân gỗ.

    Args:
        image (np.ndarray): Ảnh đầu vào (thường là ảnh đã resize).
        blur_kernel (Tuple[int, int]): Kích thước cửa sổ Gaussian Blur.

    Returns:
        np.ndarray: Ảnh nhị phân các cạnh biên khép kín liên tục.
    """
    gray = to_grayscale(image)
    blurred = apply_gaussian_blur(gray, kernel_size=blur_kernel)

    # 1. Canny với ngưỡng nhạy (low=30, high=120)
    edges = detect_canny_edges(blurred, low_threshold=30, high_threshold=120)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 2. Ngưỡng phân tách Otsu nhằm bù đắp cho các cạnh biên bị mờ nhạt do chênh lệch ánh sáng
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh_edges = detect_canny_edges(thresh, low_threshold=50, high_threshold=150)
    thresh_closed = cv2.morphologyEx(thresh_edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 3. Kết hợp 2 mặt nạ biên (Logical OR)
    combined = cv2.bitwise_or(edges_closed, thresh_closed)
    return combined



# ----------------------------------------------------------------------
# KIỂM THỬ ĐỘC LẬP (Unit Test / Demo khi chạy trực tiếp file)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import os
    import sys
    import io

    # Đảm bảo in tiếng Việt trên console Windows không lỗi charmap
    if sys.stdout.encoding.lower() != 'utf-8':
        try:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        except Exception:
            pass

    print("=" * 60)
    print("CHẠY THỬ NGHIỆM MODULE 1: TIỀN XỬ LÝ & DÒ BIÊN (TV1)")
    print("=" * 60)

    # Đọc ảnh mẫu chụp nghiêng từ dataset nếu có, nếu chưa có thì tạo ảnh giả lập
    sample_path = os.path.join("dataset", "raw", "sample_doc_skewed.png")
    if os.path.exists(sample_path):
        print(f"[*] Đọc ảnh mẫu thực nghiệm: {sample_path}")
        test_image = cv2.imread(sample_path)
    else:
        print("[*] Sử dụng ảnh mẫu giả lập có nhiễu Gaussian...")
        dummy_image = np.full((600, 800, 3), 40, dtype=np.uint8)
        pts = np.array([[180, 120], [640, 160], [600, 520], [140, 470]], np.int32)
        cv2.fillPoly(dummy_image, [pts], (230, 230, 230))
        cv2.line(dummy_image, (220, 220), (560, 250), (20, 20, 20), 3)
        cv2.line(dummy_image, (210, 270), (550, 300), (20, 20, 20), 3)
        noise = np.random.normal(0, 12, dummy_image.shape).astype(np.uint8)
        test_image = cv2.add(dummy_image, noise)

    print(f"1. Kích thước ảnh kiểm thử: {test_image.shape}")

    # Chạy pipeline Module 1
    gray_blur, raw_edges, final_edges = preprocess_and_detect_edges(
        test_image,
        blur_kernel=(5, 5),
        low_threshold=50,
        high_threshold=150,
        use_auto_canny=False,
        apply_morphology=True
    )

    print(f"2. Ảnh xám khử nhiễu: shape={gray_blur.shape}, dtype={gray_blur.dtype}")
    print(f"3. Ảnh Canny thô: số lượng pixel biên = {np.count_nonzero(raw_edges)}")
    print(f"4. Ảnh biên sau xử lý hình thái học: số lượng pixel biên = {np.count_nonzero(final_edges)}")

    # Thử nghiệm Auto-Canny
    auto_edges = auto_canny(gray_blur)
    print(f"5. Ảnh Auto-Canny: số lượng pixel biên = {np.count_nonzero(auto_edges)}")

    # Lưu kết quả kiểm thử vào dataset/output để kiểm chứng
    os.makedirs(os.path.join("dataset", "output"), exist_ok=True)
    cv2.imwrite(os.path.join("dataset", "output", "m1_gray_blur.png"), gray_blur)
    cv2.imwrite(os.path.join("dataset", "output", "m1_raw_edges.png"), raw_edges)
    cv2.imwrite(os.path.join("dataset", "output", "m1_connected_edges.png"), final_edges)
    cv2.imwrite(os.path.join("dataset", "output", "m1_auto_canny.png"), auto_edges)
    print("[*] Đã lưu các ảnh kết quả kiểm thử vào thư mục 'dataset/output/'")

    print("\n=> Module 1 hoạt động hoàn toàn chính xác và đã sẵn sàng tích hợp với Module 2!")
    print("=" * 60)
