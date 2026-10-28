
"""
Module 4: Image Enhancement
Chức năng:
    1. Chuyển ảnh sang ảnh xám.
    2. Cân bằng ánh sáng và giảm ảnh hưởng của bóng đổ.
    3. Làm nét văn bản.
    4. Nhị phân hóa thích nghi.
    5. Kết hợp các bước xử lý thành một pipeline hoàn chỉnh.

Đầu vào:
    Ảnh tài liệu dạng NumPy array.

Đầu ra:
    - Ảnh đã cân bằng ánh sáng.
    - Ảnh đã làm nét.
    - Ảnh nhị phân đen trắng.
"""

import os
from typing import Tuple

import cv2
import numpy as np


# ============================================================
# 1. KIỂM TRA ẢNH ĐẦU VÀO
# ============================================================

def validate_image(image: np.ndarray) -> None:
    """
    Kiểm tra ảnh đầu vào có hợp lệ hay không.

    Ảnh phải là NumPy array, không rỗng và có kiểu dữ liệu uint8.
    """
    if image is None:
        raise ValueError("Ảnh đầu vào không được là None.")

    if not isinstance(image, np.ndarray):
        raise TypeError("Ảnh đầu vào phải là một NumPy array.")

    if image.size == 0:
        raise ValueError("Ảnh đầu vào không được rỗng.")

    if image.dtype != np.uint8:
        raise TypeError(
            "Ảnh phải có kiểu dữ liệu uint8, "
            "với giá trị pixel nằm trong khoảng 0-255."
        )

    if image.ndim not in (2, 3):
        raise ValueError("Ảnh phải có 2 hoặc 3 chiều.")

    if image.ndim == 3 and image.shape[2] not in (3, 4):
        raise ValueError("Ảnh màu phải có 3 kênh BGR hoặc 4 kênh BGRA.")


# ============================================================
# 2. CHUYỂN ẢNH SANG ẢNH XÁM
# ============================================================

def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Chuyển ảnh màu sang ảnh xám.

    Nếu ảnh đã là ảnh xám thì trả về bản sao.
    Ảnh có 4 kênh được chuyển từ BGRA sang ảnh xám.
    """
    validate_image(image)

    if image.ndim == 2:
        return image.copy()

    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# ============================================================
# 3. CÂN BẰNG ÁNH SÁNG VÀ GIẢM BÓNG ĐỔ
# ============================================================

def correct_illumination(
    image: np.ndarray,
    sigma: float = 25.0
) -> np.ndarray:
    """
    Cân bằng ánh sáng bằng phương pháp ước lượng nền sáng.

    Các bước:
        - Chuyển ảnh sang ảnh xám.
        - Làm mờ ảnh để ước lượng ánh sáng nền.
        - Điều chỉnh độ sáng dựa trên nền ước lượng.
        - Giới hạn pixel trong khoảng 0-255.

    Tham số:
        image: Ảnh đầu vào BGR, BGRA hoặc ảnh xám.
        sigma: Độ rộng làm mờ khi ước lượng nền sáng.

    Trả về:
        Ảnh xám đã cân bằng ánh sáng.
    """
    if not isinstance(sigma, (int, float)) or isinstance(sigma, bool):
        raise TypeError("sigma phải là một số.")

    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma phải lớn hơn 0.")

    gray = convert_to_grayscale(image)

    # Ước lượng nền sáng bằng bộ lọc Gaussian.
    background = cv2.GaussianBlur(
        gray,
        (0, 0),
        sigmaX=float(sigma),
        sigmaY=float(sigma)
    )

    # Chuyển sang số thực để phép chia chính xác hơn.
    gray_float = gray.astype(np.float32)
    background_float = background.astype(np.float32)

    # Tránh chia cho 0 tại các vùng quá tối.
    background_float = np.maximum(background_float, 1.0)

    # Điều chỉnh độ sáng tương đối theo nền sáng.
    mean_background = float(np.mean(background_float))

    corrected = (
        gray_float / background_float * mean_background
    )

    # Đưa giá trị pixel về khoảng hợp lệ của ảnh 8-bit.
    corrected = np.clip(corrected, 0, 255)

    return corrected.astype(np.uint8)


# ============================================================
# 4. LÀM NÉT VĂN BẢN
# ============================================================

def sharpen_text(
    image: np.ndarray,
    strength: float = 1.0
) -> np.ndarray:
    """
    Làm nét văn bản bằng kernel tích chập.

    Pixel trung tâm được tăng trọng số, còn các pixel xung quanh
    được lấy với trọng số âm để tăng độ rõ của đường viền.

    Tham số:
        image: Ảnh đầu vào.
        strength: Cường độ làm nét.
                  0.0 là giữ nguyên ảnh.
                  1.0 là mức làm nét thông thường.
                  Giá trị lớn hơn 1.0 làm nét mạnh hơn.

    Trả về:
        Ảnh đã làm nét.
    """
    if not isinstance(strength, (int, float)) or isinstance(strength, bool):
        raise TypeError("strength phải là một số.")

    if not np.isfinite(strength) or strength < 0:
        raise ValueError("strength phải là số hữu hạn không âm.")

    validate_image(image)

    # Không xử lý nếu cường độ làm nét bằng 0.
    if strength == 0:
        return image.copy()

    alpha = float(strength)

    # Kernel làm nét có tổng hệ số bằng 1.
    kernel = np.array(
        [
            [0, -alpha, 0],
            [-alpha, 1 + 4 * alpha, -alpha],
            [0, -alpha, 0]
        ],
        dtype=np.float32
    )

    # Áp dụng kernel để làm rõ chữ và đường viền.
    sharpened = cv2.filter2D(
        image,
        ddepth=-1,
        kernel=kernel
    )

    return sharpened


# ============================================================
# 5. NHỊ PHÂN HÓA THÍCH NGHI
# ============================================================

def adaptive_threshold(
    image: np.ndarray,
    block_size: int = 31,
    C: float = 10.0
) -> np.ndarray:
    """
    Chuyển ảnh xám thành ảnh nhị phân đen trắng.

    Adaptive Threshold tính ngưỡng riêng cho từng vùng nhỏ,
    phù hợp với tài liệu có ánh sáng không đồng đều.

    Tham số:
        image: Ảnh đầu vào.
        block_size: Kích thước vùng tính ngưỡng,
                    phải là số lẻ lớn hơn 1.
        C: Giá trị được trừ khỏi ngưỡng cục bộ.

    Trả về:
        Ảnh nhị phân gồm các pixel đen và trắng.
    """
    if (
        not isinstance(block_size, (int, np.integer))
        or isinstance(block_size, (bool, np.bool_))
    ):
        raise TypeError("block_size phải là số nguyên.")

    if block_size <= 1 or block_size % 2 == 0:
        raise ValueError("block_size phải là số lẻ lớn hơn 1.")

    if not isinstance(C, (int, float)) or isinstance(C, bool):
        raise TypeError("C phải là một số.")

    if not np.isfinite(C):
        raise ValueError("C phải là số hữu hạn.")

    gray = convert_to_grayscale(image)

    # Làm mượt nhẹ để giảm nhiễu trước khi phân ngưỡng.
    smoothed = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    # Tính ngưỡng cục bộ bằng phương pháp Gaussian.
    binary = cv2.adaptiveThreshold(
        smoothed,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=int(block_size),
        C=float(C)
    )

    return binary


# ============================================================
# 6. PIPELINE TĂNG CƯỜNG CHẤT LƯỢNG ẢNH
# ============================================================

def enhance_document(
    image: np.ndarray,
    sigma: float = 25.0,
    sharpen_strength: float = 1.0,
    block_size: int = 31,
    C: float = 10.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Thực hiện toàn bộ quy trình tăng cường chất lượng ảnh.

    Các bước:
        1. Cân bằng ánh sáng.
        2. Làm nét văn bản.
        3. Nhị phân hóa thích nghi.

    Trả về:
        Một tuple gồm:
        - corrected: Ảnh đã cân bằng ánh sáng.
        - sharpened: Ảnh đã làm nét.
        - binary: Ảnh nhị phân đen trắng.

    Module 5 có thể sử dụng các ảnh đầu ra tùy theo nhu cầu.
    """
    validate_image(image)

    # Bước 1: Giảm ảnh hưởng của ánh sáng không đồng đều.
    corrected = correct_illumination(
        image,
        sigma=sigma
    )

    # Bước 2: Tăng độ rõ của văn bản.
    sharpened = sharpen_text(
        corrected,
        strength=sharpen_strength
    )

    # Bước 3: Chuyển ảnh thành đen trắng.
    binary = adaptive_threshold(
        sharpened,
        block_size=block_size,
        C=C
    )

    return corrected, sharpened, binary


# ============================================================
# 7. TẠO ẢNH MẪU ĐỂ KIỂM TRA
# ============================================================

def create_sample_image() -> np.ndarray:
    """
    Tạo ảnh tài liệu giả lập để kiểm tra Module 4.

    Ảnh gồm nền trắng, đường viền và một số dòng chữ.
    Sau đó tạo hiệu ứng ánh sáng không đồng đều.
    """
    height = 800
    width = 600

    image = np.full(
        (height, width, 3),
        255,
        dtype=np.uint8
    )

    # Vẽ khung trang tài liệu.
    cv2.rectangle(
        image,
        (40, 30),
        (560, 770),
        (40, 40, 40),
        2
    )

    # Tạo tiêu đề.
    cv2.putText(
        image,
        "DOCUMENT SCANNER",
        (80, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (20, 20, 20),
        2,
        cv2.LINE_AA
    )

    # Tạo các dòng văn bản giả lập.
    for i in range(9):
        y = 160 + i * 55

        cv2.putText(
            image,
            f"Document sample line {i + 1}",
            (75, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (35, 35, 35),
            1,
            cv2.LINE_AA
        )

        cv2.line(
            image,
            (75, y + 15),
            (520, y + 15),
            (120, 120, 120),
            1
        )

    # Tạo hiệu ứng bóng đổ thay đổi theo chiều ngang.
    image_float = image.astype(np.float32)

    lighting_gradient = np.linspace(
        0.55,
        1.0,
        width,
        dtype=np.float32
    )

    lighting_mask = np.tile(
        lighting_gradient,
        (height, 1)
    )

    image_with_shadow = (
        image_float * lighting_mask[:, :, np.newaxis]
    )

    return np.clip(
        image_with_shadow,
        0,
        255
    ).astype(np.uint8)


# ============================================================
# 8. KIỂM TRA MODULE 4 ĐỘC LẬP
# ============================================================

if __name__ == "__main__":
    # Đường dẫn ảnh đầu vào và thư mục lưu kết quả.
    input_path = os.path.join(
        "dataset",
        "raw",
        "sample_doc_skewed.png"
    )

    output_dir = os.path.join(
        "dataset",
        "output"
    )

    # Tạo thư mục đầu ra nếu chưa tồn tại.
    os.makedirs(output_dir, exist_ok=True)

    # Đọc ảnh mẫu từ thư mục dữ liệu.
    input_image = cv2.imread(
        input_path,
        cv2.IMREAD_COLOR
    )

    # Tạo ảnh mẫu nếu không tìm thấy file đầu vào.
    if input_image is None:
        print("Không tìm thấy ảnh mẫu:", input_path)
        print("Đang tạo ảnh tài liệu giả lập...")
        input_image = create_sample_image()

    print("Kích thước ảnh đầu vào:", input_image.shape)

    try:
        # Chạy toàn bộ pipeline tăng cường ảnh.
        corrected, sharpened, binary = enhance_document(
            input_image
        )

        # Lưu ba ảnh kết quả để kiểm tra.
        output_paths = {
            "corrected": os.path.join(
                output_dir,
                "m4_illumination_corrected.png"
            ),
            "sharpened": os.path.join(
                output_dir,
                "m4_sharpened.png"
            ),
            "binary": os.path.join(
                output_dir,
                "m4_adaptive_threshold.png"
            )
        }

        save_results = [
            cv2.imwrite(output_paths["corrected"], corrected),
            cv2.imwrite(output_paths["sharpened"], sharpened),
            cv2.imwrite(output_paths["binary"], binary)
        ]

        if not all(save_results):
            raise IOError("Không thể lưu một hoặc nhiều ảnh kết quả.")

        print("\nImage enhancement completed successfully!")
        print("Illumination corrected:", corrected.shape)
        print("Sharpened image:", sharpened.shape)
        print("Adaptive threshold:", binary.shape)

        print("\nOutput files:")
        for path in output_paths.values():
            print(path)

    except (ValueError, TypeError, IOError, cv2.error) as error:
        print("Lỗi khi xử lý ảnh:", error)
