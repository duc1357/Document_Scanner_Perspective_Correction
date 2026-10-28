"""
Ứng dụng Document Scanner & Perspective Correction
Tích hợp toàn diện 4 Module:
    - Module 1: Tiền xử lý & Dò biên (Gaussian Blur + Canny Edge Detection)
    - Module 2: Trích xuất 4 góc & Nắn thẳng hình học (Perspective Transform)
    - Module 3: Hiệu chỉnh hướng văn bản tự động (ORB + Hamming + RANSAC & Deskew)
    - Module 4: Nâng cao chất lượng ảnh (Cân bằng sáng, Làm nét & Nhị phân hóa thích nghi)
"""

import os
import sys
import io
import argparse
import time
import cv2
import numpy as np

# Fix Unicode/UTF-8 output trên terminal Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

from modules import (
    warp_document_pipeline,
    OrientationCorrector,
    enhance_document,
)


def scan_document(
    image_path: str,
    output_dir: str = "dataset/output/scanned",
    enable_orientation: bool = True,
    enable_enhancement: bool = True
) -> dict:
    """
    Quy trình quét tài liệu tự động toàn diện:
    1. Đọc ảnh đầu vào
    2. Nắn thẳng phối cảnh (Module 1 + 2)
    3. Hiệu chỉnh chiều xoay (Module 3)
    4. Tăng cường độ nét & nhị phân hóa (Module 4)
    5. Lưu kết quả
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Không tìm thấy file ảnh: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Không thể đọc file ảnh: {image_path}")

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    results = {}

    print(f"\n[*] Bắt đầu xử lý: {os.path.basename(image_path)} ({image.shape[1]}x{image.shape[0]})")
    t0 = time.perf_counter()

    # Bước 1 & 2: Dò biên & Nắn thẳng phối cảnh
    warped, corners, edge_mask = warp_document_pipeline(image)
    if warped is None:
        print("[!] Không tìm thấy khung tài liệu 4 góc. Sử dụng ảnh gốc.")
        warped = image.copy()
    else:
        print("  [+] Bước 1 & 2 (Dò biên & Nắn phẳng): Thành công.")

    results["warped"] = warped
    warped_path = os.path.join(output_dir, f"{base_name}_01_warped.jpg")
    cv2.imwrite(warped_path, warped)

    # Bước 3: Tự động xoay thẳng văn bản
    current_img = warped
    if enable_orientation:
        corrector = OrientationCorrector()
        current_img = corrector.process(current_img)
        print("  [+] Bước 3 (Hiệu chỉnh hướng ORB/Deskew): Thành công.")
        results["oriented"] = current_img
        oriented_path = os.path.join(output_dir, f"{base_name}_02_oriented.jpg")
        cv2.imwrite(oriented_path, current_img)

    # Bước 4: Tăng cường chất lượng ảnh & Làm nét chữ
    if enable_enhancement:
        corrected, sharpened, binary = enhance_document(
            current_img,
            sigma=25.0,
            sharpen_strength=1.2,
            block_size=31,
            C=10.0
        )
        print("  [+] Bước 4 (Cân bằng sáng & Nhị phân hóa): Thành công.")
        results["sharpened"] = sharpened
        results["binary"] = binary

        sharpened_path = os.path.join(output_dir, f"{base_name}_03_sharpened.jpg")
        binary_path = os.path.join(output_dir, f"{base_name}_04_binary.png")
        cv2.imwrite(sharpened_path, sharpened)
        cv2.imwrite(binary_path, binary)

    total_time_ms = (time.perf_counter() - t0) * 1000
    print(f"[v] Hoàn tất quét tài liệu trong {total_time_ms:.1f} ms!")
    print(f"    Thư mục kết quả: {output_dir}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Document Scanner & Perspective Correction Pipeline")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="dataset/raw/TaiLieuA4_01.jpg",
        help="Đường dẫn đến ảnh tài liệu cần quét"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="dataset/output/scanned",
        help="Thư mục lưu ảnh kết quả"
    )
    args = parser.parse_args()

    try:
        scan_document(args.input, args.output)
    except Exception as e:
        print(f"[!] Lỗi: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
