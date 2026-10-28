"""
Script kiểm thử Module 3: Hiệu chỉnh hướng văn bản tự động (ORB + Hamming + RANSAC + Deskew)
Thành viên phụ trách: TV3 (Pháp)
"""

import os
import sys
import io
import time
import cv2
import numpy as np

# Fix Unicode/UTF-8 output trên terminal Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from modules.orientation_corrector import OrientationCorrector


def run_module3_tests():
    raw_dir = os.path.join(PROJECT_ROOT, "dataset", "raw")
    output_dir = os.path.join(PROJECT_ROOT, "dataset", "output", "tv3_orientation")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("KIỂM THỬ THỰC NGHIỆM MODULE 3: ORB + HAMMING + RANSAC & STANDALONE DESKEW")
    print("=" * 80)

    corrector = OrientationCorrector(n_features=3000, min_matches=12, ransac_threshold=3.0)

    # 1. Thử nghiệm tính toán góc xoay RANSAC với các góc mô phỏng (15, 30, 45, 90 độ)
    sample_path = os.path.join(raw_dir, "sample_doc_skewed.png")
    if not os.path.exists(sample_path):
        sample_path = os.path.join(raw_dir, "TaiLieuA4_01.jpg")

    ref_img = cv2.imread(sample_path)
    if ref_img is None:
        print("[!] Không tìm thấy ảnh mẫu kiểm thử.")
        return

    print("\n[PHẦN 1] KIỂM THỬ ĐỐI SÁNH ORB + RANSAC TRÊN CÁC GÓC XOAY MÔ PHỎNG:")
    print("-" * 80)
    print(f"{'Góc thực tế (độ)':<20} | {'Góc tính toán (độ)':<25} | {'Sai số tuyệt đối':<20} | {'Kết quả'}")
    print("-" * 80)

    test_angles = [0.0, 10.0, 15.0, 30.0, -20.0]
    h, w = ref_img.shape[:2]
    center = (w // 2, h // 2)

    for target_angle in test_angles:
        M = cv2.getRotationMatrix2D(center, target_angle, 1.0)
        rotated = cv2.warpAffine(ref_img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        start_t = time.perf_counter()
        matrix, estimated_angle, success = corrector.estimate_affine_matrix(rotated, ref_img)
        elapsed_ms = (time.perf_counter() - start_t) * 1000

        if success:
            err = abs(abs(estimated_angle) - abs(target_angle))
            status = "ĐẠT" if err < 1.0 else "CHẤP NHẬN"
            print(f"{target_angle:<20.1f} | {estimated_angle:<25.2f} | {err:<20.2f} | {status} ({elapsed_ms:.1f}ms)")
        else:
            print(f"{target_angle:<20.1f} | {'Không hội tụ':<25} | {'-':<20} | THẤT BẠI")

    # 2. Thử nghiệm Standalone Deskew trên tập ảnh raw
    print("\n[PHẦN 2] KIỂM THỬ KHỬ NGHIÊNG TỰ ĐỘNG (STANDALONE DESKEW):")
    print("-" * 80)
    image_files = sorted([f for f in os.listdir(raw_dir) if f.lower().endswith(('.jpg', '.png'))])[:5]

    for fname in image_files:
        img_path = os.path.join(raw_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            continue

        start_t = time.perf_counter()
        corrected = corrector.process(img)
        elapsed_ms = (time.perf_counter() - start_t) * 1000

        out_path = os.path.join(output_dir, f"corrected_{fname}")
        cv2.imwrite(out_path, corrected)
        print(f"  + Đã xử lý {fname:<20} -> {out_path} ({elapsed_ms:.1f}ms)")

    print("\n" + "=" * 80)
    print(f"Hoàn tất kiểm thử Module 3. Ảnh kết quả lưu tại: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    run_module3_tests()
