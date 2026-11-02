"""
Script kiểm thử chuẩn hóa API Tuần 4 - Module 2 (TV2 - nTrường)
Mục tiêu:
    - Kiểm định tính toàn vẹn của API Module 2 (warp_document_pipeline, draw_detected_quad).
    - Đo lường độ trễ xử lý (Latency - ms), tính toán FPS trên 27 ảnh thực tế.
    - Xác thực định dạng dữ liệu đầu ra: warped (uint8), orig_corners (4, 2), edge_mask (uint8).
    - Xuất báo cáo CSV chuẩn hóa phục vụ tích hợp với TV5 (UI Desktop).
"""

import os
import sys
import time
import csv
from pathlib import Path
import cv2
import numpy as np

# Thiết lập đường dẫn thư mục gốc
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from modules.perspective_warp import warp_document_pipeline, draw_detected_quad, order_points

INPUT_DIR = PROJECT_ROOT / "dataset" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "dataset" / "output" / "tv2_week4"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def validate_outputs(warped, corners, edge_mask, orig_shape):
    """
    Xác thực nghiêm ngặt định dạng dữ liệu chuẩn hóa của API Module 2.
    """
    # 1. Kiểm tra edge_mask (Module 1 -> Module 2)
    if not isinstance(edge_mask, np.ndarray) or edge_mask.dtype != np.uint8 or edge_mask.ndim != 2:
        return False, "edge_mask không hợp lệ (phải là np.ndarray uint8 2D)"

    # 2. Kiểm tra corners
    if corners is None or not isinstance(corners, np.ndarray):
        return False, "corners bị None hoặc không phải np.ndarray"
    if corners.shape != (4, 2):
        return False, f"corners có shape {corners.shape} thay vì (4, 2)"

    # 3. Kiểm tra warped
    if warped is None or not isinstance(warped, np.ndarray) or warped.dtype != np.uint8:
        return False, "warped bị None hoặc không phải uint8 np.ndarray"
    if warped.size == 0 or warped.shape[0] < 50 or warped.shape[1] < 50:
        return False, f"warped có kích thước quá nhỏ: {warped.shape}"

    return True, "Hợp lệ"


def test_image(image_path: Path):
    """
    Kiểm thử API Module 2 trên một file ảnh.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        return {
            "image": image_path.name,
            "status": "FAIL",
            "time_ms": 0.0,
            "orig_size": "N/A",
            "warped_size": "N/A",
            "details": "Không thể đọc ảnh"
        }

    h, w = image.shape[:2]
    orig_size_str = f"{w}x{h}"

    t0 = time.perf_counter()
    try:
        warped, corners, edge_mask = warp_document_pipeline(image)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        is_valid, reason = validate_outputs(warped, corners, edge_mask, image.shape)
        if not is_valid:
            return {
                "image": image_path.name,
                "status": "FAIL",
                "time_ms": round(latency_ms, 2),
                "orig_size": orig_size_str,
                "warped_size": "N/A",
                "details": reason
            }

        warped_size_str = f"{warped.shape[1]}x{warped.shape[0]}"

        # Trực quan hóa 4 góc phục vụ Màn hình 1 UI
        vis_corners = draw_detected_quad(image, corners)

        # Lưu artifacts kiểm thử tuần 4
        cv2.imwrite(str(OUTPUT_DIR / f"{image_path.stem}_01_corners.jpg"), vis_corners)
        cv2.imwrite(str(OUTPUT_DIR / f"{image_path.stem}_02_edge_mask.png"), edge_mask)
        cv2.imwrite(str(OUTPUT_DIR / f"{image_path.stem}_03_warped.jpg"), warped)

        return {
            "image": image_path.name,
            "status": "PASS",
            "time_ms": round(latency_ms, 2),
            "orig_size": orig_size_str,
            "warped_size": warped_size_str,
            "details": "API chuẩn hóa đạt 100% tiêu chí đầu ra"
        }

    except Exception as err:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "image": image_path.name,
            "status": "FAIL",
            "time_ms": round(latency_ms, 2),
            "orig_size": orig_size_str,
            "warped_size": "N/A",
            "details": str(err)
        }


def main():
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths = sorted([p for p in INPUT_DIR.iterdir() if p.suffix.lower() in extensions])

    print("=" * 80)
    print("BÁO CÁO KIỂM THỬ CHUẨN HÓA API TUẦN 4 - MODULE 2 (PERSPECTIVE WARP - TV2)")
    print(f"Tổng số ảnh thực nghiệm: {len(image_paths)}")
    print("=" * 80)
    print(f"{'STT':<4} | {'Tên File':<22} | {'Kích thước gốc':<14} | {'Kích thước nắn':<14} | {'Thời gian':<10} | {'Kết quả':<8}")
    print("-" * 80)

    results = []
    total_time = 0.0
    passed_count = 0

    for idx, img_path in enumerate(image_paths, start=1):
        res = test_image(img_path)
        results.append(res)
        if res["status"] == "PASS":
            passed_count += 1
            total_time += res["time_ms"]
            print(f"{idx:<4} | {res['image']:<22} | {res['orig_size']:<14} | {res['warped_size']:<14} | {res['time_ms']:>6.1f} ms | {res['status']:<8}")
        else:
            print(f"{idx:<4} | {res['image']:<22} | {res['orig_size']:<14} | {'N/A':<14} | {res['time_ms']:>6.1f} ms | {res['status']} ({res['details']})")

    avg_time = (total_time / passed_count) if passed_count > 0 else 0.0
    fps = (1000.0 / avg_time) if avg_time > 0 else 0.0

    print("=" * 80)
    print("KẾT QUẢ TỔNG KẾT:")
    print(f"  + Thành công: {passed_count}/{len(image_paths)} ({(passed_count/len(image_paths)*100):.1f}%)")
    print(f"  + Thời gian xử lý trung bình: {avg_time:.2f} ms (~{fps:.1f} FPS)")
    print(f"  + Thư mục lưu kết quả: {OUTPUT_DIR}")

    # Xuất báo cáo CSV
    csv_file = OUTPUT_DIR / "api_test_report.csv"
    with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "status", "time_ms", "orig_size", "warped_size", "details"])
        writer.writeheader()
        writer.writerows(results)

    print(f"  + Báo cáo CSV đã ghi tại: {csv_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()

