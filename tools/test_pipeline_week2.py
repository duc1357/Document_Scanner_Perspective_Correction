"""
Script kiểm thử toàn diện Tuần 2: Ghép nối Module 1 (Dò biên) & Module 2 (Nắn thẳng góc nhìn)
Đánh giá trên toàn bộ bộ dữ liệu ảnh thực nghiệm (25 ảnh thực tế + 2 ảnh mẫu).
"""

import os
import sys
import io
import time
from typing import List, Tuple
import cv2
import numpy as np

# Fix Unicode/UTF-8 output trên terminal Windows
if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    except Exception:
        pass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.perspective_warp import warp_document_pipeline, order_points


def draw_detected_quad(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """
    Vẽ khung đa giác 4 góc nhận diện được lên ảnh gốc để trực quan hóa kết quả.
    """
    vis = image.copy()
    ordered = order_points(corners)
    pts = ordered.astype(int)

    # 1. Vẽ các cạnh viền đa giác
    cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 0), thickness=max(2, int(vis.shape[0] / 300)))

    # 2. Vẽ 4 đỉnh và tên đỉnh
    labels = ["TL", "TR", "BR", "BL"]
    colors = [(0, 0, 255), (255, 0, 0), (0, 255, 255), (255, 0, 255)]
    radius = max(5, int(vis.shape[0] / 120))
    font_scale = max(0.6, vis.shape[0] / 1000.0)

    for i, (pt, label, col) in enumerate(zip(pts, labels, colors)):
        cv2.circle(vis, tuple(pt), radius, col, -1)
        cv2.circle(vis, tuple(pt), radius + 2, (255, 255, 255), 2)
        cv2.putText(
            vis,
            f"{label}",
            (pt[0] + 12, pt[1] + 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thickness=3,
            lineType=cv2.LINE_AA
        )
        cv2.putText(
            vis,
            f"{label}",
            (pt[0] + 12, pt[1] + 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness=1,
            lineType=cv2.LINE_AA
        )

    return vis


def run_week2_benchmark():
    raw_dir = os.path.join("dataset", "raw")
    output_dir = os.path.join("dataset", "output", "pipeline_week2")
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(raw_dir):
        print(f"[!] Thư mục {raw_dir} không tồn tại.")
        return

    # Lấy danh sách ảnh hợp lệ
    image_files = sorted([
        f for f in os.listdir(raw_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
    ])

    print("=" * 80)
    print("BÁO CÁO THỰC NGHIỆM TUẦN 2: GHÉP NỐI MODULE 1 & 2 TRÊN DATASET THỰC TẾ")
    print(f"Tổng số ảnh thực nghiệm: {len(image_files)}")
    print("=" * 80)
    print(f"{'STT':<4} | {'Tên File':<24} | {'Kích thước gốc':<15} | {'Thời gian':<10} | {'Kết quả'}")
    print("-" * 80)

    success_count = 0
    total_time = 0.0

    for idx, fname in enumerate(image_files, 1):
        fpath = os.path.join(raw_dir, fname)
        img = cv2.imread(fpath)
        if img is None:
            print(f"{idx:<4} | {fname:<24} | {'Lỗi đọc file':<15} | {'-':<10} | THẤT BẠI")
            continue

        orig_shape_str = f"{img.shape[1]}x{img.shape[0]}"

        # Đo thời gian thực thi trọn vẹn pipeline
        start_t = time.perf_counter()
        warped, corners, edges = warp_document_pipeline(img, target_process_height=600)
        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        total_time += elapsed_ms

        if warped is not None and corners is not None:
            success_count += 1
            status = "THÀNH CÔNG"

            # Lưu ảnh phát hiện 4 góc trên ảnh gốc
            vis_corners = draw_detected_quad(img, corners)
            base_name = os.path.splitext(fname)[0]
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_01_corners.jpg"), vis_corners)
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_02_warped.jpg"), warped)
        else:
            status = "THẤT BẠI (Không tìm thấy 4 góc)"

        print(f"{idx:<4} | {fname:<24} | {orig_shape_str:<15} | {elapsed_ms:6.1f} ms   | {status}")

    print("=" * 80)
    avg_time = total_time / max(1, len(image_files))
    success_rate = (success_count / max(1, len(image_files))) * 100.0
    print(f"KẾT QUẢ TỔNG KẾT:")
    print(f"  + Số ảnh xử lý thành công: {success_count}/{len(image_files)} ({success_rate:.1f}%)")
    print(f"  + Thời gian xử lý trung bình mỗi trang: {avg_time:.1f} ms (~{1000.0/max(1, avg_time):.1f} FPS)")
    print(f"  + Thư mục lưu kết quả trực quan: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    run_week2_benchmark()
