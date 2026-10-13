"""
Script kiểm thử Module 1 (và đánh giá sơ bộ đường bao) trên ảnh tài liệu thực tế.
"""

import os
import sys
import io
import cv2
import numpy as np

# Fix UTF-8 output trên Windows console
if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    except Exception:
        pass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.edge_detector import (
    to_grayscale,
    apply_gaussian_blur,
    detect_canny_edges,
    auto_canny,
    connect_edges,
    preprocess_and_detect_edges
)


def run_test():
    img_path = os.path.join("dataset", "raw", "real_doc_sample.png")
    if not os.path.exists(img_path):
        print(f"[LỖI] Không tìm thấy ảnh: {img_path}")
        return

    img = cv2.imread(img_path)
    print(f"[*] Đã tải ảnh thực tế thành công: shape = {img.shape}")

    output_dir = os.path.join("dataset", "output", "real_test")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Pipeline tiêu chuẩn với Canny thủ công (50, 150)
    gray_blur, raw_edges_manual, connected_manual = preprocess_and_detect_edges(
        img,
        blur_kernel=(5, 5),
        low_threshold=50,
        high_threshold=150,
        use_auto_canny=False,
        apply_morphology=True
    )

    # 2. Pipeline với Auto-Canny (dựa trên median)
    gray = to_grayscale(img)
    median_val = np.median(gray)
    auto_edges = auto_canny(gray_blur, sigma=0.33)
    connected_auto = connect_edges(auto_edges, kernel_size=(5, 5), iterations=1)

    # 3. Thử nghiệm Canny với ngưỡng tối ưu hơn cho ảnh này nếu cần
    # Thử thêm với kernel làm mờ lớn hơn (7, 7) để giảm biên từ chữ viết
    gray_blur_7 = apply_gaussian_blur(gray, kernel_size=(7, 7))
    custom_edges = detect_canny_edges(gray_blur_7, low_threshold=75, high_threshold=200)
    connected_custom = connect_edges(custom_edges, kernel_size=(5, 5), iterations=1)

    # 4. Lưu tất cả ảnh kết quả
    cv2.imwrite(os.path.join(output_dir, "01_gray_blur.png"), gray_blur)
    cv2.imwrite(os.path.join(output_dir, "02_raw_edges_manual.png"), raw_edges_manual)
    cv2.imwrite(os.path.join(output_dir, "03_connected_manual.png"), connected_manual)
    cv2.imwrite(os.path.join(output_dir, "04_auto_canny.png"), auto_edges)
    cv2.imwrite(os.path.join(output_dir, "05_connected_auto.png"), connected_auto)
    cv2.imwrite(os.path.join(output_dir, "06_custom_edges.png"), custom_edges)
    cv2.imwrite(os.path.join(output_dir, "07_connected_custom.png"), connected_custom)

    print(f"[*] Giá trị trung vị (Median) của ảnh xám: {median_val:.1f}")
    print(f"[*] Số pixel biên (Manual 50-150): {np.count_nonzero(raw_edges_manual)}")
    print(f"[*] Số pixel biên (Auto Canny): {np.count_nonzero(auto_edges)}")
    print(f"[*] Số pixel biên (Custom 75-200, blur 7x7): {np.count_nonzero(custom_edges)}")

    # 5. Đánh giá sơ bộ xem mép ngoài của tờ giấy có tạo thành Contour lớn nhất không
    contours, _ = cv2.findContours(connected_manual, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    vis_contour = img.copy()
    if contours:
        print(f"[*] Tìm thấy {len(contours)} contours.")
        max_c = contours[0]
        peri = cv2.arcLength(max_c, True)
        approx = cv2.approxPolyDP(max_c, 0.02 * peri, True)
        print(f"[*] Contour lớn nhất: diện tích = {cv2.contourArea(max_c):.1f}, số đỉnh xấp xỉ = {len(approx)}")
        cv2.drawContours(vis_contour, [max_c], -1, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(output_dir, "08_max_contour_manual.png"), vis_contour)

    # Thử với connected_custom
    contours_c, _ = cv2.findContours(connected_custom, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours_c = sorted(contours_c, key=cv2.contourArea, reverse=True)
    vis_contour_c = img.copy()
    if contours_c:
        max_c = contours_c[0]
        peri = cv2.arcLength(max_c, True)
        approx = cv2.approxPolyDP(max_c, 0.02 * peri, True)
        print(f"[*] Contour lớn nhất (Custom): diện tích = {cv2.contourArea(max_c):.1f}, số đỉnh xấp xỉ = {len(approx)}")
        cv2.drawContours(vis_contour_c, [max_c], -1, (0, 0, 255), 2)
        cv2.imwrite(os.path.join(output_dir, "09_max_contour_custom.png"), vis_contour_c)

    print(f"[OK] Đã hoàn tất và lưu các ảnh kết quả vào: {output_dir}")


if __name__ == "__main__":
    run_test()
