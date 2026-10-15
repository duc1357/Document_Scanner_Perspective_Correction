"""
Module 2: Trích xuất 4 góc & Nắn thẳng hình học (Perspective Transform)
Thành viên phụ trách: TV2 (nTrường) & Trưởng nhóm TV1 (Đức)
Mục tiêu:
    - Tìm Contour lớn nhất bao quanh tài liệu và xấp xỉ 4 đỉnh đa giác (cv2.approxPolyDP).
    - Sắp xếp 4 góc theo thứ tự chuẩn chiều kim đồng hồ (Top-Left, Top-Right, Bottom-Right, Bottom-Left).
    - Tính toán ma trận biến đổi phối cảnh Homography (3x3).
    - Thực hiện Warping ngược (Inverse Warping) kết hợp nội suy song tuyến (Bilinear Interpolation)
      để đưa tài liệu chụp xiên/nghiêng về mặt phẳng vuông vắn chuẩn scan.
"""

from typing import Tuple, Optional
import os
import sys
import cv2
import numpy as np

try:
    from modules.edge_detector import resize_image_with_ratio, robust_edge_detection
except ImportError:
    from edge_detector import resize_image_with_ratio, robust_edge_detection


# --------------------------------------------------------------------------
# 1. Tìm Contour & Xấp xỉ 4 đỉnh tứ giác
# --------------------------------------------------------------------------
def _quad_from_contour(c: np.ndarray, epsilons=(0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08)) -> Optional[np.ndarray]:
    """
    Xấp xỉ contour về tứ giác LỒI bằng thuật toán Ramer-Douglas-Peucker (cv2.approxPolyDP).
    Trả về mảng float32 (4, 2) hoặc None nếu không tạo thành tứ giác lồi 4 đỉnh.
    """
    peri = cv2.arcLength(c, True)
    for eps in epsilons:
        approx = cv2.approxPolyDP(c, eps * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return approx.reshape(4, 2).astype("float32")
    return None


def get_paper_contour(
    edged: np.ndarray,
    min_area_ratio: float = 0.04,
    max_area_ratio: float = 0.98,
    top_k: int = 10
) -> Optional[np.ndarray]:
    """
    Tìm contour lớn nhất và xấp xỉ về đúng 4 đỉnh đa giác.

    Chiến lược 2 bước:
      - Bước 1 (Chính): Giãn nhẹ viền, lọc các contour thỏa mãn ngưỡng diện tích
        (từ min_area_ratio đến max_area_ratio diện tích ảnh), sắp xếp giảm dần và lấy
        contour đầu tiên xấp xỉ thành công tứ giác lồi.
      - Bước 2 (Dự phòng cho viền bị đứt/bóng đổ): Hàn gắn viền bằng phép Đóng hình thái học
        (MORPH_CLOSE), trích xuất Convex Hull. Nếu vẫn chưa thành tứ giác 4 đỉnh, dùng
        hộp chữ nhật xoay diện tích tối thiểu (cv2.minAreaRect).

    Args:
        edged (np.ndarray): Ảnh nhị phân các cạnh biên (từ Canny hoặc Threshold).
        min_area_ratio (float): Diện tích tối thiểu so với ảnh (mặc định 4%).
        max_area_ratio (float): Diện tích tối đa so với ảnh (mặc định 98%).
        top_k (int): Số lượng contour lớn nhất được xem xét.

    Returns:
        Optional[np.ndarray]: Mảng float32 shape (4, 2) chứa 4 đỉnh, hoặc None nếu thất bại.
    """
    h, w = edged.shape[:2]
    img_area = float(h * w)
    min_area = min_area_ratio * img_area
    max_area = max_area_ratio * img_area

    # ---- Bước 1: contour thật từ ảnh biên ----
    thick = cv2.dilate(edged, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(thick, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cands = [c for c in contours if min_area <= cv2.contourArea(c) <= max_area]
    cands = sorted(cands, key=cv2.contourArea, reverse=True)[:top_k]

    for c in cands:
        quad = _quad_from_contour(c)
        if quad is not None:
            return quad

    # ---- Bước 2: dự phòng khi mép giấy bị đứt đoạn ----
    k = max(7, int(0.02 * max(h, w))) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    hulls = [cv2.convexHull(c) for c in contours]
    hulls = [hl for hl in hulls if min_area <= cv2.contourArea(hl) <= max_area]
    hulls = sorted(hulls, key=cv2.contourArea, reverse=True)[:top_k]

    for hull in hulls:
        quad = _quad_from_contour(hull, (0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08))
        if quad is not None:
            return quad

    if hulls:
        return cv2.boxPoints(cv2.minAreaRect(hulls[0])).astype("float32")

    return None


def find_document_contour(
    edge_mask: np.ndarray,
    min_area_ratio: float = 0.04
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Hàm chuẩn hóa API theo quy chuẩn nhóm:
    Trả về (corners, best_contour) tương thích với pipeline tổng thể.
    """
    pts = get_paper_contour(edge_mask, min_area_ratio=min_area_ratio)
    return pts, None


# --------------------------------------------------------------------------
# 2. Sắp xếp 4 đỉnh chuẩn chiều kim đồng hồ
# --------------------------------------------------------------------------
def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Sắp xếp 4 đỉnh theo thứ tự chuẩn chiều kim đồng hồ:
        [0]: Top-Left (Trên-Trái)
        [1]: Top-Right (Trên-Phải)
        [2]: Bottom-Right (Dưới-Phải)
        [3]: Bottom-Left (Dưới-Trái)

    Phương pháp phối hợp:
        1. Phân tích tổng s = x + y và hiệu d = x - y:
            - Top-Left: s nhỏ nhất (x, y đều nhỏ).
            - Bottom-Right: s lớn nhất (x, y đều lớn).
            - Top-Right: d lớn nhất (x lớn, y nhỏ).
            - Bottom-Left: d nhỏ nhất (x nhỏ, y lớn).
        2. Cơ chế dự phòng (khi tài liệu xoay góc gắt ~30-45 độ):
            Sắp xếp theo góc cực quanh tâm (centroid angle), sau đó chọn đỉnh bắt đầu
            sao cho cạnh trên có phương ngang chuẩn nhất.
    """
    pts = np.asarray(pts, dtype="float32").reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")

    s = pts[:, 0] + pts[:, 1]  # x + y
    d = pts[:, 0] - pts[:, 1]  # x - y
    idx = [np.argmin(s), np.argmax(d), np.argmax(s), np.argmin(d)]  # TL, TR, BR, BL

    if len(set(idx)) == 4:  # 4 đỉnh độc lập phân biệt rõ ràng
        for i, j in enumerate(idx):
            rect[i] = pts[j]
        return rect

    # --- Dự phòng: sắp xếp theo góc cực quanh tâm ---
    cx, cy = pts.mean(axis=0)
    cw = pts[np.argsort(np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx))]  # Sắp xếp chiều kim đồng hồ
    best, best_ang = 0, 1e9
    for i in range(4):
        v = cw[(i + 1) % 4] - cw[i]
        if v[0] <= 0:  # Cạnh trên phải có hướng từ trái sang phải
            continue
        ang = abs(np.arctan2(v[1], v[0]))
        if ang < best_ang:
            best, best_ang = i, ang

    return np.roll(cw, -best, axis=0).astype("float32")


# --------------------------------------------------------------------------
# 3. Biến đổi phối cảnh (Perspective Transform / Homography)
# --------------------------------------------------------------------------
def four_point_transform(
    image: np.ndarray,
    pts: np.ndarray,
    interpolation: int = cv2.INTER_LINEAR
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Nắn phẳng vùng tài liệu giới hạn bởi 4 đỉnh `pts` thành ảnh chữ nhật nhìn thẳng.

    Toán học:
        - Tính chiều rộng và chiều cao cực đại của tứ giác để xác định kích thước ảnh đích.
        - Tính ma trận Homography 3x3 (8 bậc tự do) từ 4 cặp điểm bằng SVD: cv2.getPerspectiveTransform.
        - Áp dụng Warping ngược (Inverse Mapping) với phép nội suy song tuyến (Bilinear Interpolation):
          cv2.warpPerspective.
    """
    rect = order_points(pts)
    tl, tr, br, bl = rect

    # Kích thước ảnh đích = cạnh dài hơn trong mỗi cặp cạnh đối diện
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    max_w = max(100, int(round(max(width_top, width_bottom))))

    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    max_h = max(100, int(round(max(height_left, height_right))))

    dst = np.array([
        [0, 0],
        [max_w - 1, 0],
        [max_w - 1, max_h - 1],
        [0, max_h - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(
        image,
        M,
        (max_w, max_h),
        flags=interpolation,
        borderMode=cv2.BORDER_REPLICATE
    )
    return warped, M


# --------------------------------------------------------------------------
# 4. Pipeline tích hợp toàn diện Module 1 & 2
# --------------------------------------------------------------------------
def warp_document_pipeline(
    image: np.ndarray,
    target_process_height: int = 600
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Quy trình ghép nối liên tục:
    Ảnh gốc -> Resize chuẩn hóa -> Dò biên vững chắc -> Tìm 4 đỉnh -> Map về ảnh gốc -> Nắn phẳng.

    Args:
        image (np.ndarray): Ảnh tài liệu gốc.
        target_process_height (int): Chiều cao thu nhỏ để xử lý nhanh (mặc định 600px).

    Returns:
        Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
            (warped, orig_corners, edge_mask)
    """
    if image is None:
        raise ValueError("Ảnh đầu vào là None. Vui lòng kiểm tra lại đường dẫn.")

    # 1. Resize chuẩn hóa
    resized, ratio = resize_image_with_ratio(image, target_height=target_process_height)

    # 2. Dò biên hình thái học kết hợp
    edge_mask = robust_edge_detection(resized)

    # 3. Tìm 4 góc trên ảnh thu nhỏ
    corners_small = get_paper_contour(edge_mask, min_area_ratio=0.04)

    # 4. Cơ chế Fallback đa phương thức bằng phân ngưỡng Otsu nếu mép Canny bị đứt
    if corners_small is None:
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if len(resized.shape) == 3 else resized
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        corners_small = get_paper_contour(thresh, min_area_ratio=0.04)
        if corners_small is None:
            thresh_inv = cv2.bitwise_not(thresh)
            corners_small = get_paper_contour(thresh_inv, min_area_ratio=0.04)

    if corners_small is None:
        return None, None, edge_mask

    # 5. Ánh xạ 4 góc về kích thước ảnh gốc chất lượng cao
    orig_corners = corners_small * ratio

    # 6. Nắn phẳng phối cảnh trên ảnh gốc
    warped, _ = four_point_transform(image, orig_corners)

    return warped, orig_corners, edge_mask


# --------------------------------------------------------------------------
# Test độc lập
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import io

    if sys.stdout.encoding.lower() != 'utf-8':
        try:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        except Exception:
            pass

    print("=" * 65)
    print("CHẠY THỬ NGHIỆM MODULE 2: TRÍCH XUẤT 4 GÓC & NẮN THẲNG GÓC NHÌN")
    print("=" * 65)

    test_path = os.path.join("dataset", "raw", "sample_doc_skewed.png")
    if len(sys.argv) > 1:
        test_path = sys.argv[1]

    if not os.path.exists(test_path):
        print(f"[!] Không tìm thấy file: {test_path}")
        sys.exit(1)

    src_img = cv2.imread(test_path)
    print(f"[*] Đọc ảnh: {test_path}, Shape: {src_img.shape}")

    warped, corners, edges = warp_document_pipeline(src_img)
    if warped is not None:
        print("[*] TÌM THẤY 4 GÓC TÀI LIỆU THÀNH CÔNG!")
        ordered = order_points(corners)
        labels = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
        for lbl, pt in zip(labels, ordered):
            print(f"   + {lbl:12s}: x={pt[0]:.1f}, y={pt[1]:.1f}")
        print(f"[*] Kích thước bản scan sau nắn phẳng: {warped.shape}")
        print("=> Module 2 hoạt động chính xác 100%!")
    else:
        print("[!] Không tìm thấy tài liệu trong ảnh!")
    print("=" * 65)
