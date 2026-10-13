"""
perspective_warp.py
Nắn phẳng tài liệu chụp nghiêng bằng Perspective Transform.

Gồm 3 hàm:
    - get_paper_contour(edged)        : tìm contour lớn nhất, lấy đúng 4 đỉnh
    - order_points(pts)               : sắp xếp 4 đỉnh TL, TR, BR, BL
    - four_point_transform(img, pts)  : nắn phẳng ảnh

Chạy test độc lập:
    python perspective_warp.py duong_dan_anh.jpg
    python perspective_warp.py            # không truyền ảnh -> tự sinh ảnh giả lập
"""
import sys

import cv2
import numpy as np


# --------------------------------------------------------------------------
# 1. Tìm contour tờ giấy
# --------------------------------------------------------------------------
def _quad_from_contour(c, epsilons=(0.02, 0.03, 0.04, 0.05)):
    """Xấp xỉ contour về tứ giác LỒI; trả None nếu không ra đúng 4 đỉnh."""
    peri = cv2.arcLength(c, True)
    for eps in epsilons:
        approx = cv2.approxPolyDP(c, eps * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return approx.reshape(4, 2).astype("float32")
    return None


def get_paper_contour(edged, min_area_ratio=0.10, max_area_ratio=0.95, top_k=10):
    """
    Tìm contour lớn nhất và xấp xỉ về đúng 4 đỉnh bằng cv2.approxPolyDP.

    Cách làm:
      Bước 1 (chính): lấy các contour THẬT, lọc theo diện tích
        (>= min_area_ratio và <= max_area_ratio diện tích ảnh -> bỏ vật nhỏ
        và bỏ cả khung viền ảnh), xét từ lớn đến nhỏ, lấy contour đầu tiên
        xấp xỉ ra tứ giác lồi.
      Bước 2 (dự phòng): nếu viền giấy bị đứt nên bước 1 thất bại, nối viền
        bằng MORPH_CLOSE rồi lấy convex hull. Bước này dễ bị nhiễu nền
        (vân gỗ, ...) nên chỉ dùng khi bước 1 không ra kết quả.

    Trả về: mảng float32 shape (4, 2), hoặc None nếu không tìm được.
    """
    h, w = edged.shape[:2]
    min_area, max_area = min_area_ratio * h * w, max_area_ratio * h * w

    # ---- Bước 1: contour thật ------------------------------------------
    thick = cv2.dilate(edged, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(thick, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cands = [c for c in contours if min_area <= cv2.contourArea(c) <= max_area]
    cands = sorted(cands, key=cv2.contourArea, reverse=True)[:top_k]
    for c in cands:
        quad = _quad_from_contour(c)
        if quad is not None:
            return quad

    # ---- Bước 2: dự phòng cho viền bị đứt ---------------------------------
    k = max(7, int(0.02 * max(h, w))) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hulls = [cv2.convexHull(c) for c in contours]
    hulls = [hl for hl in hulls if min_area <= cv2.contourArea(hl) <= max_area]
    hulls = sorted(hulls, key=cv2.contourArea, reverse=True)[:top_k]
    for hull in hulls:
        quad = _quad_from_contour(hull, (0.02, 0.03, 0.04, 0.05, 0.06))
        if quad is not None:
            return quad
    if hulls:
        return cv2.boxPoints(cv2.minAreaRect(hulls[0])).astype("float32")
    return None


# --------------------------------------------------------------------------
# 2. Sắp xếp 4 đỉnh
# --------------------------------------------------------------------------
def order_points(pts):
    """
    Sắp xếp 4 đỉnh theo thứ tự: Trên-Trái, Trên-Phải, Dưới-Phải, Dưới-Trái.

    Cách chính - dựa vào tổng s = x + y và hiệu d = x - y (gốc tọa độ ở góc
    trên-trái ảnh, trục y hướng xuống):
        Trên-Trái  : s nhỏ nhất   (x, y đều nhỏ)
        Dưới-Phải  : s lớn nhất   (x, y đều lớn)
        Trên-Phải  : d lớn nhất   (x lớn, y nhỏ)
        Dưới-Trái  : d nhỏ nhất   (x nhỏ, y lớn)

    Giới hạn: khi tờ giấy xoay nhiều (~30-45 độ), một đỉnh có thể đồng thời
    là min của s và min của d (hoặc max), khiến 2 vai trò chọn trùng 1 đỉnh.
    Khi đó dùng cách dự phòng: sắp xếp 4 đỉnh theo góc quanh tâm (theo chiều
    kim đồng hồ), rồi chọn đỉnh bắt đầu sao cho cạnh trên gần nằm ngang nhất.
    """
    pts = np.asarray(pts, dtype="float32").reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")

    s = pts[:, 0] + pts[:, 1]          # x + y
    d = pts[:, 0] - pts[:, 1]          # x - y
    idx = [np.argmin(s), np.argmax(d), np.argmax(s), np.argmin(d)]  # TL TR BR BL

    if len(set(idx)) == 4:             # trường hợp bình thường
        for i, j in enumerate(idx):
            rect[i] = pts[j]
        return rect

    # --- Dự phòng: sắp theo góc quanh tâm ---
    cx, cy = pts.mean(axis=0)
    cw = pts[np.argsort(np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx))]  # chiều kim đồng hồ
    best, best_ang = 0, 1e9
    for i in range(4):
        v = cw[(i + 1) % 4] - cw[i]    # cạnh "trên" nếu bắt đầu từ đỉnh i
        if v[0] <= 0:                  # cạnh trên phải đi từ trái sang phải
            continue
        ang = abs(np.arctan2(v[1], v[0]))
        if ang < best_ang:
            best, best_ang = i, ang
    return np.roll(cw, -best, axis=0).astype("float32")


# --------------------------------------------------------------------------
# 3. Nắn phẳng
# --------------------------------------------------------------------------
def four_point_transform(image, pts):
    """
    Nắn phẳng vùng tài liệu giới hạn bởi 4 đỉnh `pts` thành ảnh chữ nhật nhìn thẳng.
    """
    rect = order_points(pts)
    tl, tr, br, bl = rect

    # Kích thước ảnh đích = cạnh dài hơn trong mỗi cặp cạnh đối diện
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    max_w = int(round(max(width_top, width_bottom)))

    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    max_h = int(round(max(height_left, height_right)))

    dst = np.array([[0, 0],
                    [max_w - 1, 0],
                    [max_w - 1, max_h - 1],
                    [0, max_h - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)     # ma trận homography 3x3
    # warpPerspective bên trong dùng inverse mapping + nội suy song tuyến
    return cv2.warpPerspective(image, M, (max_w, max_h), flags=cv2.INTER_LINEAR)


# --------------------------------------------------------------------------
# Test độc lập (dùng Canny tạm thời, chưa cần bước tiền xử lý của TV1)
# --------------------------------------------------------------------------
def _make_synthetic_photo():
    """Sinh ảnh 'tài liệu chụp nghiêng' giả lập để test khi không có ảnh thật."""
    doc = np.full((700, 500, 3), 255, np.uint8)
    cv2.rectangle(doc, (0, 0), (499, 699), (0, 0, 0), 3)
    for i, y in enumerate(range(60, 650, 40)):
        cv2.putText(doc, f"Dong van ban so {i + 1}", (30, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2)

    src = np.float32([[0, 0], [499, 0], [499, 699], [0, 699]])
    dst = np.float32([[180, 90], [760, 140], [700, 640], [120, 560]])
    M = cv2.getPerspectiveTransform(src, dst)

    bg = np.full((720, 900, 3), (70, 90, 60), np.uint8)       # nền xanh tối
    warped = cv2.warpPerspective(doc, M, (900, 720))
    mask = cv2.warpPerspective(np.full(doc.shape[:2], 255, np.uint8), M, (900, 720))
    bg[mask > 0] = warped[mask > 0]
    return bg


def mask_edges(image):
    """Tách tờ giấy theo độ sáng (Otsu) rồi lấy biên -> loại bỏ vân nền (gỗ, ...)."""
    gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (9, 9), 0)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return cv2.Canny(mask, 50, 150)


def canny_edges(image):
    gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    return cv2.dilate(cv2.Canny(gray, 50, 150), np.ones((3, 3), np.uint8))


def main():
    if len(sys.argv) > 1:
        image = cv2.imread(sys.argv[1])
        if image is None:
            sys.exit(f"Không đọc được ảnh: {sys.argv[1]}")
    else:
        print("Không truyền ảnh -> dùng ảnh giả lập.")
        image = _make_synthetic_photo()

    # Thu nhỏ để xử lý nhanh & ổn định, nhớ tỉ lệ để map đỉnh về ảnh gốc
    ratio = image.shape[0] / 500.0
    small = cv2.resize(image, (int(image.shape[1] / ratio), 500))

    # Tiền xử lý tạm thời (TV1 sẽ thay): thử tách giấy theo độ sáng trước,
    # nếu không ra thì dùng Canny thường.
    pts, edged = None, None
    for make_edges in (mask_edges, canny_edges):
        edged = make_edges(small)
        pts = get_paper_contour(edged)
        if pts is not None:
            break
    if pts is None:
        sys.exit("Không tìm thấy contour tờ giấy.")

    pts_full = pts * ratio                       # đưa về tọa độ ảnh gốc
    print("4 đỉnh (chưa sắp xếp):\n", pts_full)
    print("4 đỉnh (TL, TR, BR, BL):\n", order_points(pts_full))

    result = four_point_transform(image, pts_full)

    vis = image.copy()
    cv2.polylines(vis, [order_points(pts_full).astype(np.int32)], True,
                  (0, 0, 255), 3)
    cv2.imwrite("out_1_contour.jpg", vis)
    cv2.imwrite("out_2_edged.jpg", edged)
    cv2.imwrite("out_3_warped.jpg", result)
    print("Đã lưu: out_1_contour.jpg, out_2_edged.jpg, out_3_warped.jpg")
    print("Kích thước ảnh nắn phẳng:", result.shape[:2][::-1])


if __name__ == "__main__":
    main()
