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
def get_paper_contour(edged, min_area_ratio=0.10, top_k=5):
    """
    Tìm contour lớn nhất và xấp xỉ về đúng 4 đỉnh bằng cv2.approxPolyDP.

    Các điểm làm cho hàm chạy ổn định với ảnh thật:
      1. MORPH_CLOSE : nối các đoạn viền Canny bị đứt.
      2. min_area_ratio : bỏ contour nhỏ hơn 10% diện tích ảnh (phím, cốc, ...)
         -> không bao giờ chọn nhầm vật nhỏ có hình tứ giác.
      3. convexHull : làm viền lồi, bỏ các chỗ lõm/răng cưa.
      4. Thử nhiều epsilon (2% -> 6% chu vi) cho tới khi ra đúng 4 đỉnh.
      5. Fallback minAreaRect nếu vẫn không ra 4 đỉnh.

    Trả về: mảng float32 shape (4, 2), hoặc None nếu không tìm được.
    """
    h, w = edged.shape[:2]
    min_area = min_area_ratio * h * w

    k = max(7, int(0.02 * max(h, w))) | 1          # kernel lẻ, tỉ lệ theo cỡ ảnh
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    # Dùng diện tích của hull lồi để lọc/xếp hạng: viền bị hở vẫn có hull
    # bao trọn tờ giấy, còn vật nhỏ (phím, cốc, ...) bị loại vì quá bé.
    hulls = [cv2.convexHull(c) for c in contours]
    hulls = [hl for hl in hulls if cv2.contourArea(hl) >= min_area]
    if not hulls:
        return None
    hulls = sorted(hulls, key=cv2.contourArea, reverse=True)[:top_k]

    for hull in hulls:
        peri = cv2.arcLength(hull, True)
        for eps in (0.02, 0.03, 0.04, 0.05, 0.06):
            approx = cv2.approxPolyDP(hull, eps * peri, True)
            if len(approx) == 4:
                return approx.reshape(4, 2).astype("float32")

    # Fallback: hình chữ nhật xoay nhỏ nhất bao contour lớn nhất
    rect = cv2.minAreaRect(hulls[0])
    return cv2.boxPoints(rect).astype("float32")


# --------------------------------------------------------------------------
# 2. Sắp xếp 4 đỉnh
# --------------------------------------------------------------------------
def order_points(pts):
    """
    Sắp xếp 4 đỉnh theo thứ tự: Trên-Trái, Trên-Phải, Dưới-Phải, Dưới-Trái.

    Dựa vào tổng s = x + y và hiệu d = x - y  (gốc tọa độ ở góc trên-trái ảnh,
    trục y hướng xuống):
        Trên-Trái  : s nhỏ nhất   (x, y đều nhỏ)
        Dưới-Phải  : s lớn nhất   (x, y đều lớn)
        Trên-Phải  : d lớn nhất   (x lớn, y nhỏ)
        Dưới-Trái  : d nhỏ nhất   (x nhỏ, y lớn)
    """
    pts = np.asarray(pts, dtype="float32").reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")

    s = pts[:, 0] + pts[:, 1]          # x + y
    d = pts[:, 0] - pts[:, 1]          # x - y

    rect[0] = pts[np.argmin(s)]        # Trên-Trái
    rect[2] = pts[np.argmax(s)]        # Dưới-Phải
    rect[1] = pts[np.argmax(d)]        # Trên-Phải
    rect[3] = pts[np.argmin(d)]        # Dưới-Trái
    return rect


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

    # Tiền xử lý tạm thời bằng Canny
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 50, 150)
    edged = cv2.dilate(edged, np.ones((3, 3), np.uint8), iterations=1)

    pts = get_paper_contour(edged)
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
