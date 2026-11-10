
"""
DOCUMENT SCANNER - PERSPECTIVE CORRECTION
========================================
TV5 - Desktop Interface with CustomTkinter

Chức năng:
1. Chọn ảnh tài liệu
2. Hiển thị ảnh gốc
3. Phát hiện biên Canny
4. Tự động tìm 4 góc tài liệu
5. Chọn 4 góc thủ công
6. Hiệu chỉnh phối cảnh
7. Xoay ảnh scan trái/phải 90 độ
8. Lưu PNG/JPG
9. Làm mới và thay đổi kích thước giao diện

Cài đặt:
pip install customtkinter opencv-python pillow numpy

Chạy:
python desktop_app.py
"""

import os
import tkinter as tk
import traceback

from tkinter import filedialog, messagebox

import cv2
import numpy as np
import customtkinter as ctk

from PIL import Image, ImageTk, ImageOps

from modules.image_enhancer import enhance_document


# ============================================================
# 1. CẤU HÌNH GIAO DIỆN
# ============================================================

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

APP_TITLE = "Document Scanner - Perspective Correction"

BG = "#181B22"
PANEL = "#242933"
CANVAS_BG = "#151922"
TEXT = "#E8ECF2"
MUTED = "#A0A7B4"

BLUE = "#2563EB"
GREEN = "#15803D"
ORANGE = "#C47A19"
GRAY = "#555B65"
RED = "#F87171"

MAX_PREVIEW = (1800, 1800)


# ============================================================
# 2. CÁC HÀM XỬ LÝ ẢNH
# ============================================================

def order_points(points):
    """
    Chuẩn hóa thứ tự 4 điểm theo chu vi:
    Trên trái -> Trên phải -> Dưới phải -> Dưới trái.

    Lưu ý:
    Đây là thứ tự hình học theo tọa độ ảnh.
    Hướng đọc văn bản có thể cần xoay thêm.
    """

    pts = np.asarray(
        points, dtype=np.float32
    ).reshape(4, 2)

    if len(np.unique(pts, axis=0)) != 4:
        raise ValueError(
            "Bốn điểm không được trùng nhau."
        )

    center = pts.mean(axis=0)

    angles = np.arctan2(
        pts[:, 1] - center[1],
        pts[:, 0] - center[0]
    )

    pts = pts[np.argsort(angles)]

    start = np.argmin(
        pts[:, 0] + pts[:, 1]
    )

    pts = np.roll(
        pts,
        -int(start),
        axis=0
    )

    if not cv2.isContourConvex(
        pts.reshape(4, 1, 2)
    ):
        raise ValueError(
            "Bốn điểm không tạo thành tứ giác lồi."
        )

    if abs(cv2.contourArea(pts)) < 100:
        raise ValueError(
            "Vùng chọn quá nhỏ hoặc không hợp lệ."
        )

    return pts.astype(np.float32)


def warp_document(image, points):
    """
    Hiệu chỉnh phối cảnh tài liệu bằng
    cv2.getPerspectiveTransform và cv2.warpPerspective.
    """

    rect = order_points(points)

    tl, tr, br, bl = rect

    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)

    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)

    width = int(round(max(
        width_top,
        width_bottom
    )))

    height = int(round(max(
        height_left,
        height_right
    )))

    if width < 30 or height < 30:
        raise ValueError(
            "Kích thước tài liệu không hợp lệ."
        )

    destination = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(
        rect,
        destination
    )

    scanned = cv2.warpPerspective(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )

    return scanned


def get_contours(binary):
    """Lấy contours tương thích với OpenCV."""

    result = cv2.findContours(
        binary,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    return result[-2]


def quad_score(points, image_shape):
    """
    Đánh giá tứ giác ứng viên theo diện tích,
    hình dạng và độ hợp lệ.
    """

    try:
        rect = order_points(points)
    except ValueError:
        return -1

    h, w = image_shape[:2]
    image_area = h * w

    area = abs(cv2.contourArea(rect))

    if area < image_area * 0.08:
        return -1

    if area > image_area * 0.98:
        return -1

    tl, tr, br, bl = rect

    width = max(
        np.linalg.norm(tr - tl),
        np.linalg.norm(br - bl)
    )

    height = max(
        np.linalg.norm(bl - tl),
        np.linalg.norm(br - tr)
    )

    if min(width, height) < 30:
        return -1

    ratio = width / max(height, 1)

    if ratio < 0.20 or ratio > 5.0:
        return -1

    for i in range(4):

        a = rect[(i - 1) % 4] - rect[i]
        b = rect[(i + 1) % 4] - rect[i]

        norm = (
            np.linalg.norm(a) *
            np.linalg.norm(b)
        )

        if norm < 1e-8:
            return -1

        cosine = np.dot(a, b) / norm

        if abs(cosine) > 0.97:
            return -1

    return area / image_area


def find_document_quad(binary, shape):
    """
    Tìm tứ giác tài liệu từ các contour.

    Thử nhiều mức epsilon nhằm tăng khả năng
    nhận diện đường viền giấy.
    """

    contours = sorted(
        get_contours(binary),
        key=cv2.contourArea,
        reverse=True
    )[:40]

    best_quad = None
    best_score = -1

    for contour in contours:

        perimeter = cv2.arcLength(
            contour,
            True
        )

        if perimeter <= 0:
            continue

        for epsilon_rate in (
            0.01,
            0.015,
            0.02,
            0.03,
            0.04
        ):

            approx = cv2.approxPolyDP(
                contour,
                epsilon_rate * perimeter,
                True
            )

            if len(approx) != 4:
                continue

            if not cv2.isContourConvex(approx):
                continue

            points = approx.reshape(4, 2)

            score = quad_score(
                points,
                shape
            )

            if score > best_score:
                best_score = score
                best_quad = points.copy()

    return best_quad


def detect_document(image):
    """
    Xử lý ảnh tự động.

    Pipeline:
    Resize -> Grayscale -> Gaussian Blur
    -> Canny -> Morphology -> Contours
    -> Perspective Transform

    Nếu contour Canny thất bại:
    Thử phân đoạn vùng giấy sáng.

    Kết quả:
    canny_full, scanned, points, method
    """

    if image is None or image.size == 0:
        raise ValueError(
            "Ảnh đầu vào không hợp lệ."
        )

    original_h, original_w = image.shape[:2]

    if min(original_h, original_w) < 40:
        raise ValueError(
            "Ảnh quá nhỏ để xử lý."
        )

    # --------------------------------------------------------
    # BƯỚC 1: RESIZE
    # --------------------------------------------------------

    scale = min(
        1.0,
        1100.0 / max(original_h, original_w)
    )

    if scale < 1.0:

        resized = cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA
        )

    else:
        resized = image.copy()

    # --------------------------------------------------------
    # BƯỚC 2: GRAYSCALE
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2GRAY
    )

    # --------------------------------------------------------
    # BƯỚC 3: GAUSSIAN BLUR
    # --------------------------------------------------------

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # --------------------------------------------------------
    # BƯỚC 4: CANNY
    # --------------------------------------------------------

    canny = cv2.Canny(
        blurred,
        threshold1=50,
        threshold2=150
    )

    # --------------------------------------------------------
    # BƯỚC 5: MORPHOLOGY
    # --------------------------------------------------------

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (5, 5)
    )

    closed = cv2.morphologyEx(
        canny,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    detection_edges = cv2.dilate(
        closed,
        np.ones((3, 3), dtype=np.uint8),
        iterations=1
    )

    # --------------------------------------------------------
    # BƯỚC 6: TÌM 4 GÓC BẰNG CANNY
    # --------------------------------------------------------

    points = find_document_quad(
        detection_edges,
        resized.shape
    )

    method = "Canny + Contour"

    # --------------------------------------------------------
    # BƯỚC 7: PHÁT HIỆN VÙNG GIẤY SÁNG DỰ PHÒNG
    # --------------------------------------------------------

    if points is None:

        hsv = cv2.cvtColor(
            resized,
            cv2.COLOR_BGR2HSV
        )

        white_mask = cv2.inRange(
            hsv,
            np.array(
                [0, 0, 125],
                dtype=np.uint8
            ),
            np.array(
                [179, 105, 255],
                dtype=np.uint8
            )
        )

        white_mask = cv2.morphologyEx(
            white_mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (13, 13)
            ),
            iterations=2
        )

        white_mask = cv2.morphologyEx(
            white_mask,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (5, 5)
            )
        )

        points = find_document_quad(
            white_mask,
            resized.shape
        )

        method = "Phân đoạn vùng giấy sáng"

    # --------------------------------------------------------
    # BƯỚC 8: CHUẨN BỊ ẢNH CANNY
    # --------------------------------------------------------

    canny_full = cv2.resize(
        canny,
        (original_w, original_h),
        interpolation=cv2.INTER_NEAREST
    )

    if points is None:
        return canny_full, None, None, None

    # --------------------------------------------------------
    # BƯỚC 9: QUY ĐỔI TỌA ĐỘ VỀ ẢNH GỐC
    # --------------------------------------------------------

    original_points = (
        points.astype(np.float32) / scale
    )

    original_points[:, 0] = np.clip(
        original_points[:, 0],
        0,
        original_w - 1
    )

    original_points[:, 1] = np.clip(
        original_points[:, 1],
        0,
        original_h - 1
    )

    # --------------------------------------------------------
    # BƯỚC 10: PERSPECTIVE TRANSFORM
    # --------------------------------------------------------

    scanned = warp_document(
        image,
        original_points
    )

    return (
        canny_full,
        scanned,
        original_points,
        method
    )


# ============================================================
# 3. CỬA SỔ CHỌN 4 GÓC THỦ CÔNG
# ============================================================

class ManualCornerDialog(ctk.CTkToplevel):

    def __init__(self, master, image, callback):
        super().__init__(master)

        self.title("Chọn 4 góc tài liệu")
        self.geometry("950x750")
        self.minsize(650, 500)

        self.callback = callback
        self.points = []

        h, w = image.shape[:2]

        self.original_h = h
        self.original_w = w

        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        self.source = Image.fromarray(rgb)

        self.photo = None
        self.scale = 1.0

        self.offset_x = 0
        self.offset_y = 0

        self.grid_columnconfigure(
            0, weight=1
        )

        self.grid_rowconfigure(
            1, weight=1
        )

        self.info = ctk.CTkLabel(
            self,
            text=(
                "Chọn lần lượt: Trên trái → "
                "Trên phải → Dưới phải → Dưới trái"
            ),
            font=ctk.CTkFont(
                size=15,
                weight="bold"
            )
        )

        self.info.grid(
            row=0,
            column=0,
            padx=12,
            pady=12
        )

        self.canvas = tk.Canvas(
            self,
            bg=CANVAS_BG,
            highlightthickness=0
        )

        self.canvas.grid(
            row=1,
            column=0,
            padx=15,
            pady=8,
            sticky="nsew"
        )

        self.canvas.bind(
            "<Configure>",
            self.redraw
        )

        self.canvas.bind(
            "<Button-1>",
            self.add_point
        )

        toolbar = ctk.CTkFrame(self)

        toolbar.grid(
            row=2,
            column=0,
            padx=15,
            pady=12,
            sticky="ew"
        )

        ctk.CTkButton(
            toolbar,
            text="Hoàn tác",
            command=self.undo_point
        ).pack(
            side="left",
            padx=8,
            pady=10
        )

        ctk.CTkButton(
            toolbar,
            text="Chọn lại",
            command=self.reset_points
        ).pack(
            side="left",
            padx=8
        )

        ctk.CTkButton(
            toolbar,
            text="Nắn phẳng",
            fg_color=GREEN,
            command=self.apply_points
        ).pack(
            side="right",
            padx=8
        )

        self.transient(master)
        self.focus_force()

    # --------------------------------------------------------
    # HIỂN THỊ ẢNH VÀ CÁC ĐIỂM ĐÃ CHỌN
    # --------------------------------------------------------

    def redraw(self, event=None):

        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()

        if cw < 30 or ch < 30:
            return

        self.scale = min(
            cw / self.original_w,
            ch / self.original_h
        )

        width = max(
            1,
            int(self.original_w * self.scale)
        )

        height = max(
            1,
            int(self.original_h * self.scale)
        )

        self.offset_x = (cw - width) / 2
        self.offset_y = (ch - height) / 2

        preview = self.source.resize(
            (width, height),
            Image.Resampling.LANCZOS
        )

        self.photo = ImageTk.PhotoImage(
            preview,
            master=self
        )

        self.canvas.delete("all")

        self.canvas.create_image(
            self.offset_x,
            self.offset_y,
            image=self.photo,
            anchor="nw"
        )

        screen_points = []

        for index, (x, y) in enumerate(self.points):

            sx = self.offset_x + x * self.scale
            sy = self.offset_y + y * self.scale

            screen_points.append((sx, sy))

            self.canvas.create_oval(
                sx - 6,
                sy - 6,
                sx + 6,
                sy + 6,
                fill="#FF3B30",
                outline="white",
                width=2
            )

            self.canvas.create_text(
                sx + 13,
                sy - 13,
                text=str(index + 1),
                fill="yellow",
                font=("Arial", 13, "bold")
            )

        for i in range(len(screen_points) - 1):

            self.canvas.create_line(
                *screen_points[i],
                *screen_points[i + 1],
                fill="#00FF88",
                width=2
            )

        if len(screen_points) == 4:

            self.canvas.create_line(
                *screen_points[3],
                *screen_points[0],
                fill="#00FF88",
                width=2
            )

    # --------------------------------------------------------
    # CHỌN ĐIỂM
    # --------------------------------------------------------

    def add_point(self, event):

        if len(self.points) >= 4:
            return

        x = (
            event.x - self.offset_x
        ) / self.scale

        y = (
            event.y - self.offset_y
        ) / self.scale

        if not (
            0 <= x < self.original_w
            and 0 <= y < self.original_h
        ):
            return

        self.points.append((x, y))

        self.info.configure(
            text=f"Đã chọn {len(self.points)}/4 góc"
        )

        self.redraw()

    def undo_point(self):

        if self.points:
            self.points.pop()
            self.redraw()

    def reset_points(self):

        self.points.clear()

        self.info.configure(
            text="Chọn lại 4 góc theo thứ tự"
        )

        self.redraw()

    # --------------------------------------------------------
    # ÁP DỤNG 4 ĐIỂM
    # --------------------------------------------------------

    def apply_points(self):

        if len(self.points) != 4:

            messagebox.showwarning(
                "Thiếu điểm",
                "Bạn cần chọn đủ 4 góc tài liệu.",
                parent=self
            )

            return

        try:

            points = order_points(
                np.array(
                    self.points,
                    dtype=np.float32
                )
            )

            # Trả kết quả về ứng dụng chính.
            # Đóng cửa sổ chọn điểm trước.
            callback = self.callback
            self.destroy()

            callback(points)

        except Exception as error:

            if self.winfo_exists():
                messagebox.showerror(
                    "Lỗi chọn góc",
                    str(error),
                    parent=self
                )
            else:
                traceback.print_exc()


# ============================================================
# 4. ỨNG DỤNG DESKTOP CHÍNH
# ============================================================

class DocumentScannerApp(ctk.CTk):

    def __init__(self):
        super().__init__()

        self.title(APP_TITLE)
        self.geometry("1500x820")
        self.minsize(1100, 600)

        self.configure(
            fg_color=BG
        )

        # ----------------------------------------------------
        # ẢNH GỐC VÀ KẾT QUẢ
        # ----------------------------------------------------

        self.original_image = None
        self.canny_image = None
        self.scanned_image = None
        self.enhanced_images = {}

        self.image_path = None

        # ----------------------------------------------------
        # BỘ NHỚ HIỂN THỊ
        # ----------------------------------------------------

        self.canvases = {}
        self.preview_images = {}
        self.photo_refs = {}
        self.resize_jobs = {}

        self.processing = False
        self.is_closing = False
        self.process_job = None

        self.build_ui()
        self.update_buttons()

        self.protocol(
            "WM_DELETE_WINDOW",
            self.on_close
        )

    # ========================================================
    # 5. TẠO GIAO DIỆN CUSTOMTKINTER
    # ========================================================

    def build_ui(self):

        self.grid_columnconfigure(
            0, weight=1
        )

        self.grid_rowconfigure(
            3, weight=1
        )

        # ----------------------------------------------------
        # HEADER
        # ----------------------------------------------------

        header = ctk.CTkFrame(
            self,
            fg_color=PANEL,
            corner_radius=12
        )

        header.grid(
            row=0,
            column=0,
            padx=15,
            pady=(15, 8),
            sticky="ew"
        )

        ctk.CTkLabel(
            header,
            text="DOCUMENT SCANNER",
            font=ctk.CTkFont(
                size=27,
                weight="bold"
            ),
            text_color=TEXT
        ).pack(
            pady=(12, 4)
        )

        ctk.CTkLabel(
            header,
            text=(
                "Document Perspective Correction "
                "| Desktop Application"
            ),
            text_color=MUTED
        ).pack(
            pady=(0, 12)
        )

        # ----------------------------------------------------
        # THANH CÔNG CỤ
        # ----------------------------------------------------

        toolbar = ctk.CTkFrame(
            self,
            fg_color=PANEL
        )

        toolbar.grid(
            row=1,
            column=0,
            padx=15,
            pady=8,
            sticky="ew"
        )

        self.btn_open = ctk.CTkButton(
            toolbar,
            text="Chọn ảnh",
            width=130,
            command=self.choose_image
        )

        self.btn_open.pack(
            side="left",
            padx=(12, 5),
            pady=12
        )

        self.btn_process = ctk.CTkButton(
            toolbar,
            text="Xử lý tự động",
            width=140,
            fg_color=GREEN,
            command=self.process_image
        )

        self.btn_process.pack(
            side="left",
            padx=5
        )

        self.btn_manual = ctk.CTkButton(
            toolbar,
            text="Chọn 4 góc",
            width=130,
            fg_color=ORANGE,
            command=self.manual_select
        )

        self.btn_manual.pack(
            side="left",
            padx=5
        )

        self.btn_rotate_left = ctk.CTkButton(
            toolbar,
            text="Xoay trái 90°",
            width=125,
            fg_color="#506380",
            command=self.rotate_left
        )

        self.btn_rotate_left.pack(
            side="left",
            padx=5
        )

        self.btn_rotate_right = ctk.CTkButton(
            toolbar,
            text="Xoay phải 90°",
            width=125,
            fg_color="#506380",
            command=self.rotate_right
        )

        self.btn_rotate_right.pack(
            side="left",
            padx=5
        )

        self.btn_save = ctk.CTkButton(
            toolbar,
            text="Lưu bản scan",
            width=130,
            command=self.save_image
        )

        self.btn_save.pack(
            side="left",
            padx=5
        )

        self.btn_reset = ctk.CTkButton(
            toolbar,
            text="Làm mới",
            width=105,
            fg_color=GRAY,
            command=self.reset_images
        )

        self.btn_reset.pack(
            side="left",
            padx=5
        )

        # ----------------------------------------------------
        # ĐIỀU KHIỂN NÂNG CAO CHẤT LƯỢNG ẢNH
        # ----------------------------------------------------

        enhancement_toolbar = ctk.CTkFrame(
            self,
            fg_color=PANEL
        )

        enhancement_toolbar.grid(
            row=2,
            column=0,
            padx=15,
            pady=(0, 8),
            sticky="ew"
        )

        self.btn_enhance = ctk.CTkButton(
            enhancement_toolbar,
            text="Nâng cao chất lượng",
            width=165,
            fg_color=GREEN,
            command=self.enhance_image
        )

        self.btn_enhance.pack(
            side="left",
            padx=(12, 8),
            pady=9
        )

        ctk.CTkLabel(
            enhancement_toolbar,
            text="Kết quả:",
            text_color=TEXT
        ).pack(
            side="left",
            padx=(8, 5)
        )

        self.enhancement_selector = ctk.CTkOptionMenu(
            enhancement_toolbar,
            values=[
                "Cân bằng sáng",
                "Làm nét",
                "Nhị phân"
            ],
            command=self.select_enhancement
        )

        self.enhancement_selector.pack(
            side="left",
            padx=5,
            pady=9
        )

        self.btn_save_enhanced = ctk.CTkButton(
            enhancement_toolbar,
            text="Lưu kết quả nâng cao",
            width=175,
            command=self.save_enhanced_image
        )

        self.btn_save_enhanced.pack(
            side="left",
            padx=8
        )

        # ----------------------------------------------------
        # CÁC KHUNG SO SÁNH
        # ----------------------------------------------------

        content = ctk.CTkFrame(
            self,
            fg_color="transparent"
        )

        content.grid(
            row=3,
            column=0,
            padx=15,
            pady=8,
            sticky="nsew"
        )

        content.grid_rowconfigure(
            0, weight=1
        )

        panels = [
            ("original", "01 - ẢNH GỐC"),
            ("canny", "02 - BIÊN CANNY"),
            ("scanned", "03 - BẢN SCAN NẮN PHẲNG"),
            ("enhanced", "04 - KẾT QUẢ NÂNG CAO")
        ]

        for index, (key, title) in enumerate(panels):

            content.grid_columnconfigure(
                index,
                weight=1,
                uniform="panels"
            )

            frame = ctk.CTkFrame(
                content,
                fg_color=PANEL,
                corner_radius=12
            )

            frame.grid(
                row=0,
                column=index,
                padx=6,
                sticky="nsew"
            )

            frame.grid_columnconfigure(
                0, weight=1
            )

            frame.grid_rowconfigure(
                1, weight=1
            )

            ctk.CTkLabel(
                frame,
                text=title,
                font=ctk.CTkFont(
                    size=16,
                    weight="bold"
                ),
                text_color=TEXT
            ).grid(
                row=0,
                column=0,
                pady=14
            )

            # Dùng Canvas thay CTkImage.
            # Giải quyết các lỗi tham chiếu pyimage
            # và light/dark image không cùng size.
            canvas = tk.Canvas(
                frame,
                bg=CANVAS_BG,
                highlightthickness=0,
                borderwidth=0
            )

            canvas.grid(
                row=1,
                column=0,
                padx=10,
                pady=(0, 10),
                sticky="nsew"
            )

            self.canvases[key] = canvas

            canvas.bind(
                "<Configure>",
                lambda event, k=key:
                    self.schedule_refresh(k)
            )

            self.draw_placeholder(key)

        # ----------------------------------------------------
        # THANH TRẠNG THÁI
        # ----------------------------------------------------

        self.status_label = ctk.CTkLabel(
            self,
            text="Trạng thái: Sẵn sàng",
            anchor="w",
            text_color=MUTED,
            font=ctk.CTkFont(size=13)
        )

        self.status_label.grid(
            row=4,
            column=0,
            padx=25,
            pady=(5, 12),
            sticky="ew"
        )

    # ========================================================
    # 6. CẬP NHẬT TRẠNG THÁI
    # ========================================================

    def set_status(self, message, error=False):

        self.status_label.configure(
            text=f"Trạng thái: {message}",
            text_color=RED if error else MUTED
        )

    def update_buttons(self):

        buttons = (
            self.btn_open,
            self.btn_process,
            self.btn_manual,
            self.btn_rotate_left,
            self.btn_rotate_right,
            self.btn_save,
            self.btn_enhance,
            self.enhancement_selector,
            self.btn_save_enhanced,
            self.btn_reset
        )

        if self.processing:

            for button in buttons:
                button.configure(state="disabled")

            return

        has_original = self.original_image is not None
        has_scan = self.scanned_image is not None
        has_enhancement = bool(self.enhanced_images)

        self.btn_open.configure(
            state="normal"
        )

        self.btn_reset.configure(
            state="normal"
        )

        self.btn_process.configure(
            state="normal" if has_original else "disabled"
        )

        self.btn_manual.configure(
            state="normal" if has_original else "disabled"
        )

        for button in (
            self.btn_rotate_left,
            self.btn_rotate_right,
            self.btn_save
        ):
            button.configure(
                state="normal" if has_scan else "disabled"
            )

        self.btn_enhance.configure(
            state="normal" if has_scan else "disabled"
        )

        self.enhancement_selector.configure(
            state="normal" if has_enhancement else "disabled"
        )

        self.btn_save_enhanced.configure(
            state="normal" if has_enhancement else "disabled"
        )

    # ========================================================
    # 7. CHỌN ẢNH
    # ========================================================

    def choose_image(self):

        if self.processing:
            return

        path = filedialog.askopenfilename(
            parent=self,
            title="Chọn ảnh tài liệu",
            filetypes=[
                (
                    "Image files",
                    "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"
                ),
                ("All files", "*.*")
            ]
        )

        if not path:
            return

        try:

            with Image.open(path) as opened:

                pil_image = ImageOps.exif_transpose(
                    opened
                ).convert("RGB")

            rgb = np.array(pil_image)

            self.original_image = cv2.cvtColor(
                rgb,
                cv2.COLOR_RGB2BGR
            )

            self.image_path = path

            self.canny_image = None
            self.scanned_image = None
            self.enhanced_images.clear()

            self.clear_panel("canny")
            self.clear_panel("scanned")
            self.clear_panel("enhanced")

            self.display_image(
                "original",
                self.original_image
            )

            self.set_status(
                f"Đã tải ảnh: {os.path.basename(path)}"
            )

            self.update_buttons()

        except Exception as error:

            traceback.print_exc()

            messagebox.showerror(
                "Lỗi mở ảnh",
                str(error),
                parent=self
            )

            self.set_status(
                "Không thể mở ảnh.",
                error=True
            )

    # ========================================================
    # 8. XỬ LÝ TỰ ĐỘNG
    # ========================================================

    def process_image(self):

        if self.original_image is None:
            return

        if self.processing:
            return

        self.processing = True
        self.update_buttons()

        self.set_status(
            "Đang xử lý ảnh tài liệu..."
        )

        self.process_job = self.after(
            40,
            self.run_processing
        )

    def run_processing(self):

        self.process_job = None

        try:

            (
                canny,
                scanned,
                points,
                method
            ) = detect_document(
                self.original_image
            )

            self.canny_image = canny
            self.scanned_image = scanned
            self.enhanced_images.clear()
            self.clear_panel("enhanced")

            self.display_image(
                "canny",
                self.canny_image
            )

            if scanned is None:

                self.clear_panel("scanned")

                self.set_status(
                    "Không tìm thấy 4 góc. "
                    "Hãy chọn 4 góc thủ công.",
                    error=True
                )

                messagebox.showwarning(
                    "Không tìm thấy tài liệu",
                    (
                        "Thuật toán chưa tìm được "
                        "4 góc tài liệu.\n\n"
                        "Bạn hãy sử dụng nút "
                        "'Chọn 4 góc' để nắn phẳng."
                    ),
                    parent=self
                )

                return

            self.display_image(
                "scanned",
                self.scanned_image
            )

            self.set_status(
                f"Xử lý thành công bằng {method}. "
                "Có thể xoay ảnh nếu chưa đúng chiều."
            )

        except Exception as error:

            traceback.print_exc()

            self.canny_image = None
            self.scanned_image = None
            self.enhanced_images.clear()

            self.clear_panel("canny")
            self.clear_panel("scanned")
            self.clear_panel("enhanced")

            self.set_status(
                "Xử lý ảnh thất bại.",
                error=True
            )

            messagebox.showerror(
                "Lỗi xử lý ảnh",
                str(error),
                parent=self
            )

        finally:

            self.processing = False
            self.update_buttons()

    # ========================================================
    # 9. CHỌN 4 GÓC THỦ CÔNG
    # ========================================================

    def manual_select(self):

        if self.original_image is None:
            return

        ManualCornerDialog(
            self,
            self.original_image,
            self.apply_manual_points
        )

    def apply_manual_points(self, points):

        try:

            scanned = warp_document(
                self.original_image,
                points
            )

            self.scanned_image = scanned
            self.enhanced_images.clear()
            self.clear_panel("enhanced")

            self.display_image(
                "scanned",
                self.scanned_image
            )

            # Tạo biên Canny nếu chưa xử lý tự động.
            if self.canny_image is None:

                gray = cv2.cvtColor(
                    self.original_image,
                    cv2.COLOR_BGR2GRAY
                )

                blurred = cv2.GaussianBlur(
                    gray,
                    (5, 5),
                    0
                )

                self.canny_image = cv2.Canny(
                    blurred,
                    50,
                    150
                )

                self.display_image(
                    "canny",
                    self.canny_image
                )

            self.set_status(
                "Đã nắn phẳng bằng 4 góc thủ công. "
                "Có thể xoay ảnh trái/phải hoặc nâng cao chất lượng."
            )

            self.update_buttons()

        except Exception as error:

            traceback.print_exc()

            self.set_status(
                "Nắn phẳng thủ công thất bại.",
                error=True
            )

            messagebox.showerror(
                "Lỗi nắn phẳng",
                str(error),
                parent=self
            )

    # ========================================================
    # 10. XOAY ẢNH SCAN
    # ========================================================

    def clear_enhancement(self):
        self.enhanced_images.clear()
        self.clear_panel("enhanced")
        self.update_buttons()

    def rotate_left(self):
        """
        Xoay bản scan 90 độ ngược chiều kim đồng hồ.
        """

        if self.scanned_image is None:
            return

        self.scanned_image = cv2.rotate(
            self.scanned_image,
            cv2.ROTATE_90_COUNTERCLOCKWISE
        )
        self.clear_enhancement()

        self.display_image(
            "scanned",
            self.scanned_image
        )

        self.set_status(
            "Đã xoay bản scan sang trái 90°."
        )

    def rotate_right(self):
        """
        Xoay bản scan 90 độ theo chiều kim đồng hồ.
        """

        if self.scanned_image is None:
            return

        self.scanned_image = cv2.rotate(
            self.scanned_image,
            cv2.ROTATE_90_CLOCKWISE
        )
        self.clear_enhancement()

        self.display_image(
            "scanned",
            self.scanned_image
        )

        self.set_status(
            "Đã xoay bản scan sang phải 90°."
        )

    # ========================================================
    # 11. NÂNG CAO CHẤT LƯỢNG ẢNH (MODULE 4)
    # ========================================================

    def enhance_image(self):

        if self.scanned_image is None or self.processing:
            return

        self.processing = True
        self.update_buttons()
        self.set_status(
            "Đang cân bằng sáng, làm nét và nhị phân hóa..."
        )

        self.process_job = self.after(
            40,
            self.run_enhancement
        )

    def run_enhancement(self):

        self.process_job = None

        try:
            corrected, sharpened, binary = enhance_document(
                self.scanned_image,
                sigma=25.0,
                sharpen_strength=1.2,
                block_size=31,
                C=10.0
            )

            self.enhanced_images = {
                "Cân bằng sáng": corrected,
                "Làm nét": sharpened,
                "Nhị phân": binary
            }

            self.select_enhancement(
                self.enhancement_selector.get()
            )

            self.set_status(
                "Đã tăng cường chất lượng ảnh. "
                "Chọn kết quả muốn xem hoặc lưu."
            )

        except Exception as error:

            traceback.print_exc()

            self.set_status(
                "Tăng cường chất lượng ảnh thất bại.",
                error=True
            )

            messagebox.showerror(
                "Lỗi nâng cao ảnh",
                str(error),
                parent=self
            )

        finally:
            self.processing = False
            self.update_buttons()

    def select_enhancement(self, selection):

        image = self.enhanced_images.get(selection)

        if image is not None:
            self.display_image("enhanced", image)

    # ========================================================
    # 12. HIỂN THỊ ẢNH LÊN CANVAS
    # ========================================================

    def display_image(self, key, image):

        if image is None:
            self.clear_panel(key)
            return

        if image.ndim == 2:

            pil = Image.fromarray(
                image
            ).convert("RGB")

        elif image.ndim == 3 and image.shape[2] == 3:

            rgb = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )

            pil = Image.fromarray(rgb)

        elif image.ndim == 3 and image.shape[2] == 4:

            rgba = cv2.cvtColor(
                image,
                cv2.COLOR_BGRA2RGBA
            )

            pil = Image.fromarray(rgba)

        else:

            raise ValueError(
                "Định dạng ảnh không hỗ trợ."
            )

        # Chỉ thu nhỏ bản xem trước.
        # Ảnh gốc và ảnh lưu không bị thay đổi.
        pil.thumbnail(
            MAX_PREVIEW,
            Image.Resampling.LANCZOS
        )

        self.preview_images[key] = pil

        self.schedule_refresh(key)

    def schedule_refresh(self, key):

        if self.is_closing:
            return

        previous_job = self.resize_jobs.pop(
            key,
            None
        )

        if previous_job is not None:

            try:
                self.after_cancel(previous_job)

            except tk.TclError:
                pass

        self.resize_jobs[key] = self.after(
            80,
            lambda k=key: self.refresh_image(k)
        )

    def refresh_image(self, key):

        self.resize_jobs.pop(key, None)

        if self.is_closing:
            return

        if key not in self.preview_images:
            self.draw_placeholder(key)
            return

        canvas = self.canvases[key]

        if not canvas.winfo_exists():
            return

        cw = canvas.winfo_width()
        ch = canvas.winfo_height()

        if cw < 30 or ch < 30:
            return

        preview = self.preview_images[key].copy()

        preview.thumbnail(
            (
                max(cw - 20, 1),
                max(ch - 20, 1)
            ),
            Image.Resampling.LANCZOS
        )

        # Tạo PhotoImage gắn với Tk master hiện tại.
        photo = ImageTk.PhotoImage(
            preview,
            master=self
        )

        # Xóa hình cũ trước khi thay tham chiếu.
        canvas.delete("all")

        canvas.create_image(
            cw // 2,
            ch // 2,
            image=photo,
            anchor="center"
        )

        # Giữ đối tượng ảnh trong bộ nhớ.
        self.photo_refs[key] = photo

    # ========================================================
    # 13. PLACEHOLDER VÀ XÓA ẢNH
    # ========================================================

    def draw_placeholder(self, key):

        canvas = self.canvases[key]

        if not canvas.winfo_exists():
            return

        canvas.delete("all")

        cw = max(
            canvas.winfo_width(),
            100
        )

        ch = max(
            canvas.winfo_height(),
            100
        )

        canvas.create_text(
            cw // 2,
            ch // 2,
            text="Chưa có ảnh",
            fill=MUTED,
            font=("Arial", 12)
        )

    def clear_panel(self, key):

        job = self.resize_jobs.pop(
            key,
            None
        )

        if job is not None:

            try:
                self.after_cancel(job)

            except tk.TclError:
                pass

        self.preview_images.pop(
            key,
            None
        )

        canvas = self.canvases[key]

        # Xóa khỏi Canvas trước khi giải phóng ảnh.
        canvas.delete("all")

        self.photo_refs.pop(
            key,
            None
        )

        self.draw_placeholder(key)

    # ========================================================
    # 14. LƯU ẢNH SCAN
    # ========================================================

    def save_image(self):

        if self.scanned_image is None:

            messagebox.showwarning(
                "Thông báo",
                "Chưa có bản scan để lưu.",
                parent=self
            )

            return

        path = filedialog.asksaveasfilename(
            parent=self,
            title="Lưu bản scan",
            defaultextension=".png",
            initialfile="document_scan.png",
            filetypes=[
                ("PNG Image", "*.png"),
                ("JPEG Image", "*.jpg")
            ]
        )

        if not path:
            return

        try:

            extension = os.path.splitext(
                path
            )[1].lower()

            if extension not in (
                ".png",
                ".jpg",
                ".jpeg"
            ):

                raise ValueError(
                    "Định dạng lưu không hợp lệ."
                )

            # Ảnh được lưu đúng theo hướng sau khi xoay.
            success, encoded = cv2.imencode(
                extension,
                self.scanned_image
            )

            if not success:

                raise ValueError(
                    "Không thể mã hóa bản scan."
                )

            # Hỗ trợ đường dẫn Unicode trên Windows.
            encoded.tofile(path)

            self.set_status(
                f"Đã lưu bản scan: {path}"
            )

            messagebox.showinfo(
                "Thành công",
                (
                    "Đã lưu bản scan thành công!\n\n"
                    + path
                ),
                parent=self
            )

        except Exception as error:

            traceback.print_exc()

            self.set_status(
                "Lưu bản scan thất bại.",
                error=True
            )

            messagebox.showerror(
                "Lỗi lưu ảnh",
                str(error),
                parent=self
            )

    def save_enhanced_image(self):

        selection = self.enhancement_selector.get()
        image = self.enhanced_images.get(selection)

        if image is None:
            messagebox.showwarning(
                "Thông báo",
                "Chưa có kết quả nâng cao để lưu.",
                parent=self
            )
            return

        file_stem = selection.lower().replace(" ", "_")
        path = filedialog.asksaveasfilename(
            parent=self,
            title="Lưu kết quả nâng cao",
            defaultextension=".png",
            initialfile=f"document_{file_stem}.png",
            filetypes=[
                ("PNG Image", "*.png"),
                ("JPEG Image", "*.jpg")
            ]
        )

        if not path:
            return

        try:
            extension = os.path.splitext(path)[1].lower()

            if extension not in (".png", ".jpg", ".jpeg"):
                raise ValueError("Định dạng lưu không hợp lệ.")

            success, encoded = cv2.imencode(extension, image)

            if not success:
                raise ValueError("Không thể mã hóa kết quả nâng cao.")

            encoded.tofile(path)
            self.set_status(f"Đã lưu kết quả nâng cao: {path}")

            messagebox.showinfo(
                "Thành công",
                f"Đã lưu kết quả nâng cao ({selection}) thành công!\n\n{path}",
                parent=self
            )

        except Exception as error:
            traceback.print_exc()
            self.set_status("Lưu kết quả nâng cao thất bại.", error=True)

            messagebox.showerror(
                "Lỗi lưu ảnh",
                str(error),
                parent=self
            )

    # ========================================================
    # 15. LÀM MỚI ỨNG DỤNG
    # ========================================================

    def reset_images(self):

        if self.processing:
            return

        self.original_image = None
        self.canny_image = None
        self.scanned_image = None
        self.enhanced_images.clear()

        self.image_path = None

        for key in self.canvases:
            self.clear_panel(key)

        self.set_status(
            "Đã làm mới ứng dụng."
        )

        self.update_buttons()

    # ========================================================
    # 16. ĐÓNG ỨNG DỤNG
    # ========================================================

    def on_close(self):

        self.is_closing = True

        if self.process_job is not None:

            try:
                self.after_cancel(self.process_job)

            except tk.TclError:
                pass

        for job in list(
            self.resize_jobs.values()
        ):

            try:
                self.after_cancel(job)

            except tk.TclError:
                pass

        self.resize_jobs.clear()

        for canvas in self.canvases.values():
            canvas.delete("all")

        self.preview_images.clear()
        self.photo_refs.clear()

        self.destroy()


# ============================================================
# 17. KHỞI CHẠY CHƯƠNG TRÌNH
# ============================================================

def main():

    app = DocumentScannerApp()
    app.mainloop()


if __name__ == "__main__":
    main()
