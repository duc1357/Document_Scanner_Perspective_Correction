
"""
DOCUMENT SCANNER - PERSPECTIVE CORRECTION
TV5 - DESKTOP APPLICATION - WEEK 5

Features:
- CustomTkinter desktop interface with 4 image panels
- Module 1: Canny Edge Detection + interactive sliders
- Module 2: Automatic/manual Perspective Correction
- Module 3: Orientation Correction
- Module 4: Image Enhancement
- Rotate left/right
- Save PNG/JPG
- Export PDF: A4, A3, Letter
- Reset and responsive image previews

Required:
    pip install customtkinter opencv-python pillow numpy reportlab

Run:
    python desktop_app.py
"""

import io
import os
import traceback
import tkinter as tk

from tkinter import filedialog, messagebox

import cv2
import numpy as np
import customtkinter as ctk

from PIL import Image, ImageTk, ImageOps

from reportlab.pdfgen import canvas as pdf_canvas
from reportlab.lib.pagesizes import A4, A3, letter
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader

# ============================================================
# IMPORT FOUR OFFICIAL GROUP MODULES
# ============================================================

from modules.edge_detector import (
    preprocess_and_detect_edges
)

from modules.perspective_warp import (
    get_paper_contour,
    four_point_transform,
    order_points
)

from modules.orientation_corrector import (
    OrientationCorrector
)

from modules.image_enhancer import (
    enhance_document
)


# ============================================================
# APPLICATION SETTINGS
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
PURPLE = "#7C3AED"
RED = "#F87171"

MAX_PREVIEW = (1800, 1800)


# ============================================================
# IMAGE HELPERS
# ============================================================

def numpy_to_pil(image):
    """Convert grayscale/BGR/BGRA NumPy image to RGB PIL."""

    if image is None:
        raise ValueError("Không có ảnh để chuyển đổi.")

    if image.ndim == 2:
        return Image.fromarray(image).convert("RGB")

    if image.ndim == 3 and image.shape[2] == 3:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    if image.ndim == 3 and image.shape[2] == 4:
        rgba = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        return Image.fromarray(rgba).convert("RGB")

    raise ValueError("Định dạng ảnh không được hỗ trợ.")


def save_numpy_image(image, path):
    """Save PNG/JPG, including Unicode paths on Windows."""

    extension = os.path.splitext(path)[1].lower()

    if extension not in (".png", ".jpg", ".jpeg"):
        raise ValueError("Chỉ hỗ trợ PNG hoặc JPG.")

    success, encoded = cv2.imencode(extension, image)

    if not success:
        raise IOError("Không thể mã hóa ảnh.")

    encoded.tofile(path)



def detect_document(image, low=50, high=150):
    """
    Improved automatic document detection.

    Module 1:
        preprocess_and_detect_edges()

    Module 2:
        four_point_transform()

    Approach:
        1. Detect Canny edges
        2. Segment bright paper
        3. Generate contour candidates
        4. Score valid quadrilaterals
        5. Reject unreliable detections
        6. Warp the best candidate
    """

    if image is None or image.size == 0:
        raise ValueError("Ảnh đầu vào không hợp lệ.")

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Yêu cầu ảnh màu BGR.")

    original_h, original_w = image.shape[:2]

    if min(original_h, original_w) < 40:
        raise ValueError("Ảnh tài liệu quá nhỏ.")

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

    h, w = resized.shape[:2]
    image_area = float(h * w)

    # ==============================================
    # MODULE 1 - CANNY EDGE DETECTION
    # ==============================================

    _, raw_edges, connected_edges = (
        preprocess_and_detect_edges(
            resized,
            blur_kernel=(5, 5),
            low_threshold=int(low),
            high_threshold=int(high),
            use_auto_canny=False,
            apply_morphology=True
        )
    )

    canny_full = cv2.resize(
        raw_edges,
        (original_w, original_h),
        interpolation=cv2.INTER_NEAREST
    )

    # ==============================================
    # SEGMENT BRIGHT DOCUMENT REGIONS
    # ==============================================

    hsv = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2HSV
    )

    paper_mask = cv2.inRange(
        hsv,
        np.array([0, 0, 120], dtype=np.uint8),
        np.array([179, 105, 255], dtype=np.uint8)
    )

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (15, 15)
    )

    paper_mask = cv2.morphologyEx(
        paper_mask,
        cv2.MORPH_CLOSE,
        close_kernel,
        iterations=2
    )

    paper_mask = cv2.morphologyEx(
        paper_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (5, 5)
        )
    )

    # Otsu provides a second segmentation method.
    gray = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2GRAY
    )

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    _, otsu_mask = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    otsu_mask = cv2.morphologyEx(
        otsu_mask,
        cv2.MORPH_CLOSE,
        close_kernel,
        iterations=2
    )

    masks = [
        ("Vùng giấy sáng", paper_mask),
        ("Phân ngưỡng Otsu", otsu_mask),
        ("Canny + Morphology", connected_edges)
    ]

    candidates = []

    # ==============================================
    # SCORE FOUR-CORNER CANDIDATES
    # ==============================================

    def evaluate_candidate(points, contour, method):
        try:
            quad = order_points(
                np.asarray(
                    points,
                    dtype=np.float32
                ).reshape(4, 2)
            )

            if len(np.unique(quad, axis=0)) != 4:
                return

            if not cv2.isContourConvex(
                quad.reshape(4, 1, 2)
            ):
                return

            area = abs(cv2.contourArea(quad))
            area_ratio = area / image_area

            # Reject small internal text regions
            if not (0.18 <= area_ratio <= 0.95):
                return

            tl, tr, br, bl = quad

            width = max(
                np.linalg.norm(tr - tl),
                np.linalg.norm(br - bl)
            )

            height = max(
                np.linalg.norm(bl - tl),
                np.linalg.norm(br - tr)
            )

            if min(width, height) < 0.25 * min(h, w):
                return

            aspect = width / max(height, 1.0)

            # Allow portrait and landscape documents,
            # but reject very thin strips.
            if not (0.35 <= aspect <= 2.85):
                return

            # Check whether the contour resembles
            # the proposed rectangular region.
            contour_area = abs(
                cv2.contourArea(contour)
            )

            fill_ratio = min(
                contour_area / max(area, 1.0),
                1.0
            )

            if fill_ratio < 0.70:
                return

            # Measure paper-like appearance inside quad.
            region = np.zeros(
                (h, w),
                dtype=np.uint8
            )

            cv2.fillConvexPoly(
                region,
                quad.astype(np.int32),
                255
            )

            inside_pixels = region > 0

            if not np.any(inside_pixels):
                return

            paper_fraction = float(
                np.mean(
                    paper_mask[inside_pixels] > 0
                )
            )

            # Reject suspicious candidates selected
            # almost entirely from printed text.
            if paper_fraction < 0.45:
                return

            # Prefer large, rectangular regions
            # with paper-like color.
            score = (
                0.55 * area_ratio
                + 0.25 * fill_ratio
                + 0.20 * paper_fraction
            )

            candidates.append(
                (score, quad, method)
            )

        except (ValueError, cv2.error):
            return

    for method, mask in masks:

        contours_result = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        contours = sorted(
            contours_result[-2],
            key=cv2.contourArea,
            reverse=True
        )[:20]

        for contour in contours:

            if cv2.contourArea(contour) < (
                0.12 * image_area
            ):
                continue

            hull = cv2.convexHull(contour)

            perimeter = cv2.arcLength(
                hull,
                True
            )

            if perimeter <= 0:
                continue

            for epsilon_rate in (
                0.01, 0.015, 0.02,
                0.03, 0.04, 0.05
            ):

                approx = cv2.approxPolyDP(
                    hull,
                    epsilon_rate * perimeter,
                    True
                )

                if len(approx) != 4:
                    continue

                evaluate_candidate(
                    approx.reshape(4, 2),
                    hull,
                    method
                )

    # ==============================================
    # SELECT BEST CANDIDATE
    # ==============================================

    if not candidates:
        return canny_full, None, None, None

    candidates.sort(
        key=lambda item: item[0],
        reverse=True
    )

    best_score, corners, method = candidates[0]

    # Conservative confidence screening.
    # This is a heuristic score, not a probability.
    if best_score < 0.50:
        return canny_full, None, None, None

    # ==============================================
    # MAP CORNERS BACK TO ORIGINAL RESOLUTION
    # ==============================================

    original_points = (
        corners.astype(np.float32) / scale
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

    # ==============================================
    # MODULE 2 - PERSPECTIVE TRANSFORM
    # ==============================================

    scanned, matrix = four_point_transform(
        image,
        original_points
    )

    return (
        canny_full,
        scanned,
        original_points,
        f"{method} (score={best_score:.3f})"
    )


# ============================================================
# MANUAL FOUR-CORNER SELECTION DIALOG
# ============================================================

class ManualCornerDialog(ctk.CTkToplevel):

    def __init__(self, master, image, callback):
        super().__init__(master)

        self.title("Chọn 4 góc tài liệu")
        self.geometry("950x750")
        self.minsize(650, 500)

        self.callback = callback
        self.points = []

        self.original_h, self.original_w = image.shape[:2]
        self.source = numpy_to_pil(image)

        self.photo = None
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self.info = ctk.CTkLabel(
            self,
            text=(
                "Chọn lần lượt: Trên trái → Trên phải "
                "→ Dưới phải → Dưới trái"
            ),
            font=ctk.CTkFont(size=15, weight="bold")
        )

        self.info.grid(
            row=0, column=0, padx=12, pady=12
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

        self.canvas.bind("<Configure>", self.redraw)
        self.canvas.bind("<Button-1>", self.add_point)

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
        ).pack(side="left", padx=8, pady=10)

        ctk.CTkButton(
            toolbar,
            text="Chọn lại",
            command=self.reset_points
        ).pack(side="left", padx=8)

        ctk.CTkButton(
            toolbar,
            text="Nắn phẳng",
            fg_color=GREEN,
            command=self.apply_points
        ).pack(side="right", padx=8)

        self.transient(master)
        self.focus_force()

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
            1, int(self.original_w * self.scale)
        )
        height = max(
            1, int(self.original_h * self.scale)
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
                sx - 6, sy - 6,
                sx + 6, sy + 6,
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

    def add_point(self, event):

        if len(self.points) >= 4:
            return

        x = (event.x - self.offset_x) / self.scale
        y = (event.y - self.offset_y) / self.scale

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
            self.info.configure(
                text=f"Đã chọn {len(self.points)}/4 góc"
            )
            self.redraw()

    def reset_points(self):
        self.points.clear()
        self.info.configure(
            text="Chọn lại 4 góc theo thứ tự"
        )
        self.redraw()

    def apply_points(self):

        if len(self.points) != 4:
            messagebox.showwarning(
                "Thiếu điểm",
                "Bạn cần chọn đủ bốn góc.",
                parent=self
            )
            return

        try:
            points = np.asarray(
                self.points,
                dtype=np.float32
            )

            if len(np.unique(points, axis=0)) != 4:
                raise ValueError(
                    "Bốn điểm không được trùng nhau."
                )

            # Ensure the points form a valid quadrilateral
            hull = cv2.convexHull(points)

            if len(hull) != 4:
                raise ValueError(
                    "Bốn điểm phải tạo thành tứ giác lồi."
                )

            if cv2.contourArea(hull) < 100:
                raise ValueError(
                    "Vùng tài liệu được chọn quá nhỏ."
                )

            ordered = order_points(points)

            callback = self.callback
            self.destroy()
            callback(ordered)

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
# MAIN DESKTOP APPLICATION
# ============================================================

class DocumentScannerApp(ctk.CTk):

    def __init__(self):
        super().__init__()

        self.title(APP_TITLE)
        self.geometry("1500x850")
        self.minsize(1100, 650)
        self.configure(fg_color=BG)

        # Images
        self.original_image = None
        self.canny_image = None
        self.scanned_image = None
        self.enhanced_images = {}
        self.image_path = None

        # Preview memory
        self.canvases = {}
        self.preview_images = {}
        self.photo_refs = {}
        self.resize_jobs = {}

        # Processing state
        self.processing = False
        self.is_closing = False
        self.process_job = None
        self.canny_preview_job = None

        # Module 1 parameters
        self.canny_low = 50
        self.canny_high = 150

        # Module 3
        self.orientation_corrector = OrientationCorrector()

        # PDF options
        self.pdf_size = tk.StringVar(
            master=self, value="A4"
        )

        self.pdf_source = tk.StringVar(
            master=self, value="Bản scan"
        )

        self.build_ui()
        self.update_buttons()

        self.protocol(
            "WM_DELETE_WINDOW",
            self.on_close
        )

    # ========================================================
    # BUILD UI
    # ========================================================

    def build_ui(self):

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(3, weight=1)

        # HEADER
        header = ctk.CTkFrame(
            self,
            fg_color=PANEL,
            corner_radius=12
        )

        header.grid(
            row=0,
            column=0,
            padx=15,
            pady=(12, 6),
            sticky="ew"
        )

        ctk.CTkLabel(
            header,
            text="DOCUMENT SCANNER",
            font=ctk.CTkFont(
                size=25,
                weight="bold"
            ),
            text_color=TEXT
        ).pack(pady=(9, 2))

        ctk.CTkLabel(
            header,
            text=(
                "Perspective Correction | "
                "4-Module Integration | TV5 Week 5"
            ),
            text_color=MUTED
        ).pack(pady=(0, 8))

        # CANNY & PDF SETTINGS
        settings = ctk.CTkFrame(
            header,
            fg_color="transparent"
        )

        settings.pack(
            fill="x",
            padx=18,
            pady=(2, 9)
        )

        settings.grid_columnconfigure(0, weight=1)
        settings.grid_columnconfigure(1, weight=1)

        # Low threshold slider
        low_frame = ctk.CTkFrame(
            settings,
            fg_color="transparent"
        )
        low_frame.grid(
            row=0,
            column=0,
            padx=10,
            sticky="ew"
        )

        self.low_label = ctk.CTkLabel(
            low_frame,
            text="Canny Low: 50"
        )
        self.low_label.pack(anchor="w")

        self.low_slider = ctk.CTkSlider(
            low_frame,
            from_=0,
            to=254,
            number_of_steps=254,
            command=self.on_low_change
        )
        self.low_slider.pack(fill="x", pady=4)
        self.low_slider.set(50)

        # High threshold slider
        high_frame = ctk.CTkFrame(
            settings,
            fg_color="transparent"
        )
        high_frame.grid(
            row=0,
            column=1,
            padx=10,
            sticky="ew"
        )

        self.high_label = ctk.CTkLabel(
            high_frame,
            text="Canny High: 150"
        )
        self.high_label.pack(anchor="w")

        self.high_slider = ctk.CTkSlider(
            high_frame,
            from_=1,
            to=255,
            number_of_steps=254,
            command=self.on_high_change
        )
        self.high_slider.pack(fill="x", pady=4)
        self.high_slider.set(150)

        # PDF settings
        pdf_options = ctk.CTkFrame(
            settings,
            fg_color="transparent"
        )

        pdf_options.grid(
            row=1,
            column=0,
            columnspan=2,
            pady=(6, 0)
        )

        ctk.CTkLabel(
            pdf_options,
            text="Khổ giấy PDF:"
        ).pack(side="left", padx=5)

        ctk.CTkOptionMenu(
            pdf_options,
            values=["A4", "A3", "Letter"],
            variable=self.pdf_size,
            width=100
        ).pack(side="left", padx=5)

        ctk.CTkLabel(
            pdf_options,
            text="Nguồn ảnh:"
        ).pack(side="left", padx=(15, 5))

        ctk.CTkOptionMenu(
            pdf_options,
            values=["Bản scan", "Kết quả nâng cao"],
            variable=self.pdf_source,
            width=155
        ).pack(side="left", padx=5)

        # MAIN TOOLBAR
        toolbar = ctk.CTkFrame(
            self,
            fg_color=PANEL
        )

        toolbar.grid(
            row=1,
            column=0,
            padx=15,
            pady=6,
            sticky="ew"
        )

        self.btn_open = ctk.CTkButton(
            toolbar,
            text="Chọn ảnh",
            width=100,
            command=self.choose_image
        )
        self.btn_open.pack(
            side="left", padx=(10, 4), pady=10
        )

        self.btn_process = ctk.CTkButton(
            toolbar,
            text="Xử lý tự động",
            width=120,
            fg_color=GREEN,
            command=self.process_image
        )
        self.btn_process.pack(side="left", padx=4)

        self.btn_manual = ctk.CTkButton(
            toolbar,
            text="Chọn 4 góc",
            width=105,
            fg_color=ORANGE,
            command=self.manual_select
        )
        self.btn_manual.pack(side="left", padx=4)

        self.btn_orientation = ctk.CTkButton(
            toolbar,
            text="Hiệu chỉnh hướng",
            width=135,
            fg_color="#0F766E",
            command=self.correct_orientation
        )
        self.btn_orientation.pack(side="left", padx=4)

        self.btn_rotate_left = ctk.CTkButton(
            toolbar,
            text="Xoay trái 90°",
            width=105,
            fg_color="#506380",
            command=self.rotate_left
        )
        self.btn_rotate_left.pack(side="left", padx=4)

        self.btn_rotate_right = ctk.CTkButton(
            toolbar,
            text="Xoay phải 90°",
            width=105,
            fg_color="#506380",
            command=self.rotate_right
        )
        self.btn_rotate_right.pack(side="left", padx=4)

        self.btn_save = ctk.CTkButton(
            toolbar,
            text="Lưu bản scan",
            width=115,
            command=self.save_image
        )
        self.btn_save.pack(side="left", padx=4)

        self.btn_pdf = ctk.CTkButton(
            toolbar,
            text="Xuất PDF",
            width=100,
            fg_color=PURPLE,
            command=self.export_pdf
        )
        self.btn_pdf.pack(side="left", padx=4)

        self.btn_reset = ctk.CTkButton(
            toolbar,
            text="Làm mới",
            width=85,
            fg_color=GRAY,
            command=self.reset_images
        )
        self.btn_reset.pack(side="left", padx=4)

        # IMAGE ENHANCEMENT TOOLBAR
        enhancement_toolbar = ctk.CTkFrame(
            self,
            fg_color=PANEL
        )

        enhancement_toolbar.grid(
            row=2,
            column=0,
            padx=15,
            pady=(0, 6),
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
            text="Kết quả:"
        ).pack(side="left", padx=(8, 5))

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
            padx=5
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

        # FOUR IMAGE PANELS
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

        content.grid_rowconfigure(0, weight=1)

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
                padx=5,
                sticky="nsew"
            )

            frame.grid_columnconfigure(0, weight=1)
            frame.grid_rowconfigure(1, weight=1)

            ctk.CTkLabel(
                frame,
                text=title,
                font=ctk.CTkFont(
                    size=14,
                    weight="bold"
                ),
                text_color=TEXT
            ).grid(
                row=0,
                column=0,
                pady=12
            )

            # Tk Canvas avoids CTkImage reference issues.
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

        # STATUS BAR
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
            pady=(4, 10),
            sticky="ew"
        )

    # ========================================================
    # STATUS AND BUTTON STATES
    # ========================================================

    def set_status(self, message, error=False):
        if self.is_closing:
            return

        self.status_label.configure(
            text=f"Trạng thái: {message}",
            text_color=RED if error else MUTED
        )

    def update_buttons(self):

        controls = [
            self.btn_open,
            self.btn_process,
            self.btn_manual,
            self.btn_orientation,
            self.btn_rotate_left,
            self.btn_rotate_right,
            self.btn_save,
            self.btn_pdf,
            self.btn_reset,
            self.btn_enhance,
            self.enhancement_selector,
            self.btn_save_enhanced,
            self.low_slider,
            self.high_slider
        ]

        if self.processing:
            for control in controls:
                control.configure(state="disabled")
            return

        has_original = self.original_image is not None
        has_scan = self.scanned_image is not None
        has_enhancement = bool(self.enhanced_images)

        self.btn_open.configure(state="normal")
        self.btn_reset.configure(state="normal")

        for button in (
            self.btn_process,
            self.btn_manual
        ):
            button.configure(
                state="normal" if has_original else "disabled"
            )

        for slider in (
            self.low_slider,
            self.high_slider
        ):
            slider.configure(
                state="normal" if has_original else "disabled"
            )

        for button in (
            self.btn_orientation,
            self.btn_rotate_left,
            self.btn_rotate_right,
            self.btn_save,
            self.btn_enhance
        ):
            button.configure(
                state="normal" if has_scan else "disabled"
            )

        self.enhancement_selector.configure(
            state="normal" if has_enhancement else "disabled"
        )

        self.btn_save_enhanced.configure(
            state="normal" if has_enhancement else "disabled"
        )

        self.btn_pdf.configure(
            state="normal"
            if has_scan or has_enhancement
            else "disabled"
        )

    # ========================================================
    # IMAGE INPUT
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
                pil = ImageOps.exif_transpose(
                    opened
                ).convert("RGB")

            rgb = np.asarray(pil)

            self.original_image = cv2.cvtColor(
                rgb,
                cv2.COLOR_RGB2BGR
            )

            self.image_path = path

            self.canny_image = None
            self.scanned_image = None
            self.enhanced_images.clear()

            self.display_image(
                "original",
                self.original_image
            )

            for key in ("canny", "scanned", "enhanced"):
                self.clear_panel(key)

            self.update_buttons()

            self.set_status(
                f"Đã tải ảnh: {os.path.basename(path)}"
            )

            self.schedule_canny_preview()

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
    # CANNY SLIDERS - MODULE 1
    # ========================================================

    def on_low_change(self, value):

        self.canny_low = min(
            254,
            int(round(value))
        )

        if self.canny_low >= self.canny_high:
            self.canny_high = self.canny_low + 1
            self.high_slider.set(self.canny_high)

        self.low_label.configure(
            text=f"Canny Low: {self.canny_low}"
        )

        self.high_label.configure(
            text=f"Canny High: {self.canny_high}"
        )

        self.schedule_canny_preview()

    def on_high_change(self, value):

        self.canny_high = max(
            1,
            int(round(value))
        )

        if self.canny_high <= self.canny_low:
            self.canny_low = self.canny_high - 1
            self.low_slider.set(self.canny_low)

        self.low_label.configure(
            text=f"Canny Low: {self.canny_low}"
        )

        self.high_label.configure(
            text=f"Canny High: {self.canny_high}"
        )

        self.schedule_canny_preview()

    def cancel_canny_preview(self):

        if self.canny_preview_job is not None:
            try:
                self.after_cancel(self.canny_preview_job)
            except tk.TclError:
                pass

            self.canny_preview_job = None

    def schedule_canny_preview(self):

        if not hasattr(self, "original_image"):
            return

        if self.original_image is None:
            return

        if self.processing or self.is_closing:
            return

        self.cancel_canny_preview()

        self.canny_preview_job = self.after(
            150,
            self.refresh_canny_preview
        )

    def refresh_canny_preview(self):

        self.canny_preview_job = None

        if self.original_image is None or self.processing:
            return

        try:
            image = self.original_image
            h, w = image.shape[:2]

            scale = min(
                1.0,
                1100.0 / max(h, w)
            )

            if scale < 1.0:
                image = cv2.resize(
                    image,
                    None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_AREA
                )

            # OFFICIAL MODULE 1
            _, raw_edges, _ = (
                preprocess_and_detect_edges(
                    image,
                    blur_kernel=(5, 5),
                    low_threshold=self.canny_low,
                    high_threshold=self.canny_high,
                    use_auto_canny=False,
                    apply_morphology=True
                )
            )

            self.canny_image = raw_edges
            self.display_image("canny", raw_edges)

            # Existing scan no longer matches thresholds
            self.scanned_image = None
            self.clear_enhancement()

            self.clear_panel("scanned")
            self.update_buttons()

            self.set_status(
                f"Canny Low={self.canny_low}, "
                f"High={self.canny_high}. "
                "Nhấn Xử lý tự động để nắn phẳng."
            )

        except Exception as error:
            traceback.print_exc()
            self.set_status(
                f"Lỗi xem trước Canny: {error}",
                error=True
            )

    # ========================================================
    # AUTOMATIC DOCUMENT SCANNING - MODULES 1 & 2
    # ========================================================

    def process_image(self):

        if self.original_image is None or self.processing:
            return

        self.cancel_canny_preview()

        self.processing = True
        self.update_buttons()

        self.set_status(
            "Đang phát hiện cạnh và nắn phẳng tài liệu..."
        )

        self.process_job = self.after(
            40,
            self.run_processing
        )

    def run_processing(self):

        self.process_job = None

        try:
            canny, scanned, points, method = (
                detect_document(
                    self.original_image,
                    low=self.canny_low,
                    high=self.canny_high
                )
            )

            self.canny_image = canny
            self.scanned_image = scanned

            self.clear_enhancement()
            self.display_image("canny", canny)

            if scanned is None:
                self.clear_panel("scanned")

                self.set_status(
                    "Không tìm thấy 4 góc tài liệu.",
                    error=True
                )

                messagebox.showwarning(
                    "Không tìm thấy tài liệu",
                    "Không tìm được 4 góc.\n\n"
                    "Hãy thử điều chỉnh ngưỡng Canny "
                    "hoặc dùng nút Chọn 4 góc.",
                    parent=self
                )
                return

            self.display_image(
                "scanned",
                scanned
            )

            self.set_status(
                f"Đã nắn phẳng bằng {method}. "
                "Có thể hiệu chỉnh hướng và nâng cao ảnh."
            )

        except Exception as error:
            traceback.print_exc()

            self.canny_image = None
            self.scanned_image = None
            self.clear_enhancement()

            self.clear_panel("canny")
            self.clear_panel("scanned")

            messagebox.showerror(
                "Lỗi xử lý ảnh",
                str(error),
                parent=self
            )

            self.set_status(
                "Xử lý ảnh thất bại.",
                error=True
            )

        finally:
            self.processing = False
            self.update_buttons()

    # ========================================================
    # MANUAL FOUR-CORNER SELECTION - MODULE 2
    # ========================================================

    def manual_select(self):

        if self.original_image is None or self.processing:
            return

        ManualCornerDialog(
            self,
            self.original_image,
            self.apply_manual_points
        )

    def apply_manual_points(self, points):

        try:
            scanned, matrix = four_point_transform(
                self.original_image,
                points
            )

            self.scanned_image = scanned
            self.clear_enhancement()

            self.display_image(
                "scanned",
                scanned
            )

            if self.canny_image is None:
                self.refresh_canny_preview()

                # refresh_canny_preview invalidates scans,
                # so restore the manual result afterward.
                self.scanned_image = scanned
                self.display_image("scanned", scanned)

            self.set_status(
                "Đã nắn phẳng bằng 4 góc thủ công."
            )

            self.update_buttons()

        except Exception as error:
            traceback.print_exc()

            messagebox.showerror(
                "Lỗi nắn phẳng",
                str(error),
                parent=self
            )

            self.set_status(
                "Nắn phẳng thủ công thất bại.",
                error=True
            )

    # ========================================================
    # ORIENTATION CORRECTION - MODULE 3
    # ========================================================

    def correct_orientation(self):
        """
        Use standalone deskew without forcing landscape
        documents into portrait orientation.

        Users can still rotate left/right manually.
        """

        if self.scanned_image is None or self.processing:
            return

        try:
            corrected, angle = (
                self.orientation_corrector.deskew_standalone(
                    self.scanned_image
                )
            )

            self.scanned_image = corrected.copy()

            self.clear_enhancement()

            self.display_image(
                "scanned",
                self.scanned_image
            )

            self.set_status(
                f"Đã hiệu chỉnh góc nghiêng: {angle:.2f}°."
            )

            self.update_buttons()

        except Exception as error:
            traceback.print_exc()

            messagebox.showerror(
                "Lỗi hiệu chỉnh hướng",
                str(error),
                parent=self
            )

            self.set_status(
                "Hiệu chỉnh hướng thất bại.",
                error=True
            )

    # ========================================================
    # ROTATE SCANNED IMAGE
    # ========================================================

    def rotate_left(self):

        if self.scanned_image is None or self.processing:
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

        self.update_buttons()

        self.set_status(
            "Đã xoay bản scan sang trái 90°."
        )

    def rotate_right(self):

        if self.scanned_image is None or self.processing:
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

        self.update_buttons()

        self.set_status(
            "Đã xoay bản scan sang phải 90°."
        )

    # ========================================================
    # IMAGE ENHANCEMENT - MODULE 4
    # ========================================================

    def clear_enhancement(self):

        self.enhanced_images.clear()

        if "enhanced" in self.canvases:
            self.clear_panel("enhanced")

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
            # OFFICIAL MODULE 4
            corrected, sharpened, binary = (
                enhance_document(
                    self.scanned_image,
                    sigma=25.0,
                    sharpen_strength=1.2,
                    block_size=31,
                    C=10.0
                )
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
                "Tăng cường chất lượng thành công. "
                "Có thể lưu ảnh hoặc xuất PDF."
            )

        except Exception as error:
            traceback.print_exc()

            self.clear_enhancement()

            messagebox.showerror(
                "Lỗi nâng cao chất lượng",
                str(error),
                parent=self
            )

            self.set_status(
                "Nâng cao chất lượng ảnh thất bại.",
                error=True
            )

        finally:
            self.processing = False
            self.update_buttons()

    def select_enhancement(self, selection):

        image = self.enhanced_images.get(selection)

        if image is not None:
            self.display_image("enhanced", image)

    # ========================================================
    # DISPLAY IMAGES ON TK CANVAS
    # ========================================================

    def display_image(self, key, image):

        if image is None:
            self.clear_panel(key)
            return

        pil = numpy_to_pil(image)

        # Limit preview size only; preserve original arrays.
        pil.thumbnail(
            MAX_PREVIEW,
            Image.Resampling.LANCZOS
        )

        self.preview_images[key] = pil
        self.schedule_refresh(key)

    def schedule_refresh(self, key):

        if self.is_closing:
            return

        previous = self.resize_jobs.pop(
            key, None
        )

        if previous is not None:
            try:
                self.after_cancel(previous)
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

        photo = ImageTk.PhotoImage(
            preview,
            master=self
        )

        canvas.delete("all")

        canvas.create_image(
            cw // 2,
            ch // 2,
            image=photo,
            anchor="center"
        )

        # Keep reference alive, preventing pyimage errors.
        self.photo_refs[key] = photo

    def draw_placeholder(self, key):

        canvas = self.canvases[key]

        if not canvas.winfo_exists():
            return

        canvas.delete("all")

        cw = max(canvas.winfo_width(), 100)
        ch = max(canvas.winfo_height(), 100)

        canvas.create_text(
            cw // 2,
            ch // 2,
            text="Chưa có ảnh",
            fill=MUTED,
            font=("Arial", 12)
        )

    def clear_panel(self, key):

        job = self.resize_jobs.pop(key, None)

        if job is not None:
            try:
                self.after_cancel(job)
            except tk.TclError:
                pass

        self.preview_images.pop(key, None)

        canvas = self.canvases[key]
        canvas.delete("all")

        self.photo_refs.pop(key, None)
        self.draw_placeholder(key)

    # ========================================================
    # SAVE SCANNED IMAGE
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
            save_numpy_image(
                self.scanned_image,
                path
            )

            self.set_status(
                f"Đã lưu bản scan: {path}"
            )

            messagebox.showinfo(
                "Thành công",
                f"Đã lưu ảnh!\n\n{path}",
                parent=self
            )

        except Exception as error:
            traceback.print_exc()

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
                "Chưa có ảnh nâng cao để lưu.",
                parent=self
            )
            return

        stem = selection.lower().replace(" ", "_")

        path = filedialog.asksaveasfilename(
            parent=self,
            title="Lưu kết quả nâng cao",
            defaultextension=".png",
            initialfile=f"document_{stem}.png",
            filetypes=[
                ("PNG Image", "*.png"),
                ("JPEG Image", "*.jpg")
            ]
        )

        if not path:
            return

        try:
            save_numpy_image(image, path)

            self.set_status(
                f"Đã lưu ảnh nâng cao: {path}"
            )

            messagebox.showinfo(
                "Thành công",
                f"Đã lưu {selection}!\n\n{path}",
                parent=self
            )

        except Exception as error:
            traceback.print_exc()

            messagebox.showerror(
                "Lỗi lưu ảnh nâng cao",
                str(error),
                parent=self
            )

    # ========================================================
    # EXPORT PDF - TV5 WEEK 5
    # ========================================================

    def export_pdf(self):

        source_type = self.pdf_source.get()

        if source_type == "Kết quả nâng cao":
            selection = self.enhancement_selector.get()
            image = self.enhanced_images.get(selection)
        else:
            image = self.scanned_image

        if image is None:
            messagebox.showwarning(
                "Thông báo",
                "Nguồn ảnh đã chọn chưa có kết quả.",
                parent=self
            )
            return

        path = filedialog.asksaveasfilename(
            parent=self,
            title="Xuất tài liệu PDF",
            defaultextension=".pdf",
            initialfile="document_scan.pdf",
            filetypes=[
                ("PDF Document", "*.pdf")
            ]
        )

        if not path:
            return

        try:
            paper = self.pdf_size.get()

            page_sizes = {
                "A4": A4,
                "A3": A3,
                "Letter": letter
            }

            page_w, page_h = page_sizes[paper]

            # Use full-resolution scan, not UI preview.
            pil_image = numpy_to_pil(image)

            img_w, img_h = pil_image.size

            # Landscape document -> landscape PDF page.
            if img_w > img_h:
                page_w, page_h = page_h, page_w

            margin = 10 * mm

            usable_w = page_w - 2 * margin
            usable_h = page_h - 2 * margin

            scale = min(
                usable_w / img_w,
                usable_h / img_h
            )

            draw_w = img_w * scale
            draw_h = img_h * scale

            x = (page_w - draw_w) / 2
            y = (page_h - draw_h) / 2

            image_buffer = io.BytesIO()

            pil_image.save(
                image_buffer,
                format="PNG"
            )
            image_buffer.seek(0)

            pdf = pdf_canvas.Canvas(
                path,
                pagesize=(page_w, page_h)
            )

            pdf.setTitle(
                "Document Scanner - Scanned Document"
            )

            pdf.setAuthor(
                "Document Scanner - TV5"
            )

            pdf.drawImage(
                ImageReader(image_buffer),
                x,
                y,
                width=draw_w,
                height=draw_h
            )

            pdf.showPage()
            pdf.save()

            self.set_status(
                f"Đã xuất PDF {paper}: {path}"
            )

            messagebox.showinfo(
                "Xuất PDF thành công",
                f"Đã lưu PDF khổ {paper}!\n\n{path}",
                parent=self
            )

        except Exception as error:
            traceback.print_exc()

            self.set_status(
                "Xuất PDF thất bại.",
                error=True
            )

            messagebox.showerror(
                "Lỗi xuất PDF",
                str(error),
                parent=self
            )

    # ========================================================
    # RESET
    # ========================================================

    def reset_images(self):

        if self.processing:
            return

        self.cancel_canny_preview()

        self.original_image = None
        self.canny_image = None
        self.scanned_image = None
        self.enhanced_images.clear()
        self.image_path = None

        for key in self.canvases:
            self.clear_panel(key)

        self.update_buttons()

        self.set_status(
            "Đã làm mới ứng dụng."
        )

    # ========================================================
    # CLOSE
    # ========================================================

    def on_close(self):

        self.is_closing = True

        self.cancel_canny_preview()

        if self.process_job is not None:
            try:
                self.after_cancel(self.process_job)
            except tk.TclError:
                pass

        for job in list(self.resize_jobs.values()):
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
# RUN APPLICATION
# ============================================================

def main():
    app = DocumentScannerApp()
    app.mainloop()


if __name__ == "__main__":
    main()
