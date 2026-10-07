"""
Script tạo ảnh mẫu tài liệu chụp nghiêng phục vụ kiểm thử cho TV1 & TV2.
Sử dụng Pillow và Numpy (đã cài đặt sẵn).
"""

import os
import sys
import io

# Đảm bảo in tiếng Việt trên console Windows không lỗi charmap cp1252
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from PIL import Image, ImageDraw
import numpy as np


def generate_sample_document(output_path: str = "dataset/raw/sample_doc_skewed.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 1. Tạo ảnh nền vân bàn gỗ (màu nâu sẫm/nhiễu)
    width, height = 900, 700
    background = np.full((height, width, 3), (45, 30, 20), dtype=np.uint8)
    noise = np.random.randint(-15, 15, (height, width, 3), dtype=np.int16)
    bg_noisy = np.clip(background.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(bg_noisy)
    draw = ImageDraw.Draw(img)

    # 2. Tạo hình tứ giác của tờ giấy nghiêng (Perspective Quadrilateral)
    paper_corners = [
        (200, 100),  # Trên - Trái
        (750, 150),  # Trên - Phải
        (680, 600),  # Dưới - Phải
        (140, 520)   # Dưới - Trái
    ]

    # Vẽ nền tờ giấy màu trắng hơi ngà
    draw.polygon(paper_corners, fill=(245, 245, 240), outline=(180, 180, 180), width=2)

    # 3. Vẽ các dòng văn bản mô phỏng trên tờ giấy
    # Các dòng kẻ mô phỏng đoạn văn
    for i in range(12):
        y_ratio = (i + 2) / 16.0
        # Nội suy tuyến tính tọa độ 2 đầu dòng kẻ
        left_x = int(paper_corners[0][0] * (1 - y_ratio) + paper_corners[3][0] * y_ratio + 40)
        left_y = int(paper_corners[0][1] * (1 - y_ratio) + paper_corners[3][1] * y_ratio + 10)
        right_x = int(paper_corners[1][0] * (1 - y_ratio) + paper_corners[2][0] * y_ratio - 40)
        right_y = int(paper_corners[1][1] * (1 - y_ratio) + paper_corners[2][1] * y_ratio + 10)

        # Vẽ dòng kẻ văn bản
        color = (30, 30, 30) if i > 0 else (10, 40, 120)  # Tiêu đề màu xanh đậm
        thickness = 4 if i == 0 else 2
        draw.line([(left_x, left_y), (right_x, right_y)], fill=color, width=thickness)

    img.save(output_path)
    print(f"[OK] Đã tạo ảnh mẫu chụp nghiêng thành công tại: {output_path}")


if __name__ == "__main__":
    generate_sample_document()
