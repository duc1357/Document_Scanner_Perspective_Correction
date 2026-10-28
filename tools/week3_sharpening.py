
from pathlib import Path
import sys
import cv2
import csv
# Thêm thư mục gốc dự án vào đường dẫn tìm kiếm của Python
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from modules.image_enhancer import sharpen_text


# Thư mục chứa ảnh đầu vào và kết quả kiểm thử
INPUT_DIR = Path("dataset/raw")
OUTPUT_DIR = Path("dataset/output/tv4_week3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def create_comparison(original, sharpened):
    """Ghép ảnh gốc và ảnh sau khi làm sắc nét để dễ so sánh."""
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY) \
        if original.ndim == 3 else original

    sharpened_gray = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY) \
        if sharpened.ndim == 3 else sharpened

    # Đưa hai ảnh về cùng kích thước nếu cần
    if original_gray.shape != sharpened_gray.shape:
        sharpened_gray = cv2.resize(
            sharpened_gray,
            (original_gray.shape[1], original_gray.shape[0])
        )

    # Ghép ảnh gốc và ảnh kết quả theo chiều ngang
    return cv2.hconcat([original_gray, sharpened_gray])


def run_week3_test():
    """Kiểm thử bộ lọc Sharpening trên toàn bộ ảnh trong dataset/raw."""
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths = [
        path for path in INPUT_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in image_extensions
    ]

    if not image_paths:
        print("Khong tim thay anh trong dataset/raw.")
        return

    report_path = OUTPUT_DIR / "sharpening_report.csv"
    report = []

    for image_path in image_paths:
        # Đọc ảnh đầu vào
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Khong doc duoc anh: {image_path.name}")
            continue

        # Chuyển ảnh sang thang xám trước khi làm sắc nét
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Gọi hàm Sharpening của Module 4
        sharpened = sharpen_text(gray)

        if sharpened is None:
            print(f"Khong co ket qua: {image_path.name}")
            continue

        # Lưu ảnh trước và sau khi xử lý
        cv2.imwrite(
            str(OUTPUT_DIR / f"{image_path.stem}_before.png"),
            gray
        )
        cv2.imwrite(
            str(OUTPUT_DIR / f"{image_path.stem}_after.png"),
            sharpened
        )

        # Lưu ảnh so sánh
        comparison = create_comparison(gray, sharpened)
        cv2.imwrite(
            str(OUTPUT_DIR / f"{image_path.stem}_comparison.png"),
            comparison
        )

        report.append({
            "image": image_path.name,
            "width": gray.shape[1],
            "height": gray.shape[0],
            "result": "success"
        })

        print(f"Da xu ly: {image_path.name}")

    # Xuất báo cáo kiểm thử
    with open(report_path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["image", "width", "height", "result"]
        )
        writer.writeheader()
        writer.writerows(report)

    print(f"\nHoan tat. Ket qua nam tai: {OUTPUT_DIR}")


if __name__ == "__main__":
    run_week3_test()
