
import sys
from pathlib import Path

import cv2
import numpy as np

# Thêm thư mục gốc dự án vào đường dẫn tìm kiếm của Python
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from modules.image_enhancer import enhance_document


# Khai báo thư mục dữ liệu và thư mục lưu kết quả
INPUT_DIR = PROJECT_ROOT / "dataset" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "dataset" / "output" / "tv4_week4"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def validate_image(image, original_shape):
    """Kiểm tra kiểu dữ liệu và kích thước ảnh đầu ra."""
    if not isinstance(image, np.ndarray):
        return False

    if image.dtype != np.uint8:
        return False

    if image.ndim != 2:
        return False

    if image.shape != original_shape[:2]:
        return False

    return True


def test_image(image_path):
    """Kiểm thử toàn bộ API Module 4 trên một ảnh."""
    image = cv2.imread(str(image_path))

    if image is None:
        return {
            "image": image_path.name,
            "status": "FAIL",
            "details": "Khong doc duoc anh"
        }

    try:
        # Chạy toàn bộ quy trình xử lý của Module 4
        results = enhance_document(image)

        if not isinstance(results, (tuple, list)) or len(results) != 3:
            raise ValueError(
                "enhance_document phai tra ve 3 anh"
            )

        corrected, sharpened, binary = results
        outputs = {
            "illumination": corrected,
            "sharpened": sharpened,
            "binary": binary
        }

        # Kiểm tra từng ảnh đầu ra
        for name, output in outputs.items():
            if not validate_image(output, image.shape):
                raise ValueError(
                    f"Anh {name} sai kieu du lieu hoac kich thuoc"
                )

            output_path = (
                OUTPUT_DIR /
                f"{image_path.stem}_{name}.png"
            )
            cv2.imwrite(str(output_path), output)

        return {
            "image": image_path.name,
            "status": "PASS",
            "details": "Ba anh dau ra hop le"
        }

    except Exception as error:
        return {
            "image": image_path.name,
            "status": "FAIL",
            "details": str(error)
        }


def main():
    """Chạy kiểm thử API trên toàn bộ ảnh trong dataset/raw."""
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    if not INPUT_DIR.exists():
        print(f"Khong tim thay thu muc: {INPUT_DIR}")
        return

    image_paths = sorted(
        path for path in INPUT_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    )

    if not image_paths:
        print("Khong tim thay anh dau vao.")
        return

    results = [test_image(path) for path in image_paths]

    report_path = OUTPUT_DIR / "api_test_report.csv"

    import csv

    with open(report_path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["image", "status", "details"]
        )
        writer.writeheader()
        writer.writerows(results)

    passed = sum(result["status"] == "PASS" for result in results)
    failed = len(results) - passed

    print("\n===== KET QUA KIEM THU MODULE 4 =====")
    print(f"Tong so anh: {len(results)}")
    print(f"Thanh cong: {passed}")
    print(f"That bai: {failed}")
    print(f"Bao cao: {report_path}")


if __name__ == "__main__":
    main()
