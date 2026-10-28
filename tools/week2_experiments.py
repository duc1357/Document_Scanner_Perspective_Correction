"""
Kiểm thử tuần 2: Module 4 - Tăng cường chất lượng ảnh
So sánh ba phương pháp CLAHE, Otsu và Adaptive Gaussian
trên toàn bộ bộ dữ liệu ảnh trong thư mục dataset/raw.
"""

import os
import csv
import time
import cv2
import numpy as np

def enhance_clahe(gray):
    """Tang do tuong phan cuc bo bang CLAHE."""
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    return clahe.apply(gray)

def apply_otsu(gray):
    """Nhi phan hoa anh bang nguong Otsu."""
    threshold, result = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return result, threshold


def apply_adaptive(gray):
    """Nhi phan hoa anh bang Adaptive Gaussian."""
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        10
    )


def create_comparison(images):
    """Ghep anh goc va ba ket qua thanh anh so sanh 2x2."""
    titles = [
        "Anh xam goc",
        "CLAHE",
        "Otsu",
        "Adaptive Gaussian"
    ]

    panels = []

    for title, image in zip(titles, images):
        panel = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        cv2.putText(
            panel,
            title,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )

        panels.append(panel)

    top = np.hstack((panels[0], panels[1]))
    bottom = np.hstack((panels[2], panels[3]))

    return np.vstack((top, bottom))


def run_benchmark():
    """Chay kiem thu tren toan bo anh trong dataset/raw."""

    # File nam trong tools, di len thu muc goc du an
    project_dir = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )

    # Thu muc du lieu goc
    raw_dir = os.path.join(
        project_dir, "dataset", "raw"
    )

    # Luu ket qua trong dataset/output/week2
    output_dir = os.path.join(
        project_dir, "dataset", "output", "week2"
    )

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("KIEM THU TUAN 2 - MODULE 4")
    print("Thu muc project:", project_dir)
    print("Thu muc raw:", raw_dir)
    print("Thu muc output:", output_dir)

    if not os.path.isdir(raw_dir):
        print("LOI: Khong tim thay thu muc dataset/raw!")
        return

    # Cac dinh dang anh duoc ho tro
    extensions = (
        ".jpg", ".jpeg", ".png",
        ".bmp", ".tif", ".tiff"
    )

    files = sorted(
        filename
        for filename in os.listdir(raw_dir)
        if filename.lower().endswith(extensions)
    )

    if not files:
        print("LOI: Khong tim thay anh trong dataset/raw!")
        return

    print("Tong so anh tim thay:", len(files))
    print("=" * 70)

    results = []

    total_times = {
        "clahe": 0.0,
        "otsu": 0.0,
        "adaptive": 0.0
    }

    success = 0

    print(
        f"{'STT':<5} {'Ten file':<25} "
        f"{'CLAHE(ms)':<13} {'Otsu(ms)':<13} "
        f"{'Adaptive(ms)':<15} {'Trang thai'}"
    )
    print("-" * 90)

    for index, filename in enumerate(files, start=1):
        image_path = os.path.join(raw_dir, filename)
        image = cv2.imread(image_path)

        if image is None:
            print(
                f"{index:<5} {filename:<25} "
                "Khong doc duoc anh"
            )
            continue

        try:
            # Chuyen anh mau sang anh xam
            gray = cv2.cvtColor(
                image, cv2.COLOR_BGR2GRAY
            )

            # CLAHE
            start = time.perf_counter()
            clahe = enhance_clahe(gray)
            clahe_ms = (
                time.perf_counter() - start
            ) * 1000

            # Otsu
            start = time.perf_counter()
            otsu, threshold = apply_otsu(gray)
            otsu_ms = (
                time.perf_counter() - start
            ) * 1000

            # Adaptive Gaussian
            start = time.perf_counter()
            adaptive = apply_adaptive(gray)
            adaptive_ms = (
                time.perf_counter() - start
            ) * 1000

            # Ten file dau ra
            name = os.path.splitext(filename)[0]

            # Luu anh ket qua tung phuong phap
            output_images = {
                f"{name}_clahe.png": clahe,
                f"{name}_otsu.png": otsu,
                f"{name}_adaptive.png": adaptive
            }

            for output_name, output_image in output_images.items():
                saved = cv2.imwrite(
                    os.path.join(output_dir, output_name),
                    output_image
                )

                if not saved:
                    raise OSError(
                        f"Khong the luu anh: {output_name}"
                    )

            # Tao va luu anh so sanh
            comparison = create_comparison(
                [gray, clahe, otsu, adaptive]
            )

            comparison_path = os.path.join(
                output_dir,
                f"{name}_comparison.jpg"
            )

            if not cv2.imwrite(
                comparison_path, comparison
            ):
                raise OSError(
                    f"Khong the luu anh: {comparison_path}"
                )

            # Cap nhat thong ke
            total_times["clahe"] += clahe_ms
            total_times["otsu"] += otsu_ms
            total_times["adaptive"] += adaptive_ms
            success += 1

            # Luu thong tin vao bao cao
            results.append({
                "ten_file": filename,
                "chieu_rong": image.shape[1],
                "chieu_cao": image.shape[0],
                "nguong_otsu": round(
                    float(threshold), 2
                ),
                "clahe_ms": round(clahe_ms, 3),
                "otsu_ms": round(otsu_ms, 3),
                "adaptive_ms": round(adaptive_ms, 3)
            })

            print(
                f"{index:<5} {filename:<25} "
                f"{clahe_ms:<13.3f} "
                f"{otsu_ms:<13.3f} "
                f"{adaptive_ms:<15.3f} "
                "Thanh cong"
            )

        except (cv2.error, ValueError, OSError) as error:
            print(
                f"{index:<5} {filename:<25} "
                f"Loi xu ly: {error}"
            )

    # Xuat bao cao CSV
    csv_path = os.path.join(
        output_dir,
        "ket_qua_thuc_nghiem_tuan2.csv"
    )

    if results:
        with open(
            csv_path,
            "w",
            newline="",
            encoding="utf-8-sig"
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=results[0].keys()
            )
            writer.writeheader()
            writer.writerows(results)

    # Bao cao tong ket
    print("\n" + "=" * 70)
    print("TONG KET THUC NGHIEM")
    print(f"So anh xu ly thanh cong: {success}/{len(files)}")
    print(f"So anh loi hoac khong doc duoc: {len(files) - success}")

    if success > 0:
        print(
            f"CLAHE trung binh: "
            f"{total_times['clahe'] / success:.3f} ms"
        )
        print(
            f"Otsu trung binh: "
            f"{total_times['otsu'] / success:.3f} ms"
        )
        print(
            f"Adaptive Gaussian trung binh: "
            f"{total_times['adaptive'] / success:.3f} ms"
        )

        print("Thu muc ket qua:", output_dir)
        print("Bao cao CSV:", csv_path)

    print("=" * 70)


if __name__ == "__main__":
    run_benchmark()
