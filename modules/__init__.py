"""
Package modules cho dự án Document Scanner & Perspective Correction.
"""

from .edge_detector import (
    to_grayscale,
    apply_gaussian_blur,
    detect_canny_edges,
    auto_canny,
    connect_edges,
    preprocess_and_detect_edges,
    resize_image_with_ratio,
    robust_edge_detection,
)

from .perspective_warp import (
    order_points,
    find_document_contour,
    four_point_transform,
    warp_document_pipeline,
)
from .orientation_corrector import OrientationCorrector
from .image_enhancer import (
    convert_to_grayscale,
    correct_illumination,
    sharpen_text,
    adaptive_threshold,
    enhance_document,
)

__all__ = [
    # Module 1
    "to_grayscale",
    "apply_gaussian_blur",
    "detect_canny_edges",
    "auto_canny",
    "connect_edges",
    "preprocess_and_detect_edges",
    "resize_image_with_ratio",
    "robust_edge_detection",
    # Module 2
    "order_points",
    "find_document_contour",
    "four_point_transform",
    "warp_document_pipeline",
    # Module 3
    "OrientationCorrector",
    # Module 4
    "convert_to_grayscale",
    "correct_illumination",
    "sharpen_text",
    "adaptive_threshold",
    "enhance_document",
]
