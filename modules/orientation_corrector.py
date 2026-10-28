import cv2
import numpy as np
from typing import Tuple, Optional


class OrientationCorrector:
    """
    Module 3: Hiệu chỉnh hướng văn bản tự động.
    Tích hợp: ORB Feature Detector, Hamming Matcher (KNN + Lowe's Ratio Test), 
    RANSAC Affine Estimation, Standalone Deskew và Rotate Fallback.
    """

    def __init__(
        self,
        n_features: int = 3000,
        min_matches: int = 12,
        ransac_threshold: float = 3.0,
        match_ratio: float = 0.75,
    ):
        """
        :param n_features: Số điểm đặc trưng ORB tối đa trích xuất.
        :param min_matches: Số lượng điểm khớp tối thiểu để thực thi RANSAC.
        :param ransac_threshold: Ngưỡng loại bỏ điểm nhiễu RANSAC (tính bằng pixel).
        :param match_ratio: Tỷ lệ lọc Lowe's Ratio Test cho Hamming Distance.
        """
        self.n_features = n_features
        self.min_matches = min_matches
        self.ransac_threshold = ransac_threshold
        self.match_ratio = match_ratio

        # 1. ORB Detector được cấu hình tối ưu cho văn bản (lấy nét đa tỉ lệ)
        self.orb = cv2.ORB_create(
            nfeatures=self.n_features,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=31,
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=31,
            fastThreshold=20,
        )

        # 2. Hamming Matcher sử dụng thuật toán KNN
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Tăng cường độ tương phản văn bản bằng CLAHE để ORB bắt điểm chính xác hơn."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

    def extract_features(self, image: np.ndarray) -> Tuple[list, Optional[np.ndarray]]:
        """Trích xuất Keypoints và Descriptors từ ảnh đã qua tiền xử lý."""
        prep_img = self.preprocess(image)
        keypoints, descriptors = self.orb.detectAndCompute(prep_img, None)
        return keypoints, descriptors

    def filter_matches_knn(self, des_src: np.ndarray, des_ref: np.ndarray) -> list:
        """Lọc điểm khớp nhiễu bằng KNN Matcher và Lowe's Ratio Test trên khoảng cách Hamming."""
        matches = self.matcher.knnMatch(des_src, des_ref, k=2)
        good_matches = []
        for match in matches:
            if len(match) == 2:
                m, n = match
                if m.distance < self.match_ratio * n.distance:
                    good_matches.append(m)
        return good_matches

    def estimate_affine_matrix(
        self, src_img: np.ndarray, ref_img: np.ndarray
    ) -> Tuple[Optional[np.ndarray], float, bool]:
        """Tính toán ma trận Affine và góc xoay bằng RANSAC."""
        kp_src, des_src = self.extract_features(src_img)
        kp_ref, des_ref = self.extract_features(ref_img)

        if des_src is None or des_ref is None:
            return None, 0.0, False

        good_matches = self.filter_matches_knn(des_src, des_ref)

        if len(good_matches) < self.min_matches:
            return None, 0.0, False

        pts_src = np.float32([kp_src[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        pts_ref = np.float32([kp_ref[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # 3. RANSAC Ước lượng Ma trận Affine biến đổi (Xoay + Dịch chuyển + Tỉ lệ)
        matrix, inliers = cv2.estimateAffinePartial2D(
            pts_src,
            pts_ref,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_threshold,
            maxIters=2000,
            confidence=0.99,
        )

        if matrix is None:
            return None, 0.0, False

        angle_rad = np.arctan2(matrix[1, 0], matrix[0, 0])
        angle_deg = np.degrees(angle_rad)

        return matrix, angle_deg, True

    def deskew_standalone(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Tự động khử nghiêng góc nhỏ (Deskew) khi không truyền vào ảnh mẫu (ref_img)."""
        gray = self.preprocess(image)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        coords = np.column_stack(np.where(thresh > 0))
        if len(coords) == 0:
            return image, 0.0

        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        if abs(angle) > 45 or abs(angle) < 0.2:
            return image, 0.0

        # Xoay ảnh theo góc nghiêng tự do
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        deskewed = cv2.warpAffine(
            image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )
        return deskewed, angle

    def rotate_fallback(self, image: np.ndarray) -> np.ndarray:
        """4. Rotate Fallback: Đưa khung hình nằm ngang (Width > Height) về dáng đứng thẳng."""
        h, w = image.shape[:2]
        if w > h:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        return image

    def process(self, src_img: np.ndarray, ref_img: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Hàm thực thi chính:
        - TH1: Có ref_img -> Nắn chỉnh tọa độ chính xác theo ảnh mẫu dùng RANSAC.
        - TH2: Không có ref_img hoặc RANSAC thất bại -> Dùng Deskew độc lập + Rotate Fallback.
        """
        if src_img is None or src_img.size == 0:
            raise ValueError("Ảnh đầu vào không hợp lệ hoặc rỗng!")

        if ref_img is not None and ref_img.size > 0:
            matrix, angle, success = self.estimate_affine_matrix(src_img, ref_img)
            if success and abs(angle) > 0.2:
                h_ref, w_ref = ref_img.shape[:2]
                corrected_img = cv2.warpAffine(
                    src_img,
                    matrix,
                    (w_ref, h_ref),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REPLICATE,
                )
                return self.rotate_fallback(corrected_img)

        # Nếu không có ảnh ref hoặc đối sánh thất bại
        deskewed_img, _ = self.deskew_standalone(src_img)
        return self.rotate_fallback(deskewed_img)
