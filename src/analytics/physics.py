import cv2
import numpy as np

class FieldTransformer:
    def __init__(self):
        self.homography_matrix = None
        self.is_calibrated = False
        self.real_width = 40.32
        self.real_height = 16.5

    def auto_calibrate(self, frame):
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Green mask (grass detection)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        green_ratio = np.sum(green_mask > 0) / (h * w)

        if green_ratio < 0.10:
            return False

        # White lines mask 
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        white_on_grass = cv2.bitwise_and(white_mask, green_mask)

        # Bounding box
        green_coords = cv2.findNonZero(green_mask)
        if green_coords is None:
            return False

        x_g, y_g, w_g, h_g = cv2.boundingRect(green_coords)

        pad_x = int(w_g * 0.05)
        pad_y = int(h_g * 0.05)

        tl = [x_g + pad_x,           y_g + pad_y]
        tr = [x_g + w_g - pad_x,     y_g + pad_y]
        bl = [x_g + pad_x,           y_g + h_g - pad_y]
        br = [x_g + w_g - pad_x,     y_g + h_g - pad_y]

        image_points = np.array([tl, tr, bl, br], dtype="float32")

        real_points = np.array([
            [0, 0],
            [self.real_width, 0],
            [0, self.real_height],
            [self.real_width, self.real_height]
        ], dtype="float32")

        self.homography_matrix = cv2.getPerspectiveTransform(image_points, real_points)
        self.is_calibrated = True
        return True

    def transform_point(self, x, y):
        if not self.is_calibrated:
            return np.array([x / 10.0, y / 10.0])

        point = np.array([[[x, y]]], dtype="float32")
        transformed = cv2.perspectiveTransform(point, self.homography_matrix)  # type: ignore
        return transformed[0][0]

    def calculate_speed(self, p1, p2, fps):
        dist = np.linalg.norm(np.array(p2) - np.array(p1))
        speed_kmh = dist * fps * 3.6
        return float(min(speed_kmh, 40.0))