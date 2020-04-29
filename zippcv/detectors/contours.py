from typing import Callable, Optional, Tuple

import imutils
import numpy as np

import cv2


class ContourDetector:
    def __init__(self, drawer:
                 Optional[Callable[[np.array, str, float,
                                    Tuple[int, int, int, int],
                                    Tuple[int, int, int], int], None]] = None,
                 iterations: int = 2, min_size: int = 500):
        self.drawer = drawer
        self.iterations = iterations
        self.min_size = min_size

    def __call__(self, image: np.array):
        threshold = cv2.dilate(image, None, iterations=self.iterations)
        contours = cv2.findContours(
            threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)

        for contour in contours:
            if cv2.contourArea(contour) < self.min_size:
                continue

            (X, Y, W, H) = cv2.boundingRect(contour)

            rect = (X, Y, X + W, Y + H)

            info = ("Motion", 1, rect, (0, 255, 0))

            if self.drawer is not None:
                self.drawer(image, *info, W)
