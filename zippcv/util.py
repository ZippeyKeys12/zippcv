import numpy as np
from PIL import Image

import cv2


def cv_to_pil(img: np.array) -> Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def pil_to_cv(img: Image) -> np.array:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
