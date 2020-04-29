from typing import Tuple

import imutils
import numpy as np
from imutils.video import FPS

import cv2


class SaliencyDetector:
    saliency_types = {
        "static": cv2.saliency.StaticSaliencyFineGrained_create,
        "motion": cv2.saliency.MotionSaliencyBinWangApr2014_create
    }

    def __init__(self, saliency_type: str = "static",
                 frame_size: Tuple[int, int] = (0, 0)):
        self.saliency_type = saliency_type
        self.frame_size = frame_size[::-1]

        try:
            self.saliency = self.saliency_types[saliency_type]()
        except KeyError:
            raise ValueError("'saliency_type' must be in {}".format(
                list(self.saliency_types.keys())))

        if saliency_type == "motion":
            self.saliency.setImagesize(*self.frame_size)
            self.saliency.init()

    def __call__(self, image: np.array):
        (H, W) = image.shape[:2]
        if self.frame_size != (W, H):
            self.frame_size = (W, H)

            if self.saliency_type == "motion":
                self.saliency = self.saliency_types["motion"]()
                self.saliency.setImagesize(*self.frame_size)
                self.saliency.init()

        if self.saliency_type == "motion":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        (success, saliency) = self.saliency.computeSaliency(image)

        if success:
            if self.saliency_type == "static":
                return cv2.threshold(saliency.astype("uint8"), 0, 255,
                                     cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

            elif self.saliency_type == "motion":
                return (saliency * 255).astype("uint8")

        return np.zeros(self.frame_size[::-1]+(3,), np.uint8)


def main():
    vs = cv2.VideoCapture("videos/static.mp4")

    saliency_type = "motion"

    saliency_detector = SaliencyDetector(saliency_type)

    fps = FPS().start()

    while True:
        _, frame = vs.read()

        if frame is None:
            break

        frame = imutils.resize(frame, width=448)

        saliency = saliency_detector(frame)

        cv2.imshow("Frame", frame)

        cv2.imshow("Saliency", saliency)

        key = cv2.waitKey(30) & 0xff

        if key == 27:
            break

        fps.update()

    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    cv2.destroyAllWindows()
    vs.release()


if __name__ == "__main__":
    main()
