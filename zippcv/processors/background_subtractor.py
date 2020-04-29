import imutils
import numpy as np
from imutils.video import FPS

import cv2


class BackgroundSubtractor:
    subtractor_types = {
        "MOG": cv2.bgsegm.createBackgroundSubtractorMOG,
        "MOG2": cv2.createBackgroundSubtractorMOG2,
        "GMG": cv2.bgsegm.createBackgroundSubtractorGMG
    }

    def __init__(self, subtractor_type: str = "GMG"):
        try:
            self.subtractor = self.subtractor_types[subtractor_type]()
        except KeyError:
            raise ValueError("'subtractor_type' must be in {}".format(
                list(self.subtractor_types.keys())))

        if subtractor_type == "GMG":
            self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        else:
            self.kernel = None

    def __call__(self, image: np.array):
        image = self.subtractor.apply(image)

        if self.kernel is not None:
            return cv2.morphologyEx(image, cv2.MORPH_OPEN, self.kernel)

        return image


def main():
    vs = cv2.VideoCapture("videos/static.mp4")

    # for _ in range(1000):
    #     vs.grab()

    background_subtractor = BackgroundSubtractor("MOG2")

    fps = FPS().start()

    while True:
        _, frame = vs.read()

        if frame is None:
            break

        frame = imutils.resize(frame, width=448)

        fgmask = background_subtractor(frame)

        cv2.imshow("Frame", frame)
        cv2.imshow("Mask", fgmask)
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
