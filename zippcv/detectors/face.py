from typing import Callable, Optional, Tuple

import imutils
import numpy as np
from imutils.video import FPS

import cv2


class FaceDetector:
    def __init__(self, drawer:
                 Optional[Callable[[np.array, str, float,
                                    Tuple[int, int, int, int],
                                    Tuple[int, int, int], int], None]] = None,
                 confidence: float = .5):
        self.drawer = drawer
        self.confidence = confidence

        self.net = cv2.dnn.readNetFromCaffe(*map("googlenet/{}".format, [
            "deploy.prototxt",
            "res10_300x300_ssd_iter_140000.caffemodel"
        ]))

    def __call__(self, image: np.array):
        (H, W) = image.shape[:2]

        # image = cv2.resize(image, (300, 300))
        # image = imutils.resize(image, width=300)

        blob = cv2.dnn.blobFromImage(
            image, 1.0, (500, 500), (104.0, 177.0, 123.0), crop=False)

        self.net.setInput(blob)
        detections = self.net.forward()

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > self.confidence:
                box = (detections[0, 0, i, 3:7] *
                       np.array(2*[W, H])).astype("int")

                info = ("Face", confidence, box, (0, 0, 255))

                if self.drawer is not None:
                    self.drawer(image, *info, W)


def main():
    def draw(frame, label, confidence, rect, color, W):
        color = tuple(map(int, color))

        cv2.rectangle(frame, tuple(rect[:2]), tuple(rect[-2:]), color, 2)

        y = rect[1] - 10 if rect[1] - 10 > 10 else rect[1] + 10
        cv2.putText(frame, "{}: {:.4f}".format(label, confidence * 100),
                    (rect[0], y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    vs = cv2.VideoCapture("videos/race.mp4")

    fps = FPS().start()

    face_detector = FaceDetector(draw)

    for _ in range(675):
        vs.grab()

    while True:
        (_, frame) = vs.read()

        if frame is None:
            break

        frame = imutils.resize(frame, width=500)

        face_detector(frame)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

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
