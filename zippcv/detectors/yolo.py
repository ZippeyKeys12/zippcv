from typing import Callable, List, Optional, Tuple

import imutils
import numpy as np
from imutils.video import FPS

import cv2


class YOLOObjectDetector:
    def __init__(self, drawer:
                 Optional[Callable[[np.array, str, float,
                                    Tuple[int, int, int, int],
                                    Tuple[int, int, int], int], None]] = None,
                 confidence: float = .5,
                 threshold: float = .3,
                 includes: Optional[List[str]] = None,
                 excludes: Optional[List[str]] = None):
        self.drawer = drawer
        self.confidence = confidence
        self.threshold = threshold
        self.includes = includes
        self.excludes = excludes

        model = "coco/{}".format

        with open(model("coco.names")) as f:
            self.LABELS = f.read().strip().split("\n")

        if includes is not None and excludes is not None:
            raise ValueError(
                "'includes' and 'excludes' can't both not be 'None'")

        np.random.seed(42)
        self.COLORS = np.random.randint(
            0, 255, size=(len(self.LABELS), 3), dtype="uint8")

        self.net = cv2.dnn.readNetFromDarknet(
            model("yolov3.cfg"), model("yolov3.weights"))
        ln = self.net.getLayerNames()
        self.ln = [ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

    def __call__(self, image: np.array):
        (H, W) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(
            image, 1 / 255.0, (448, 448), swapRB=True, crop=False)

        self.net.setInput(blob)
        outputs = self.net.forward(self.ln)

        boxes = []
        confidences = []
        classIDs = []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)

                if (self.includes is not None and
                        self.LABELS[classID] not in self.includes):
                    continue

                if (self.excludes is not None and
                        self.LABELS[classID] in self.excludes):
                    continue

                confidence = scores[classID]

                if confidence > self.confidence:
                    box = detection[0:4] * np.array(2*[W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    X = int(centerX - (width / 2))
                    Y = int(centerY - (height / 2))

                    boxes.append([X, Y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        idxs = cv2.dnn.NMSBoxes(
            boxes, confidences, self.confidence, self.threshold)

        if len(idxs) > 0:
            for i in idxs.flatten():
                box = boxes[i]
                (X, Y) = (box[0], box[1])
                (W, H) = (box[2], box[3])

                rect = (X, Y, X + W, Y + H)

                classID = classIDs[i]

                info = (self.LABELS[classID], confidences[i],
                        rect, self.COLORS[classID])

                if self.drawer is not None:
                    self.drawer(image, *info, W)


def main():
    def draw(frame, label, confidence, rect, color, W):
        color = tuple(map(int, color))

        cv2.rectangle(frame, tuple(rect[:2]), tuple(rect[-2:]), color, 2)

        y = rect[1] - 10 if rect[1] - 10 > 10 else rect[1] + 10
        cv2.putText(frame, "{}: {:.4f}".format(label, confidence * 100),
                    (rect[0], y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    vs = cv2.VideoCapture("videos/static.mp4")

    fps = FPS().start()

    object_detector = YOLOObjectDetector(draw)

    for _ in range(675):
        vs.grab()

    while True:
        (_, frame) = vs.read()

        if frame is None:
            break

        frame = imutils.resize(frame, width=448)

        object_detector(frame)

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
