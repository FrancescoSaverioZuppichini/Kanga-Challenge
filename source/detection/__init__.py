import cv2
import numpy as np
import time as time
import matplotlib.pyplot as plt 
import matplotlib.patches as patches
from logger import logging 
from abc import ABC, abstractmethod

class Detector:
    def __call__(self, frames: list = [], *args, **kwargs) -> dict:
        res = self.detect(frames, *args, **kwargs)
        return res

    @abstractmethod
    def detect(self, images: list = [], *args, **kwargs) -> dict:
        pass

    @abstractmethod
    def to_JSON() -> dict:
        pass


from detection import Detector
from logger import logging
import matplotlib.patches as patches


class EASTTextDetector(Detector):
    """
    Uses EAST algorithm based on opencv to find text in a picture.
    """
    def __init__(self):
        super().__init__()
        self.net = cv2.dnn.readNet(
            '/home/francesco/Documents/Kanga-Challenge/source/checkpoint/east/model.pb'
        )
        self.layers = [
            "feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"
        ]

    def detect(self, frames):
        #         get the last frame in the buffer
        img = frames[-1]
        H, W, c = img.shape
        start = time.time()
        blob = cv2.dnn.blobFromImage(img,
                                     1.0, (H, W), (123.68, 116.78, 103.94),
                                     swapRB=True,
                                     crop=False)
        self.net.setInput(blob)
        (scores, geometry) = self.net.forward(self.layers)
        self.scores, self.geometry = scores.squeeze(), geometry.squeeze()
        end = time.time()
        logging.info("Text detection took {:.6f} seconds".format(end - start))

        self.all_boxes, self.confidences, self.indices, (h,
                                                         w) = self.get_boxes(
                                                             self.geometry,
                                                             self.scores,
                                                             img.shape)
        self.boxes = self.nms(self.all_boxes, self.geometry, self.confidences,
                              self.indices, h, w)
        return self

    def get_boxes(self, geometry, scores, original_shape, factor=4):
        grid, _ = np.meshgrid(np.arange(original_shape[0] // 4),
                              np.arange(original_shape[0] // 4))

        cos = np.cos(geometry[4])
        sin = np.sin(geometry[4])

        h = geometry[0] + geometry[2]
        w = geometry[1] + geometry[3]

        x2 = grid * factor + cos * geometry[1] + sin * geometry[2]
        y2 = grid.T * factor - sin * geometry[1] + cos * geometry[2]
        x1 = x2 - w
        y1 = y2 - h

        indices = np.where(scores.reshape(-1) > 0.8)[0]

        boxes = np.stack([x1, y1, x2, y2])
        boxes = boxes.reshape((4, -1))
        boxes = boxes.transpose(1, 0)
        boxes = boxes[indices]
        boxes = boxes.astype(np.uint32)
        confidences = scores.reshape(-1)[indices]

        return boxes, confidences, indices, (h, w)

    def nms(self, boxes, geometry, confidences, indices, h, w):
        confThreshold = 0.5
        nmsThreshold = 0.3
        #         convert boxes in the correct format
        centers = np.array((0.5 * (boxes[:, 0] + boxes[:, 2]),
                            0.5 * (boxes[:, 1] + boxes[:, 3])))
        centers = centers.transpose(1, 0)
        angles = -1 * geometry[4].flatten()[indices] * 180 / np.pi
        dimensions = np.array(
            (w.flatten()[indices], h.flatten()[indices])).transpose(1, 0)
        detections = []
        for center, angle, dim in zip(centers, angles, dimensions):
            detections.append(
                ((center[0], center[1]), (dim[0], dim[1]), angle))
        confThreshold = 0.5
        nmsThreshold = 0.3
        indices = cv2.dnn.NMSBoxesRotated(detections, confidences,
                                          confThreshold, nmsThreshold)
        return boxes[indices].squeeze()

    def plot_boxes(self, img):
        fig, ax = plt.subplots(1)
        plt.imshow(img)

        for box in self.boxes:
            x1, y1, x2, y2 = box
            rect = patches.Rectangle((x1, y1),
                                     x2 - x1,
                                     y2 - y1,
                                     linewidth=1,
                                     edgecolor='r',
                                     facecolor='none')
            # Add the patch to the Axes
            ax.add_patch(rect)
        plt.show()
        return fig, ax