import time
import numpy as np
from dataclasses import dataclass


@dataclass
class ObjectDetection:
    coord: np.array
    conf: float
    crop: np.array
    cls: int
    timestamp: float = time.time()
    """
    Representation of a detection on an Image. This is model agnostic.
    """
    @staticmethod
    def from_yolov3(predictions, src, *args, **kwargs):
        predictions = predictions.numpy()
        for det in predictions:
            *coord, conf, _, cls = det
            coord = np.array(coord).astype(np.int)
            #             crop out from src image
            x1, y1, x2, y2 = coord
            crop = src[y1:y2, x1:x2]

            yield ObjectDetection(coord, conf, crop, int(cls))

    @property
    def area(self):
        # there is no angle
        x1, y1, x2, y2 = self.coord
        print(self.coord)
        b = x2 - x1 
        h = y2 - y1 
        return b * h
