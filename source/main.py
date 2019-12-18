import torch
import pprint
import threading
import cv2
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Project import Project
from data.VideoDataset import VideoDataset
from pathlib import Path
from data.transformation import Yolov3Transform
from detection import Yolov3Detector, OCRDetector, Detector, ObjectDetection
from detection.Yolov3Detector import Yolov3Prediction
from torchvision.transforms import Compose, Lambda, ToTensor
from torch.utils.data import DataLoader
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass, field
from utils import ThreadScheduler, DetectionPerClass, extract_only_numbers, crops_from_df_preds
from multiprocessing import Queue
from tqdm.autonotebook import tqdm
# TODO
# - [ ] try ocr with the new bb
# - [ ] if now we prediction, use the old one for class 1 and 3
from scipy.spatial.distance import cdist


class StockDetector(Detector):
    """
    Since each stooq is a square, we are going to compare the ratio of the rectangle
    """
    cls: int = 2

    def detect(self, img, det, history=None):
        # TODO  we can avoid false positve by using the history
        # to check if the value was increased or decreased before
        return np.round(img.shape[1] / img.shape[0])


classes = {0: 'player', 1: 'time', 2: 'stocks', 3: 'damage'}

transform = Compose([
    Yolov3Transform(),
    ToTensor(),
])


@dataclass
class RealTimeSmashBrosDetector(Detector):
    yolov3_detector: Yolov3Detector = Yolov3Detector(
        weights='./yolov3/weights/best.pt',
        cfg='./yolov3/cfg/yolov3-tiny-frames.cfg',
        view_img=False,
        classes=classes,
        transform=transform)
    ocr_detector: OCRDetector = OCRDetector(
        show_img=False,
        text_color=None,
        config='--psm 13 --oem 1 -c tessedit_char_whitelist=0123456789')
    stock_detector: StockDetector = StockDetector()
    frame_transform: callable = None
    show: bool = False
    skip_frames: int = 2
    history: [pd.DataFrame] = field(default_factory=list)

    def detect(self, stream, *args, **kwargs):
        im = None
        for i, frame in enumerate(stream):
            if i > 800:
                if self.show and im is None: im = plt.imshow(frame)
                if self.frame_transform is not None:
                    frame = self.frame_transform(frame)
                if i % self.skip_frames == 0:
                    preds = self.yolov3_detector(frame, *args, **kwargs)
                    if len(preds) > 0:
                        if self.show:
                            img = self.yolov3_detector.add_bb_on_img(
                                frame, preds)
                            im.set_array(img)
                        # convert pred to a pandas DataFrame
                        preds = pd.DataFrame(preds.numpy(),
                                             columns=[
                                                 'x', 'y', 'x2', 'y2', 'conf',
                                                 'foo', 'cls'
                                             ])

                        preds = preds.sort_values(by=['cls'])
                        # get out each crop
                        crops = list(crops_from_df_preds(preds, frame))
                        # value column will hold additional information
                        preds['value'] = None
                        # for each detection, extract more information
                        # for crop, (i, det) in zip(crops, preds.iterrows()):
                        #     if det.cls == 0:
                        #         # TODO find out which player is
                        #         pass
                        #     elif det.cls == 2:
                        #         stocks = self.stock_detector(
                        #             crop, det, self.history)
                        #         preds.loc[i, 'value'] = stocks
                        #     elif det.cls == 1 or det.cls == 3:
                        #         text = extract_only_numbers(
                        #             self.ocr_detector(crop))
                        #         preds.loc[i, 'value'] = text

                        print(preds)

                plt.pause(0.001)


# get and open the video
cap = cv2.VideoCapture(
    str(Project().data_dir / 'videos' / 'evo2014' /
        'Axe four stocks SilentWolf in less than a minute Evo 2014.mp4'))
detector = RealTimeSmashBrosDetector(
    show=True, frame_transform=lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2RGB))


def capread(cap):
    while (cap.isOpened()):
        ret, frame = cap.read()
        yield frame


fig = plt.figure()
plt.ion()

detector(capread(cap))

plt.ioff()
plt.show()
