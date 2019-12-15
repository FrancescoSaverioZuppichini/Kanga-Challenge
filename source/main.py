from Project import Project
import torch
from data.VideoDataset import VideoDataset
from pathlib import Path
from data.transformation import Yolov3Transform
from detection import Yolov3Detector, OCRDetector, Detector
from detection.Yolov3Detector import Yolov3Prediction
import cv2
import time
import numpy as np
from torchvision.transforms import Compose, Lambda, ToTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass
import pprint
import threading
import re
from multiprocessing import Queue

from tqdm.autonotebook import tqdm
# TODO
# - [ ] try ocr with the new bb
# - [ ] if now we prediction, use the old one for class 1 and 3

classes = {0: 'player', 1: 'time', 2: 'stocks', 3: 'damage'}
transform = Compose([
    Yolov3Transform(),
    ToTensor(),
])


@dataclass
class SmashBrosDetector(Detector):
    yolov3_detector: Yolov3Detector = Yolov3Detector(
        weights='./yolov3/weights/best.pt',
        cfg='./yolov3/cfg/yolov3-tiny-frames.cfg',
        view_img=True,
        classes=classes,
        transform=transform)
    ocr_detector: OCRDetector = OCRDetector(
        show_img=False,
        text_color=None,
        config='--psm 13 --oem 1 -c tessedit_char_whitelist=0123456789')

    def detect(self, imgs, *args, **kwargs):
        frame = imgs[-1]
        preds = self.yolov3_detector([frame], *args, **kwargs)
        if len(preds) > 0:
            preds = Yolov3Prediction(preds[0])
            preds = self.add_gui(preds, frame)
        return pred

    def add_gui(self, yolo_pred, frame):
        pred_json = yolo_pred.to_JSON()
        for pred, (crop,
                   cls) in tqdm(zip(pred_json,
                                    yolo_pred.cropped_images(frame))):
            if cls == 1 or cls == 3:
                text = self.ocr_detector([crop])[0]
                # replace 'o's with zeros
                text = text.replace('O', '0')
                text = text.replace('o', '0')
                text = re.findall(r'\d+', text)
                pred['text'] = text
            # elif cls == 0:
            #     pred['text'] = 'TODO'
            # elif cls == 2:
            #     pred['text'] = 'TODO'
        # pprint.pprint(pred_json)
        return pred_json


@dataclass
class RealTimeSmashBrosDetector(SmashBrosDetector):
    queue: Queue = Queue()
    skip_frames: int = 2
    frame_transform: callable = None
    _th: threading.Thread = None
    show: bool = False

    def detect(self, stream, *args, **kwargs):
        im = None
        for i, frame in enumerate(stream):
            if self.show and im is None: im = plt.imshow(frame)
            if self.frame_transform is not None:
                frame = self.frame_transform(frame)
            if i % self.skip_frames == 0:
                preds = self.yolov3_detector([frame], *args, **kwargs)
                if len(preds) > 0:
                    preds = Yolov3Prediction(preds[0])
                    th = self.get_th(preds, frame)
                    if not th.isAlive(): 
                        th.start()
                    if self.show:
                        img = self.yolov3_detector.add_bb_on_img(frame, preds)
                        im.set_array(img)
            plt.pause(0.001)

    def add_gui(self, *args, **kwargs):
        res = super().add_gui(*args, **kwargs)
        pprint.pprint(res)
        # little trick, fetch the superclass output and store it into our multi preocess queue
        # self.queue.put(res)

    def get_th(self, *args):
        if self._th is None or not self._th.isAlive():
            self._th = threading.Thread(target=self.add_gui, args=(args))

        return self._th


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
