from Project import Project
import torch
from data.VideoDataset import VideoDataset
from pathlib import Path
from data.transformation import Yolov3Transform
from detection import Yolov3Detector, OCRDetector
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
from multiprocessing import Queue

from tqdm.autonotebook import tqdm
# TODO
# - [ ] try ocr with the new bb 
# - [ ] if now we prediction, use the old one for class 1 and 3

@dataclass
class Yolov3Prediction:
    """
    Representation of a YoloV3 prediction with superpowers.
    """
    pred: torch.Tensor

    def __getitem__(self, i):
        return self.pred[i]

    def to_JSON(self):
        """
        Convert prediction from the model to JSON format.
        """
        pred = self.pred.tolist()
        pred_json = []
        for *xyxy, conf, _, cls in pred:
            x1, y1, x2, y2 = xyxy
            pred_json.append({
                'coord': [x1, y1, x2, y2],
                'confidence': conf,
                'class': cls
            })

        return pred_json

    def cropped_images(self, src):
        """
        Generator that returns the cropped region and the class for each detection
        """
        for det in self.pred:
            *coord, conf, _, cls = det
            x1, y1, x2, y2 = coord
            crop = src[y1.int():y2.int(), x1.int():x2.int()]
            yield crop, cls.int().item()


classes = {0: 'player', 1: 'time', 2: 'stocks', 3: 'damage'}
transform = Compose([
    Yolov3Transform(),
    ToTensor(),
])
# create our detectors
detector = Yolov3Detector(weights='./yolov3/weights/best.pt',
                          cfg='./yolov3/cfg/yolov3-tiny-frames.cfg',
                          view_img=True,
                          classes=classes,
                          transform=transform)
ocr_detector = OCRDetector(show_img=False)


def smash_bros_detector(yolo_pred, my_queue):
    pred_json = yolo_pred.to_JSON()
    for pred, (crop, cls) in tqdm(zip(pred_json, yolo_pred.cropped_images(frame))):
        if cls == 1 or cls == 3:
            crop = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
            # crop = cv2.resize(crop, (crop.shape[1] * 2, crop.shape[0] * 2))
            text = ocr_detector([crop])
            pred['text'] = text
        elif cls == 0:
            pred['text'] = 'TODO'
        elif cls == 2:
            pred['text'] = 'TODO'

    my_queue.put(pred_json)


# create a multi thread queue
my_queue = Queue()
# x will hold our current thread
x = None
# get and open the video
cap = cv2.VideoCapture(
    str(Project().data_dir / 'videos' / 'evo2014' /
        'Axe four stocks SilentWolf in less than a minute Evo 2014.mp4'))

im = None
i = 0

while (cap.isOpened()):
    ret, frame = cap.read()
    i += 1
    if i > 60 * 3:
        if i % 2 == 0:
            if im is None: im = plt.imshow(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            preds = detector([frame], conf_thres=0.3)
            if len(preds) > 0:
                yolo_pred = Yolov3Prediction(preds[0])
                # we want to add further info to our prediction
                if x is None:
                    x = threading.Thread(target=smash_bros_detector,
                                        args=(yolo_pred, my_queue))
                    x.start()
                else:
                    if not x.isAlive():
                        pprint.pprint(my_queue.get())
                        x = threading.Thread(target=smash_bros_detector,
                                            args=(yolo_pred, my_queue))
                        x.start()
                img = detector.add_bb_on_img(frame, preds[0])
                im.set_array(img)
            plt.pause(0.001)
plt.ioff()
plt.show()
