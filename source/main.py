from Project import Project
import torch
from data.VideoDataset import VideoDataset
from pathlib import Path
from data.transformation import Yolov3Transform
from detection import Yolov3Detector
import cv2
import time
import numpy as np
from torchvision.transforms import Compose, Lambda, ToTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass
import pprint

classes = {0: 'player', 1: 'time', 2: 'stocks', 3: 'damage'}
transform = Compose([
    Yolov3Transform(),
    ToTensor(),
])


@dataclass
class Yolov3Prediction:
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
                'coord': [x1.int(), y1.int(),
                          x2.int(), y2.int()],
                'confidence': conf,
                'class': cls
            })

        return pred_json


detector = Yolov3Detector(weights='./yolov3/weights/best.pt',
                  cfg='./yolov3/cfg/yolov3-tiny-frames.cfg',
                  view_img=True,
                  classes=classes,
                  transform=transform)

root = Project().data_dir / 'videos' / 'evo2014' / 'frames'

img = cv2.imread('./yolov3/data/samples/830.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
preds = detector([img], conf_thres=0.5)
yolov3_pred = Yolov3Prediction(preds[0])
pprint.pprint(yolov3_pred)
# fig = plt.figure()
# plt.ion()
#
# cap = cv2.VideoCapture(
#     str(Project().data_dir / 'videos' / 'evo2014' / 'Axe four stocks SilentWolf in less than a minute Evo 2014.mp4'))
# im = None
# i = 0
#
# while (cap.isOpened()):
#     ret, frame = cap.read()
#     i += 1
#     if i > 60 * 3:
#         if i % 2 == 0:
#             if im is None: im = plt.imshow(frame)
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             preds = detector([frame], conf_thres=0.5)
#             img = detector.add_bb_on_img(frame, preds[0])
#             im.set_array(img)
#             plt.pause(0.001)
#
# plt.ioff()
# plt.show()
