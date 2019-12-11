from Project import Project
import torch
from data.VideoDataset import VideoDataset
from pathlib import Path
from yolov3.detect import Detect, Yolov3Transform
import cv2
import time
import numpy as np
from torchvision.transforms import Compose, Lambda, ToTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# VideoDataset.from_yt('https://www.youtube.com/watch?v=bj7IX18ccdY', Project().data_dir / 'videos' / 'evo2014', force=True)
# VideoDataset.from_file(Path('/home/francesco/Documents/Kanga-Challenge/source/dataset/videos/Axe four stocks SilentWolf in less than a minute Evo 2014.mp4'))

# opt.cfg = 'cfg/yolov3-tiny-frames.cfg'
# opt.data = '/home/francesco/Documents/Kanga-Challenge/source/dataset/yolo/frames.data'
# opt.weights = 'weights/best.pt'
classes = {0: 'player', 1: 'time', 2: 'stocks', 3: 'damage'}
# datector = Detect(torch.device('cuda'), opt, classes=classes)
transform = Compose([
                      Yolov3Transform(),
                      ToTensor(),
                  ])
detector = Detect(
                  weights='./yolov3/weights/best.pt',
                  cfg='./yolov3/cfg/yolov3-tiny-frames.cfg',
                  view_img=True,
                  classes=classes,
                  transform=transform)
print(detector)
img = cv2.imread('./yolov3/data/samples/830.jpg')
img =  cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# ds = VideoDataset(frames=np.array([img, img.copy()]),
#                   transform=Compose([
#                       Lambda(lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2RGB)),
#                       Yolov3Transform(),
#                       ToTensor(),
#                   ]),
#                   return_input=True)

# print(ds[0].shape)
# dl = DataLoader(ds, batch_size=8)
with torch.no_grad():
    # while True:
    preds = detector([img])
    detector.plot_pred_on_img(img, preds[0])
