from Project import Project
import torch
from data.VideoDataset import VideoDataset
from pathlib import Path
from yolov3.detect import Detect
import cv2
import time
# VideoDataset.from_yt('https://www.youtube.com/watch?v=bj7IX18ccdY', Project().data_dir / 'videos' / 'evo2014', force=True)
# VideoDataset.from_file(Path('/home/francesco/Documents/Kanga-Challenge/source/dataset/videos/Axe four stocks SilentWolf in less than a minute Evo 2014.mp4'))

# opt.cfg = 'cfg/yolov3-tiny-frames.cfg'
# opt.data = '/home/francesco/Documents/Kanga-Challenge/source/dataset/yolo/frames.data'
# opt.weights = 'weights/best.pt'
classes = {0: 'player', 1: 'time', 2: 'stocks', 3: 'damage'}
# datector = Detect(torch.device('cuda'), opt, classes=classes)
datector = Detect(torch.device('cpu'),
                  weights='./yolov3/weights/best.pt',
                  cfg='./yolov3/cfg/yolov3-tiny-frames.cfg',
                  classes=classes)
print(datector)
img = cv2.imread('./yolov3/data/samples/830.jpg')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
n = 0
start = time.time()
with torch.no_grad():
    while True:
        datector(img)
        n += 1
        print(f'FPS = {n / (time.time() - start)}')
