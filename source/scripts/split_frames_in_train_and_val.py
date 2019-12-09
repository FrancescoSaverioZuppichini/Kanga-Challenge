import pandas as pd
from Project import Project
from logger import logging
from tqdm import tqdm
import numpy as np
import shutil
from tqdm import tqdm

project = Project()

root = project.data_dir / 'yolo/frames_copy/'
train_dir = project.data_dir / 'yolo/train'
val_dir = project.data_dir / 'yolo/val'

train_dir.mkdir(exist_ok=True)
val_dir.mkdir(exist_ok=True)

print(project)
labels = root.glob('*.txt')

ids = np.array([x.name.split('.')[0] for x in labels])
train_size = int(len(ids) * 0.8)
np.random.shuffle(ids)
train_ids, val_ids = ids[:train_size], ids[train_size:]

print(f'train_size={train_ids.shape[0]}, val_size={val_ids.shape[0]}')


def move_im_and_bb(id, out_dir):
    shutil.copy(str(root / f'{id}.jpg'), str(out_dir / f'{id}.jpg'))
    shutil.copy(str(root / f'{id}.txt'), str(out_dir / f'{id}.txt'))


list(tqdm(map(lambda x: move_im_and_bb(x, train_dir), train_ids)))
list(tqdm(map(lambda x: move_im_and_bb(x, val_dir), val_ids)))
