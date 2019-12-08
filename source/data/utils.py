import shutil
from pathlib import Path

def copy_every_n_frame_to(in_dir: Path, out_dir: str, n=10):
    paths = list(in_dir.glob('*.jpg'))
    paths.sort(key=lambda a: int(a.name.split('.')[0]))
    out_path = '/home/francesco/Documents/Kanga-Challenge/source/dataset/yolo/frames/'
    for i, path in enumerate(paths):
        if i % 10 == 0:
            shutil.copy(str(path), f'{out_dir}{i}.jpg')