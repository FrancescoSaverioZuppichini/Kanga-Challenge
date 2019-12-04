import pytube
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from logger import logging

class VideoDataset(Dataset):
    def __init__(self, frames: np.array):
        self.frames = frames

    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass

    @classmethod
    def from_yt(csl, video_url: str, out_dir: Path):
        logging.info(f"Downloading data from {video_url}")
        if not out_dir.exists(): out_dir.mkdir()
        frames = []
        return VideoDataset(frames)

