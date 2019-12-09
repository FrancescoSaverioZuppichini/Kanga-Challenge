from Project import Project
from data.VideoDataset import VideoDataset
from pathlib import Path

VideoDataset.from_yt('https://www.youtube.com/watch?v=bj7IX18ccdY', Project().data_dir / 'videos' / 'evo2014', force=True)
# VideoDataset.from_file(Path('/home/francesco/Documents/Kanga-Challenge/source/dataset/videos/Axe four stocks SilentWolf in less than a minute Evo 2014.mp4'))