from Project import Project
from data.VideoDataset import VideoDataset

VideoDataset.from_yt('https://www.youtube.com/watch?v=bj7IX18ccdY', Project().data_dir / 'evo-2014')