import pandas as pd
from Project import Project
from logger import logging
from tqdm import tqdm

project = Project()

root = project.data_dir / 'yolo/frames_copy/'
print(project)


def read_bb(path):
    return pd.read_csv(path, delimiter=' ', names=['class', 'x', 'y', 'x2', 'y2'], header=None)


df_with_gui = read_bb(root / '210.txt')

gui_boxes_df = df_with_gui[df_with_gui['class'] != 0]
gui_boxes_df.reset_index(drop=True, inplace=True)
gui_boxes_df


def add_gui(df, df_gui):
    df = df[df['class'] == 0]
    df = df.reset_index(drop=True)
    return pd.concat([df, df_gui], axis=0).reset_index(drop=True)


class AddGuiToBB:
    def __init__(self, root, gui_boxes_df):
        self.file_paths = list(root.glob('*.txt'))
        self.dfs = list(map(read_bb, self.file_paths))
        self.gui_boxes_df = gui_boxes_df

    def __call__(self):
        bar = tqdm(zip(self.dfs, self.file_paths))
        for df, file_path in bar:
            x = add_gui(df, self.gui_boxes_df)
            x.to_csv(str(file_path), sep=' ', index=False, header=False)
            bar.set_description(f'{file_path.name}')


# temp = read_bb(root / '250.txt')
# add_gui(temp, gui_boxes_df)
adder = AddGuiToBB(root, gui_boxes_df)
adder()
