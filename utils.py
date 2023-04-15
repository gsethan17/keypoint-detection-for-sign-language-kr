from tensorflow.keras.utils import Sequence
import os
from glob import glob
import pandas as pd


class DataLoader(Sequence):
    def __init__(self, base_path, mode = 'Training'):
        self.base_path = base_path
        self.video_path = os.path.join(self.base_path, mode, 'video')

        self.info_df = pd.read_csv(os.path.join(base_path, mode, 'info.csv'))



if __name__ == '__main__':
    from config import args

    base_path = os.path.join(args.data_path, args.data_dir)

    dataLoader = DataLoader(base_path)
    print(dataLoader.base_path)