import tensorflow as tf
from tensorflow.keras.utils import Sequence
import os
from glob import glob
import pandas as pd
import numpy as np
import math
import cv2
from tqdm import tqdm


class DataLoader(Sequence):
    def __init__(self, base_path, input_shape = False, batch_size = 256, shuffle = True, 
                        rate_downsampling = 1, mode = 'Training'):
        self.base_path = base_path
        self.video_path = os.path.join(self.base_path, mode, 'video')
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.shuffle = shuffle
        self.rate_downsampling = rate_downsampling

        self.info_df = pd.read_csv(os.path.join(base_path, mode, 'info.csv'))
        # print(len(self.info_df[self.info_df['label'] == '곳']))
        self.target_label = self.get_label()

        self.info_sample = self.get_sample_level_info()

        self.on_epoch_end()

    def get_onehot_y(self, y_index):
        onehot_y = np.zeros((len(y_index), len(self.target_label)))

        for i, idx in enumerate(y_index):
            onehot_y[i, idx] = 1.

        return onehot_y


    def __getitem__(self, idx):
        indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]

        # get images
        video_names = [self.info_sample['video_name'][i] for i in indices]
        frames = [self.info_sample['global_frame'][i] for i in indices]

        resize = True if self.input_shape else False

        batch_img = self.read_frames(video_names, frames, resize)

        # get onehot y
        labels = [self.info_sample['label'][i] for i in indices]
        labels_int = [self.target_label.index(label) for label in labels]
        batch_onehot_y = self.get_onehot_y(labels_int)
        # print(batch_onehot_y)
        

        batch_x = tf.convert_to_tensor(batch_img, dtype=tf.float32)
        batch_y = tf.convert_to_tensor(batch_onehot_y, dtype=tf.float32)

        return batch_x, batch_y
    
    def on_epoch_end(self):
        self.indices = np.arange(len(self.info_sample['label']))
        if self.shuffle :
            np.random.shuffle(self.indices)

    def get_sample_video(self, label, resize):

        idx = self.info_sample['label'].index(label)
        video_name = self.info_sample['video_name'][idx]
        dur_frame = self.info_sample['duration_frame'][idx]
        
        i = idx 
        while True:
            i += 1
            if self.info_sample['duration_frame'][i] != dur_frame:
                break


        global_frames = self.info_sample['global_frame'][idx:i]

        imgs = self.read_frames([video_name for i in range(len(global_frames))], global_frames, resize)
        w = imgs[0].shape[1]
        h = imgs[0].shape[0]

        out = cv2.VideoWriter(os.path.join(os.getcwd(), 'sub_KSL_{}.mp4'.format(label)),
                        cv2.VideoWriter_fourcc(*'DIVX'),
                        self.info_sample['fps'][idx],
                        (w, h),
                        )

        for img in imgs:
            out.write(img)

        out.release()


    
    def get_sample_level_info(self):
        print("[I] Data is being loaded frame by frame.")
        dic = {}
        dic['video_name'] = []
        dic['fps'] = []
        dic['label'] = []
        dic['duration_frame'] = []
        dic['global_frame'] = []
        dic['local_frame'] = []

        for i in tqdm(range(len(self.info_df))):
            cur_info = self.info_df.loc[i]
            video_name = cur_info['video_name']
            dur_time = cur_info['duration']
            label = cur_info['label']
            start_sec = cur_info['start']
            end_sec = cur_info['end']

            video = cv2.VideoCapture(os.path.join(self.video_path, video_name))

            fps = video.get(cv2.CAP_PROP_FPS)
            total_num_frame = video.get(cv2.CAP_PROP_FRAME_COUNT)
            total_sec = total_num_frame / fps

            try:
                if abs(round(total_sec, 3) - round(dur_time, 3)) > 0.03:
                    raise Exception("Duration time is incorrect.", video_name)
            except:
                continue

            start_frame = int(start_sec * fps)
            end_frame = int(end_sec * fps)
            dur_frame = end_frame - start_frame

            for frame in range(start_frame, end_frame, self.rate_downsampling):
                global_frame = frame
                local_frame = frame-start_frame
        
                dic['video_name'].append(video_name)
                dic['fps'].append(fps)
                dic['label'].append(label)
                dic['duration_frame'].append(dur_frame)
                dic['global_frame'].append(global_frame)
                dic['local_frame'].append(local_frame)

            
            # if len(dic['label']) > 100:
                # break

        return dic


    
    def read_frames(self, video_names, frames, resize=False):
        imgs = []

        for i in range(len(frames)):
            video = cv2.VideoCapture(os.path.join(self.video_path, video_names[i]))
            video.set(cv2.CAP_PROP_POS_FRAMES, frames[i])

            flag, img = video.read()
            
            if not flag:
                raise "There is no {}th frame in {}".format(start_frame, video_name)

            if resize:
                w = img.shape[1]
                h = img.shape[0]

                crop_size = w - h

                img = img[:, int(crop_size/2):int(-1*(crop_size/2)),:]

                if self.input_shape:
                    img = cv2.resize(img, self.input_shape)

            imgs.append(img)

        return imgs
        


    def get_label(self):
        return list(self.info_df['label'].unique())



    def __len__(self):
        return math.ceil(len(self.info_sample['label']) / self.batch_size)

        



if __name__ == '__main__':
    from config import args

    base_path = os.path.join(args.data_path, args.data_dir)
    input_shape = (224, 224)

    dataLoader = DataLoader(base_path,
            batch_size = 5,
            input_shape = input_shape,
            rate_downsampling=1)

    # df = dataLoader.info_df

    for x, y in dataLoader:
        print(x.shape, y.shape)
    # dataLoader.get_sample_video(dataLoader.target_label[0], resize=True)





