import tensorflow as tf
from tensorflow.keras.utils import Sequence
import os
from glob import glob
import pandas as pd
import numpy as np
import math
import cv2
from tqdm import tqdm
import json
import shutil

import time

class DataLoader_keypoint():
    def __init__(self, mode, isNorm, batch_size, shuffle):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.isNorm = isNorm
        self.del_paths = []
        
        self.mode = mode
        self.data_dir = os.path.join(os.getcwd(), 'data', 'keypoint')
        print(self.data_dir)
        
        if not os.path.isdir(self.data_dir):
            os.makedirs(os.path.join(self.data_dir, self.mode))
            
            results = {}
            results['video_name'] = []
            results['global_frame'] = []
            results['label'] = []
            results['results'] = []
        
            self.base_dir = os.path.join('/data', 'sub_KSL', self.mode)
            self.keypoint_dir = os.path.join(self.base_dir, 'keypoint')
            
            if self.mode == 'Validation':
                df_info = pd.read_csv(os.path.join(self.base_dir, 'info_sample.csv'))
            else:
                with open(os.path.join(self.base_dir, 'info_sample.json')) as f:
                    info = json.load(f)
                df_info = pd.DataFrame(info)
        
            start_points = df_info[df_info['local_frame'] == 0].copy()
            
            for i in tqdm(range(len(start_points))):
            # for i in range(len(start_points)):
                start_sample = start_points.iloc[i]

                video_name = start_sample['video_name']
                num_frame = start_sample['global_frame']
                fps = start_sample['fps']
                label = start_sample['label']
                dur = start_sample['duration_frame']
                

                condi1 = (df_info['video_name'] == video_name)
                condi2 = (df_info['fps'] == fps)
                condi3 = (df_info['label'] == label)
                condi4 = (df_info['duration_frame'] == dur)

                df_total_data = df_info[condi1 & condi2 & condi3 & condi4]
                frames = list(df_total_data['global_frame'])
                
                
                w_mor = None
                h_mor = None
                for i, frame in enumerate(frames):
                    results['video_name'].append(video_name)
                    results['global_frame'].append(frame)
                    results['label'].append(label)
                    save_path = os.path.join(self.data_dir, self.mode, "{}_{}_{}.npy".format(video_name.split('.')[0], str(frame).zfill(10), label))
                    
                    # print(video_name, frame, label)
                    if os.path.isfile(save_path):
                        results['results'].append('Already')
                        # print("Already")
                        continue
                    
                    try:
                        pose, left_hd, right_hd = self.get_valid_keypoint(video_name.split('.')[0], frame)
                    except:
                        pose == None
                        left_hd == None
                        right_hd == None
                    if pose == None or left_hd == None or right_hd == None:
                        results['results'].append('None')
                        # print("fail")
                        continue
                    arm_ws, arm_hs, left_ws, left_hs, right_ws, right_hs, error_flag = self.separate_keypoint(pose, left_hd, right_hd)
                    if error_flag:
                        results['results'].append('None')
                        # print("fail")
                        continue
                    
                    w = np.concatenate((np.array(arm_ws).reshape(1, -1), np.array(left_ws).reshape(1, -1), np.array(right_ws).reshape(1, -1)), axis=-1)
                    h = np.concatenate((np.array(arm_hs).reshape(1, -1), np.array(left_hs).reshape(1, -1), np.array(right_hs).reshape(1, -1)), axis=-1)
                    
                    try:
                        if i == 0:
                            w_mor = w
                            h_mor = h
                        else:
                            w_mor = np.concatenate((w_mor, w), axis=0)  # (num_frame, num_point)
                            h_mor = np.concatenate((h_mor, h), axis=0)  # (num_frame, num_point)
                    except:
                        results['results'].append('ambiguous')
                        continue
                        print(arm_ws, arm_hs, left_ws, left_hs, right_ws, right_hs)
                    results['results'].append('success')
                        
                
                keypoint_input = np.concatenate((np.expand_dims(w_mor, axis=-1), np.expand_dims(h_mor, axis=-1)), axis=-1) # (num_frame, num_point*2)
                np.save(os.path.join(self.data_dir, self.mode, "{}_{}_{}.npy".format(video_name.split('.')[0], str(frame).zfill(10), label)), keypoint_input)
                
            pd.DataFrame(results).to_csv(os.path.join(self.data_dir, self.mode+'_results.csv'), index=False)
                
        self.keypoint_dir = os.path.join(os.getcwd(), 'data', 'keypoint', self.mode)

        self.get_path()
        self.on_epoch_end()
                
    def get_json(self, path):
        with open(path) as f:
            dic = json.load(f)

        return dic

    def get_valid_keypoint(self, video_name, num_frame):
        keypoint_paths = glob(os.path.join(self.keypoint_dir, video_name, '{}_*0{}_keypoints.json'.format(video_name, num_frame)))
        if not len(keypoint_paths) == 1:
            if len(keypoint_paths) > 1:
                print(os.listdir(os.path.join(self.keypoint_dir, video_name)))
            return None, None, None

        dic = self.get_json(keypoint_paths[0])
        people_dic = dic['people']

        pose_2d = people_dic['pose_keypoints_2d']
        hand_left_2d = people_dic['hand_left_keypoints_2d']
        hand_right_2d = people_dic['hand_right_keypoints_2d']

        return pose_2d, hand_left_2d, hand_right_2d

    def separate_keypoint(self, pose, left_hd, right_hd):
        neck_idx = 1
        right_shoulder_idx = 2
        right_elbow_idx = 3
        right_wrist_dix = 4
        left_shoulder_idx = 5
        left_elbow_idx = 6
        left_wrist_dix = 7
        right_pelvis_idx = 9
        left_pelvis_idx = 12

        # get center of body
        center_ws = []
        center_hs = []
        center_visibles = []
        for i in [neck_idx, right_pelvis_idx, left_pelvis_idx]:
            center_ws.append(pose[(i)*3:((i+1)*3)][0])
            center_hs.append(pose[(i)*3:((i+1)*3)][1])
            center_visibles.append(pose[(i)*3:((i+1)*3)][2])

        center_w = np.mean(center_ws)
        center_h = np.mean(center_hs)


        arm_ws = []
        arm_hs = []
        arm_visibles = []
        for j in [right_shoulder_idx, right_elbow_idx, right_wrist_dix, left_shoulder_idx, left_elbow_idx, left_wrist_dix]:
            arm_ws.append(pose[(j)*3:((j+1)*3)][0] - center_w)
            arm_hs.append(pose[(j)*3:((j+1)*3)][1] - center_h)
            arm_visibles.append(pose[(j)*3:((j+1)*3)][2])

        left_ws = []
        left_hs = []
        left_visibles = []
        for k in range(int(len(left_hd)/3)):
            left_ws.append(left_hd[k*3:((k+1)*3)][0] - center_w)
            left_hs.append(left_hd[k*3:((k+1)*3)][1] - center_h)
            left_visibles.append(left_hd[k*3:((k+1)*3)][2])

        right_ws = []
        right_hs = []
        right_visibles = []
        for z in range(int(len(right_hd)/3)):
            right_ws.append(right_hd[z*3:((z+1)*3)][0] - center_w)
            right_hs.append(right_hd[z*3:((z+1)*3)][1] - center_h)
            right_visibles.append(right_hd[z*3:((z+1)*3)][2])
            
        if 0 in center_visibles + arm_visibles + left_visibles + right_visibles:
            return None, None, None, None, None, None, True
            
            
        if self.isNorm:
            w_max, w_min = np.max(arm_ws + left_ws + right_ws), np.min(arm_ws + left_ws + right_ws)
            h_max, h_min = np.max(arm_hs + left_hs + right_hs), np.min(arm_hs + left_hs + right_hs)

            arm_ws = (arm_ws - w_min) / (w_max - w_min)
            arm_hs = (arm_hs - h_min) / (h_max - h_min)

            left_ws = (left_ws - w_min) / (w_max - w_min)
            left_hs = (left_hs - h_min) / (h_max - h_min)

            right_ws = (right_ws - w_min) / (w_max - w_min)
            right_hs = (right_hs - h_min) / (h_max - h_min)

        return arm_ws, arm_hs, left_ws, left_hs, right_ws, right_hs, False

    def get_path(self):
        path_lists = np.array(glob(os.path.join(self.keypoint_dir, '*.npy')))

        del_paths = np.load(os.path.join(os.getcwd(), 'EDA', 'del_path_1.npy'))

        '''
        for path in del_paths:
            idx = list(path_lists).index(path)
            path_lists = np.delete(path_lists, idx)
        '''


        if self.mode == 'Training':
            idx_list = [x for x in range(len(path_lists))]

            val_idx = np.random.choice(idx_list, size=int(len(path_lists)*0.2), replace=False)
            self.val_paths = path_lists[val_idx]
            self.train_paths = np.delete(path_lists, val_idx, axis=0)
            self.len_val = int(math.ceil(len(self.val_paths) / self.batch_size))-1

        elif self.mode == 'Validation':
            self.train_paths = path_lists

    def __len__(self):
        return int(math.ceil(len(self.train_paths)/self.batch_size))-1

    def on_epoch_end(self):
        self.indices = np.arange(len(self.train_paths))
        if self.mode == 'Training':
            self.indices_val = np.arange(len(self.val_paths))

        if self.shuffle:
            np.random.shuffle(self.indices)
            if self.mode == 'Training':
                np.random.shuffle(self.indices_val)


    def get_onehot(self, label):
        label_list = list(np.load(os.path.join(os.getcwd(), 'EDA', 'top100.npy')))

        try:
            idx = label_list.index(label)
        except:
            return None, False

        onehot_y = np.zeros(len(label_list))

        onehot_y[idx] = 1.

        return onehot_y, True
    
    def get_data(self, paths):
        xs = None
        ys = None
        flag = False

        for i, path in enumerate(paths):
            # get y
            label = os.path.basename(path).split('.')[0].split('_')[-1]

            onehot_y, y_flag = self.get_onehot(label)
            onehot_y = np.expand_dims(onehot_y, axis=0)
            if not y_flag:
                self.del_paths.append(path)
                continue
            
            x = np.load(path, allow_pickle=True)
            if len(x.shape) < 3:
                self.del_paths.append(path)
                continue
            x = x.reshape((-1, 48*2))

            masked_x = np.ones((300, 48*2)) * 9    # max length  295
            masked_x[:x.shape[0], :] = x

            masked_x = np.expand_dims(masked_x, axis=0)


            if not flag:
                xs = masked_x
                ys = onehot_y
                flag = True

            else:
                xs = np.concatenate((xs, masked_x), axis=0)
                ys = np.concatenate((ys, onehot_y), axis=0)

        
        return xs, ys

    def __getitem__(self, idx):
        indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]

        batch_path = self.train_paths[indices]
        batch_x, batch_y = self.get_data(batch_path)

        np.save(os.path.join(os.getcwd(), 'EDA', 'del_path.npy'), np.array(self.del_paths))

        return tf.convert_to_tensor(batch_x, dtype=tf.float32), tf.convert_to_tensor(batch_y, dtype=tf.float32)

    def get_val_item(self, idx):
        indices = self.indices_val[idx*self.batch_size:(idx+1)*self.batch_size]

        batch_path = self.val_paths[indices]
        batch_x, batch_y = self.get_data(batch_path)

        return tf.convert_to_tensor(batch_x, dtype=tf.float32), tf.convert_to_tensor(batch_y, dtype=tf.float32)


def gpu_limit(GB) :
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print("########################################")
    print('{} GPU(s) is(are) available'.format(len(gpus)))
    print("########################################")
    # set the only one GPU and memory limit
    memory_limit = 1024 * GB
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(gpus[0], [
                tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)])
            print("Use only one GPU{} limited {}MB memory".format(gpus[0], memory_limit))
        except RuntimeError as e:
            print(e)
    else:
        print('GPU is not available')

# class DataLoader():
class DataLoader(Sequence):
    def __init__(self, base_path, input_shape = False, batch_size = 256, shuffle = True, 
                        rate_downsampling = 1, mode = 'Training'):
        self.base_path = base_path
        self.mode = mode
        self.video_path = os.path.join(self.base_path, mode, 'video')
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.shuffle = shuffle
        self.rate_downsampling = rate_downsampling

        self.info_df = pd.read_csv(os.path.join(base_path, mode, 'info.csv'))
        # print(len(self.info_df[self.info_df['label'] == '곳']))
        self.target_label = self.get_label()

        self.info_sample_path = os.path.join(self.base_path, mode, 'info_sample.json')

        if os.path.isfile(self.info_sample_path):
            with open(self.info_sample_path) as f:
                self.info_sample = json.load(f)
        else:
            self.info_sample = self.get_sample_level_info()

            with open(self.info_sample_path, 'w') as f:
                json.dump(self.info_sample, f)

        self.on_epoch_end()

    def get_onehot_y(self, y_index):
        onehot_y = np.zeros((len(y_index), len(self.target_label)))

        for i, idx in enumerate(y_index):
            onehot_y[i, idx] = 1.

        return onehot_y

    def save_image(self):
        resize = True
        past_video = self.info_sample['video_name'][0]
        
        list_indices = list(self.indices)
        list_indices.reverse()
        
        for i in tqdm(self.indices):
            # get images
            video_name = self.info_sample['video_name'][i]
            dur_frame = self.info_sample['duration_frame'][i]
            local_frame = self.info_sample['local_frame'][i]
            global_frame = self.info_sample['global_frame'][i]
            label = self.info_sample['label'][i]
            
            img_path = os.path.join(self.base_path, self.mode, 'image', '{}_{}_{}_{}_{}.jpg'.format(video_name.split('.')[0], global_frame, local_frame, dur_frame, label))

            if os.path.isfile(img_path):
                continue
                
            
            if i == 0 or past_video != video_name:    
                video = cv2.VideoCapture(os.path.join(self.video_path, video_name))
                past_video = video_name
                
            if not video.get(cv2.CAP_PROP_POS_FRAMES) == global_frame:
                video.set(cv2.CAP_PROP_POS_FRAMES, global_frame)
            
            flag, img = video.read()
            
            if not flag:
                continue
                print("There is no {}th frame in {}".format(frames[i], video_names[i]))
                # raise "There is no {}th frame in {}".format(frames[i], video_names[i])

            if resize:
                w = img.shape[1]
                h = img.shape[0]

                if h > w:
                    crop_size = (h - w)/2
                    img = img[int(crop_size/2):int(-1*(crop_size/2)), :, :]
                else:
                    crop_size = w - h
                    img = img[:, int(crop_size/2):int(-1*(crop_size/2)),:]

                if self.input_shape:
                    img = cv2.resize(img, self.input_shape, interpolation=cv2.INTER_AREA)
            
            cv2.imwrite(img_path, img)
    
    def get_images(self, info):
        video_name, global_frame, local_frame, dur_frame, label = info
        img_path = os.path.join(self.base_path, self.mode, 'image', '{}_{}_{}_{}_{}.jpg'.format(video_name.split('.')[0], global_frame, local_frame, dur_frame, label))
        
        img = tf.io.read_file(img_path)
        img = tf.io.decode_jpeg(img, channels=3)
        # Resize the image to the desired size
        img = tf.image.resize(img, [self.input_shape[0], self.input_shape[1]])
        img = img / 255
        
        return img
    
    def __getitem__(self, idx):
        indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        
        # get images
        video_names = [self.info_sample['video_name'][i] for i in indices]
        dur_frames = [self.info_sample['duration_frame'][i] for i in indices]
        local_frames = [self.info_sample['local_frame'][i] for i in indices]
        global_frames = [self.info_sample['global_frame'][i] for i in indices]
        labels = [self.info_sample['label'][i] for i in indices]
        
        infos = zip(indices, video_names, global_frames, local_frames, dur_frames, labels)
        
        cal_indices = []
        for info in infos:
            index, video_name, global_frame, local_frame, dur_frame, label = info
            img_path = os.path.join(self.base_path, self.mode, 'image', '{}_{}_{}_{}_{}.jpg'.format(video_name.split('.')[0], global_frame, local_frame, dur_frame, label))
            if os.path.isfile(img_path):
                cal_indices.append(index)
        
        video_names = [self.info_sample['video_name'][i] for i in cal_indices]
        dur_frames = [self.info_sample['duration_frame'][i] for i in cal_indices]
        local_frames = [self.info_sample['local_frame'][i] for i in cal_indices]
        global_frames = [self.info_sample['global_frame'][i] for i in cal_indices]
        labels = [self.info_sample['label'][i] for i in cal_indices]
        
        infos = zip(video_names, global_frames, local_frames, dur_frames, labels)
        
        batch_img = [self.get_images(info) for info in infos]


        # get onehot y
        # labels = [self.info_sample['label'][i] for i in indices]
        labels_int = [self.target_label.index(label) for label in labels]
        batch_onehot_y = self.get_onehot_y(labels_int)
        
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
            if not fps > 0.0:
                continue
            total_num_frame = video.get(cv2.CAP_PROP_FRAME_COUNT)
            if not total_num_frame > 0.0:
                continue
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
                continue
                print("There is no {}th frame in {}".format(frames[i], video_names[i]))
                # raise "There is no {}th frame in {}".format(frames[i], video_names[i])

            if resize:
                w = img.shape[1]
                h = img.shape[0]

                if h > w:
                    crop_size = (h - w)/2
                    img = img[int(crop_size/2):int(-1*(crop_size/2)), :, :]
                else:
                    crop_size = w - h
                    img = img[:, int(crop_size/2):int(-1*(crop_size/2)),:]

                if self.input_shape:
                    img = cv2.resize(img, self.input_shape, interpolation=cv2.INTER_AREA)

            imgs.append(img)

        return imgs
        


    def get_label(self):
        return list(self.info_df['label'].unique())



    def __len__(self):
        return math.ceil(len(self.info_sample['label']) / self.batch_size)

        
'''
class DataLoader_keypoint(DataLoader):
    def __init__(self, base_path, input_shape = False, batch_size = 256, shuffle = True, 
                        rate_downsampling = 1, mode = 'Training'):
        super().__init__(base_path, input_shape, batch_size, shuffle, 
                        rate_downsampling, mode)
        self.keypoint_path = os.path.join(self.base_path, mode, 'keypoint')
        
    def get_keypoint(self, videos, frames):
        self.origin_path = os.path.join('/data', 'KSL')
        
        for video, frame in zip(videos, frames):
            
            if self.mode == 'Validation':
                list_ = glob(os.path.join(self.origin_path, self.mode, 'keypoint', '*', video.split('.')[0], '{}_*0{}_keypoints.json'.format(video.split('.')[0], frame)))
                if len(list_) < 1:
                    list_ = glob(os.path.join(self.origin_path, self.mode, 'SEN', 'keypoint', '*', video.split('.')[0], '{}_*0{}_keypoints.json'.format(video.split('.')[0], frame)))
                    if len(list_) < 1:
                        list_ = glob(os.path.join(self.origin_path, self.mode, 'WORD', 'keypoint', '*', video.split('.')[0], '{}_*0{}_keypoints.json'.format(video.split('.')[0], frame)))
                        
            else:
                list_ = glob(os.path.join(self.origin_path, self.mode, '*', video.split('.')[0], '{}_*0{}_keypoints.json'.format(video.split('.')[0], frame)))
                if len(list_) < 1:
                    list_ = glob(os.path.join(self.origin_path, self.mode, '*_crowd_keypoint', '*', video.split('.')[0], '{}_*0{}_keypoints.json'.format(video.split('.')[0], frame)))
                    if len(list_) < 1:
                        list_ = glob(os.path.join(self.origin_path, self.mode, 'WORD', 'keypoint', '*', video.split('.')[0], '{}_*0{}_keypoints.json'.format(video.split('.')[0], frame)))
                        if len(list_) < 1:
                            list_ = glob(os.path.join(self.origin_path, self.mode, 'SEN', 'keypoint', '*', video.split('.')[0], '{}_*0{}_keypoints.json'.format(video.split('.')[0], frame)))
            
            if len(list_) > 1:
                print('more than 1', video, frame)
            
            if not len(list_) == 1:
                print(list_)
                print(video, frame)
                continue
                # list_ = glob(os.path.join(self.origin_path, self.mode, '**', video.split('.')[0], '{}_*0{}_keypoints.json'.format(video.split('.')[0], frame)), recursive=True)
                
                # if not len(list_) == 1:
                    # print('finally fail', video, frame)
                    # continue
            
            from_path = list_[0]
            file_name = os.path.basename(from_path)
            to_path = os.path.join(self.base_path, self.mode, 'keypoint', video.split('.')[0])
            
            if not os.path.isdir(to_path):
                os.makedirs(to_path)
            
            shutil.copy(from_path, to_path)
            # print(os.listdir(list_[0]))
        # glob(os.path.join(
        
        
    def __getitem__(self, idx):
        indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        
        
        video_names = [self.info_sample['video_name'][i] for i in indices]
        dur_frames = [self.info_sample['duration_frame'][i] for i in indices]
        local_frames = [self.info_sample['local_frame'][i] for i in indices]
        global_frames = [self.info_sample['global_frame'][i] for i in indices]
        labels = [self.info_sample['label'][i] for i in indices]
        
        self.get_keypoint(video_names, global_frames)

        
        

        # get onehot y
        # labels = [self.info_sample['label'][i] for i in indices]
        # labels_int = [self.target_label.index(label) for label in labels]
'''


if __name__ == '__main__':
    from config import args

    base_path = os.path.join(args.data_path, args.data_dir)
    input_shape = (224, 224)

    dataLoader = DataLoader_keypoint(base_path,
            # mode = 'Validation',
            batch_size = 1,
            input_shape = input_shape,
            rate_downsampling=1)
    for i in tqdm(range(len(dataLoader))):
        dataLoader[i]
    # df = dataLoader.info_df

    # for x, y in dataLoader:
        # print(x.shape, y.shape)
    # dataLoader.get_sample_video(dataLoader.target_label[0], resize=True)





