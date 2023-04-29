import json
import os
from glob import glob
from tensorflow.keras.models import model_from_json
from utils import DataLoader_keypoint, gpu_limit
from config import args
import numpy as np
from tqdm import tqdm
import tensorflow as tf

def getTrainedModel(targetPath):
    savedDirs = glob(os.path.join(targetPath, 'best_model_*'))

    if len(savedDirs) == 1:
        savedPath = savedDirs[0]
        epoch = (int(os.path.basename(savedPath).split('_')[-2]))
        metric = (float(os.path.basename(savedPath).split('_')[-1]))
    else:
        raise ValueError("There is no best model or there are more than one best models., {}".format(savedDirs))


    # load architecture
    architec_path = os.path.join(savedPath, 'architecture.json')
    with open(architec_path, 'r') as file:
        jsonConfig = json.load(file)
    model = model_from_json(jsonConfig)

    # load weights

    weight_path = os.path.join(savedPath, 'ckpt')
    load_status = model.load_weights(weight_path)
    load_status.assert_consumed()

    return model, epoch, metric


if __name__ == '__main__':
    gpu_limit(1)

    path = os.path.join(os.getcwd(), 'train_log_100')
    model, best_epoch, best_metric = getTrainedModel(path)
    model.summary()


    dataloader = DataLoader_keypoint("Validation", True, args.batch_size, True)
    # dataloader = DataLoader_keypoint("Validation", True, 1, True)

    confusion_matrix = np.zeros((100, 100))

    for i in range(len(dataloader)):
        x, y = dataloader[i]
        print(x.shape, y.shape)
        print("test: {:>6}/{:>6}".format(i+1, len(dataloader)), end='\r')

        y_pred = model(x)

        for j in tqdm(range(x.shape[0])):
            true_idx = np.argmax(y[j])
            pred_idx = np.argmax(y_pred[j])
            confusion_matrix[true_idx, pred_idx] += 1

        save_dir = os.path.join(path, 'test')
        if not os.path.isdir(save_dir):
            print("make dir")
            print(save_dir)
            os.makedirs(save_dir)
        np.save(os.path.join(save_dir, 'confusion_matrix.npy'), confusion_matrix)
