import json
import os
from glob import glob
from tensorflow.keras.models import model_from_json

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
    path = os.path.join(os.getcwd(), 'train_log')
    
    model, best_epoch, best_metric = getTrainedModel(path)
    
    model.summary()