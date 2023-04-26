from utils import DataLoader, gpu_limit
from model import get_vgg16
from config import args
import os
from tensorflow import GradientTape
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint

def compile_model(model):

    model.compile(optimizer="adam",
            loss='categorical_crossentropy',
            metrics=['accuracy'],
            run_eagerly=True,
            )

    return model
        

def train(model):
    base_path = os.path.join(args.data_path, args.data_dir)

    train_dataloader = DataLoader(base_path,
                            mode='Training',
                            batch_size = args.batch_size,
                            shuffle = args.shuffle,
                            input_shape = (args.input_h, args.input_w),
                            )
    # train_dataloader.save_image()

    print("[I] Train dataloader is loaded.")
    
    '''
    for train_x, train_y in train_dataloader:
        print(train_x.shape, train_y.shape)
        print(train_x)
        break
    '''
    val_dataloader = DataLoader(base_path,
                            mode='Validation',
                            batch_size = args.batch_size,
                            shuffle = args.shuffle,
                            input_shape = (args.input_h, args.input_w),
                            )
    # val_dataloader.save_image()
    
    print("[I] Validation dataloader is loaded.")
    
    model = compile_model(model) 
    print("[I] Model compile is completed.")

    early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            )

    tensorboard = TensorBoard(
            log_dir='logs',
            )
    checkpoint = ModelCheckpoint(
            filepath='/tmp/checkpoint',
            # save_best_only=True,
            )
    
    print("[I] Training is starting...")
    model.fit(train_dataloader, validation_data=val_dataloader,
            epochs=100,
            callbacks=[early_stopping, tensorboard, checkpoint]
            )





if __name__ == '__main__':
    gpu_limit(5)
    model = get_vgg16(
            input_shape = (args.input_h, args.input_w, 3),
            num_class = 100,
            last_activation='softmax',
            )

    train(model) 

