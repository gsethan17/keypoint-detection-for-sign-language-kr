from utils import DataLoader
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
    
    val_dataloader = DataLoader(base_path,
                            mode='Validation',
                            batch_size = args.batch_size,
                            shuffle = args.shuffle,
                            input_shape = (args.input_h, args.input_w),
                            )
    model = compile_model(model) 

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
    
    model.fit(train_dataloader, validation_data=val_dataloader,
            epochs=100,
            callbacks=[early_stopping, tensorboard, checkpoint]
            )

    '''
    loss_f = CategoricalCrossentropy()
    metric_f = CategoricalAccuracy()
    optimizer = Adam()


    for epoch in range(100):
        for x, y in train_dataloader:
            print(x.shape, y.shape)
            with GradientTape() as tape:
                pred_y = model(x)
                loss = loss_f(y, pred_y)

            gradient = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradient, model.trainable_variables))

            print(pred_y.shape)

        for x_val, y_val in val_dataloader:

            pred_y_val = model(x_val)

            print(x_val.shape, y_val.shape, pred_y_val.shape)
    '''




if __name__ == '__main__':
    model = get_vgg16(
            input_shape = (args.input_h, args.input_w, 3),
            num_class = 100,
            last_activation='softmax',
            )
    model = compile_model(model) 

    train(model) 

