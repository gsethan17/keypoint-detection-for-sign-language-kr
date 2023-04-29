from utils import DataLoader_keypoint, gpu_limit
from config import args
import tensorflow as tf
import os
import json 


if __name__ == "__main__":
    gpu_limit(args.gpu_limit)
    
    dataloader = DataLoader_keypoint("Training", True, args.batch_size, True)
    print(len(dataloader))


    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Masking(mask_value=9.,
                                      input_shape=(300, 48*2)))
    model.add(tf.keras.layers.LSTM(100, return_sequences=True, activation='relu'))
    model.add(tf.keras.layers.LSTM(100, activation='softmax'))

    model.summary()

    optimizer = tf.keras.optimizers.Adam(0.01)

    loss_f = tf.keras.losses.CategoricalCrossentropy()
    metric_1 = tf.keras.metrics.CategoricalCrossentropy()
    metric_2 = tf.keras.metrics.CategoricalAccuracy()


    log_train_dir = os.path.join(os.getcwd(), 'train_log_10', 'train_epoch')
    writer_e_train = tf.summary.create_file_writer(log_train_dir)
    log_val_dir = os.path.join(os.getcwd(), 'train_log_10', 'val_epoch')
    writer_e_val = tf.summary.create_file_writer(log_val_dir)
    log_train_dir = os.path.join(os.getcwd(), 'train_log_10', 'train_step')
    writer_s_train = tf.summary.create_file_writer(log_train_dir)
    log_val_dir = os.path.join(os.getcwd(), 'train_log_10', 'val_step')
    writer_s_val = tf.summary.create_file_writer(log_val_dir)

    train_cnt = 0
    val_cnt = 0
    val_losses = []
    for e in range(args.epochs):
        for i in range(len(dataloader)):
            x, y = dataloader[i]
            print("Epoch: {:>3} / train: {:>6}/{:>6}".format(e+1, i+1, len(dataloader)), end='\r')

            with tf.GradientTape() as tape:

                y_pred = model(x)
                loss = loss_f(y, y_pred)

            gradient = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradient, model.trainable_variables))
            train_cnt += 1

            metric_1.update_state(y, y_pred)
            metric_2.update_state(y, y_pred)

            with writer_e_train.as_default(step=train_cnt):
                tf.summary.scalar('loss_step', metric_1.result().numpy())
                tf.summary.scalar('metric_step', metric_2.result().numpy())
            
        with writer_e_train.as_default(step=e+1):
            tf.summary.scalar('loss_epoch', metric_1.result().numpy())
            tf.summary.scalar('metric_epoch', metric_2.result().numpy())
        
        print("Epoch: {:>3} / train: loss [{:>6}] / metric [{:>6}]".format(e+1, metric_1.result().numpy(),metric_2.result().numpy()))
        metric_1.reset_state()
        metric_2.reset_state()

        
        for j in range(dataloader.len_val):
            x_val, y_val = dataloader.get_val_item(j)
            print("Epoch: {:>3} / train: {:>6}/{:>6}".format(e+1, j, dataloader.len_val), end='\r')

            y_val_pred = model(x_val)
            val_cnt += 1

            metric_1.update_state(y_val, y_val_pred)
            metric_2.update_state(y_val, y_val_pred)
            with writer_s_val.as_default(step=val_cnt):
                tf.summary.scalar('loss_step', metric_1.result().numpy())
                tf.summary.scalar('metric_step', metric_2.result().numpy())
            
        with writer_e_val.as_default(step=e+1):
            tf.summary.scalar('loss_epoch', metric_1.result().numpy())
            tf.summary.scalar('metric_epoch', metric_2.result().numpy())
            
        val_losses.append(metric_1.result().numpy())
        print("Epoch: {:>3} / val: loss [{:>6}] / metric [{:>6}]".format(e+1, metric_1.result().numpy(),metric_2.result().numpy()))
            

        
        
        if e > 0 :
            min_ = min(val_losses)
            cur_ = val_losses[-1]
            
            if cur_ <= min_:
                patience = 0
                best_weights = model.get_weights()
                best_epoch = e+1
                best_loss = cur_
                best_metric = metric_2.result().numpy()
    
                save_dir = os.path.join(os.getcwd(), 'train_log_deep', 'mid_model_{}_{:.4f}'.format(best_epoch, best_metric))
                if not os.path.isdir(save_dir):
                    os.makedirs(save_dir)

                # save architecture
                json_config = model.to_json()
                save_dir_config = os.path.join(save_dir, 'architecture.json')
                with open(save_dir_config, 'w', encoding='utf-8') as file:
                    json.dump(json_config, file)

                # save weights
                save_dir_weights = os.path.join(save_dir, 'ckpt')
                model.save_weights(save_dir_weights)

            else:
                patience += 1
                if patience >= 10:
                    break
                
                
        metric_1.reset_state()
        metric_2.reset_state()

        dataloader.on_epoch_end()
        
    model.set_weights(best_weights)
    
    save_dir = os.path.join(os.getcwd(), 'train_log_deep', 'best_model_{}_{:.4f}'.format(best_epoch, best_metric))
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # save architecture
    json_config = model.to_json()
    save_dir_config = os.path.join(save_dir, 'architecture.json')
    with open(save_dir_config, 'w', encoding='utf-8') as file:
        json.dump(json_config, file)

    # save weights
    save_dir_weights = os.path.join(save_dir, 'ckpt')
    model.save_weights(save_dir_weights)

