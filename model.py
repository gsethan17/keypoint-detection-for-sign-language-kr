from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras import Sequential

def get_vgg16(input_shape, num_class, last_activation):
    vgg = VGG16(
            include_top=False,
            weights='imagenet',
            input_shape=input_shape,
            # classes=num_class,
            # classifier_activation=last_activation,
            )

    flatten = Flatten()
    dense1 = Dense(1000, activation='relu')
    output_ = Dense(num_class, activation=last_activation)

    model = Sequential(
            [vgg, flatten, dense1, output_]
            )

    print(model.summary())


    return model
