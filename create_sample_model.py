from keras.models import Model
from keras.layers import Convolution2D, BatchNormalization, MaxPooling2D, Activation, Input, UpSampling2D

import os


def conv_bn(x, filters):
    x = Activation('relu')(x)
    x = BatchNormalization(axis=1)(x)
    x = Convolution2D(filters, 3, 3, border_mode='same')(x)
    return x


def create_model():
    input_tensor = Input((1, None, None))
    x = conv_bn(input_tensor, 32)
    x = MaxPooling2D((2, 2))(x)
    x = conv_bn(x, 48)
    x = MaxPooling2D((2, 2))(x)
    x = conv_bn(x, 64)
    x = UpSampling2D((2, 2))(x)
    x = conv_bn(x, 48)
    x = UpSampling2D((2, 2))(x)
    x = conv_bn(x, 32)
    x = Convolution2D(1, 1, 1, activation='sigmoid')(x)
    model = Model(input=input_tensor, output=x)
    with open(os.path.join('input', 'new.json'), 'w') as f:
        f.write(model.to_json())


if __name__ == '__main__':
    create_model()
