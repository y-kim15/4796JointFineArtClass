from keras.models import Sequential
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications.resnet50 import ResNet50
from keras.initializers import glorot_uniform
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, InputLayer, Input, GlobalAveragePooling2D, UpSampling2D
from keras.optimizers import Adam, RMSprop, Adadelta, SGD
from contextlib import redirect_stdout
import datetime
import os
from os.path import join
import math


dirname = os.path.dirname(__file__)
MODEL_DIR = "models/logs"
N_CLASSES = 25
# Code influenced by keras application examples
# that from rasta and from ...

def get_autoencoder1(input_shape, pretrained=False):
    input = Input(shape=input_shape)

    x = Conv2D(16, 3, 3, activation='relu', border_mode='same')(input)
    x = MaxPooling2D((2, 2), border_mode='same')(x)
    x = Conv2D(8, 3, 3, activation='relu', border_mode='same')(x)
    x = MaxPooling2D((2, 2), border_mode='same')(x)
    x = Conv2D(8, 3, 3, activation='relu', border_mode='same')(x)
    encoded = MaxPooling2D((2, 2), border_mode='same')(x)

    # at this point the representation is (8, 4, 4) i.e. 128-dimensional

    x = Conv2D(8, 3, 3, activation='relu', border_mode='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, 3, 3, activation='relu', border_mode='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, 3, 3, activation='relu', border_mode='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, 3, 3, activation='sigmoid', border_mode='same')(x)

    return input, decoded


def get_alexnet(input_shape):
        inputs = Input(shape=(227, 227, 3))

        conv_1 = Conv2D(96, (11, 11), strides=(4, 4),  activation='relu', name='conv_1',
                        kernel_initializer='he_normal', input_shape=input_shape)
        conv_2 = MaxPooling2D((3, 3), strides=(2, 2))

        x = Sequential()
        x.add(conv_1)
        x.add(conv_2)
        x.add(Flatten())
        x.add(Dense(256, activation='relu'))
        x.add(Dropout(0.5))
        x.add(Dense(256, activation='relu'))
        x.add(Dropout(0.5))
        x.add(Dense(N_CLASSES, activation='softmax'))

        alexnet = x
        alexnet.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print(alexnet.summary())


def get_test1(input_shape, pretrained):
    # architecture follows general CNN design

    base_model = Sequential()
    base_model.add(InputLayer(input_shape))
    base_model.add(Conv2D(64, (7, 7), strides=(4, 4), activation='relu', padding='valid', kernel_initializer=glorot_uniform(0)))
    base_model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_last'))
    base_model.add(Conv2D(128, (5, 5), strides=(2, 2), activation='relu', kernel_initializer=glorot_uniform(0)))
    base_model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_last'))
    base_model.add(Conv2D(256, (3, 3), strides=(2, 2), activation='relu', kernel_initializer=glorot_uniform(0)))
    base_model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_last'))

    #out = Sequential()
    base_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    base_model.add(Dropout(0.5))
    base_model.add(Dense(56, activation='relu', kernel_initializer=glorot_uniform(1)))
    base_model.add(Dense(N_CLASSES, activation='softmax', kernel_initializer=glorot_uniform(1)))

    return base_model, base_model.output


def get_test2(input_shape, pretrained):
    # fixed seed, use of l2 regularisation, dropout rate of 0.2, initial Conv layer with kernel size 5x5 followed by
    # 2 layers of 3x3 each
    base_model = Sequential()
    base_model.add(InputLayer(input_shape))
    base_model.add(Conv2D(64, (5, 5), activation='relu', padding='valid', kernel_initializer=glorot_uniform(0),
                          kernel_regularizer=l2(0.01)))
    base_model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_last'))
    base_model.add(Conv2D(128, (3, 3), strides=(2, 2), activation='relu', kernel_initializer=glorot_uniform(0),
                          kernel_regularizer=l2(0.01)))
    base_model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_last'))
    base_model.add(Conv2D(256, (3, 3), strides=(2, 2), activation='relu', kernel_initializer=glorot_uniform(0),
                          kernel_regularizer=l2(0.01)))
    base_model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_last'))

    # out = Sequential()
    base_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    base_model.add(Dropout(0.2))
    base_model.add(Dense(28, activation='relu', kernel_initializer=glorot_uniform(0)))
    base_model.add(Dense(N_CLASSES, activation='softmax', kernel_initializer=glorot_uniform(0)))

    return base_model, base_model.output



def get_vgg16(input_shape, pretrained=True):
    if not pretrained:
        base_model = VGG16(include_top=False, weights=None, input_shape=input_shape)
        # return base_model
    else:
        base_model = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)

    x = Sequential()
    x.add(Flatten(input_shape=base_model.output_shape[1:]))
    #x.add(Dense(256, activation='relu'))
    # x.add(Dropout(0.5))
    #x.add(Dense(512, activation='relu'))
    # x.add(Dropout(0.5))
    x.add(Dense(N_CLASSES, activation='softmax'))
    return base_model, x


def get_inceptionv3(input_shape, pretrained=True):
    if not pretrained:
        base_model = InceptionV3(include_top=True, weights=None, input_shape=input_shape)

    else:
        base_model = InceptionV3(include_top=False, weights='imagenet', input_shape=input_shape)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(512, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    output = Dense(N_CLASSES, activation='softmax')(x)

    return base_model, output


def get_resnet50(input_shape, pretrained=True):
    if not pretrained:
        base_model = ResNet50(input_shape=input_shape, weights=None, include_top=True)
    else:
        base_model = ResNet50(input_shape=input_shape, weights='imagenet', include_top=False)

    x = base_model.output
    x = GlobalAveragePooling2D(data_format='channels_last')(x)
    output = Dense(N_CLASSES, activation='softmax')(x)
    return base_model, output


get_model = {
    "inceptionv3": get_inceptionv3,
    "vgg16": get_vgg16,
    "resnet50": get_resnet50,
    "test1": get_test1,
    'test2': get_test2,
    "auto1": get_autoencoder1
}


def get_model_name(sample_no, empty=True, model_type='test1', n_tune=0, **kwargs):
    if empty:
        now = datetime.datetime.now()
        name = model_type + '_' + str(now.month) + '-' + str(now.day) + '-' + str(now.hour) + '-' + str(now.minute)
        name = name + "_empty"

    else:
        name = kwargs["name"].rsplit("_", 1)[0]  # just get model_type_time form
    if n_tune == 0:
        tune = 'full1'
    else:
        tune = str(n_tune)
    name = name + '_tune-' + tune
    name = name + "-no-" + str(sample_no)
    return name

def save_summary(dir_path, name, model):
    file_name = name + '_' + "summary.txt"
    file_path = join(dir_path, file_name)
    with open(file_path, 'w+') as f:  # os.path.join(".", "models", file_name)
        with redirect_stdout(f):
            model.summary()


def vgg_preprocess_input(x):
    # 'RGB'->'BGR'
    x = x[:, :, ::-1]

    x[:, :, 0] -= 133.104
    x[:, :, 0] -= 119.973
    x[:, :, 0] -= 104.432

    # x = preprocess_input(x)

    return x


def wp_preprocess_input(x):

    x[:, :, 0] -= 133.104
    x[:, :, 0] -= 119.973
    x[:, :, 0] -= 104.432

    return x


def get_generator(path, batch_size, target_size, horizontal_flip, pre_type, train_type):
    if pre_type:
        datagen = ImageDataGenerator(horizontal_flip=horizontal_flip, preprocessing_function=vgg_preprocess_input)
    else:
        datagen = ImageDataGenerator(horizontal_flip=horizontal_flip, preprocessing_function=wp_preprocess_input)

    if train_type:
        generator = datagen.flow_from_directory(
            path,
            target_size=target_size,
            batch_size=batch_size,
            class_mode='categorical')
    else:
        generator = datagen.flow_from_directory(
            path,
            target_size=target_size,
            batch_size=batch_size,
            class_mode=None)

    return generator


# learning rate schedule
def step_decay(epoch):
    initial_lrate = 0.1
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate


optimiser = {
    'adam': Adam,
    'rmsprop': RMSprop,
    'adadelta': Adadelta,
    'sgd': SGD
}


def get_optimiser(opt, lr, decay, mom):
    if decay.isdigit():
        if opt == 'sgd':
            return optimiser[opt](lr=lr, decay=float(decay), momentum=mom, nesterov=True)
        else:
            return optimiser[opt](lr=lr, decay=float(decay))
    else:
        if opt != 'sgd':
            return optimiser[opt](lr=lr)
        else:
            return optimiser[opt](lr=lr, momentum=mom)










