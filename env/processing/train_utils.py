from keras.models import Sequential, Model
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications.resnet50 import ResNet50
from keras.initializers import glorot_uniform, VarianceScaling
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
from keras.constraints import MaxNorm
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, InputLayer, Input, \
    GlobalAveragePooling2D, UpSampling2D, BatchNormalization, Activation, ZeroPadding2D
from keras.optimizers import Adam, RMSprop, Adadelta, SGD
from contextlib import redirect_stdout
import datetime
import os
from os.path import join
import math
from keras.preprocessing.image import load_img, img_to_array
import re

dirname = os.path.dirname(__file__)
MODEL_DIR = "models/logs"
N_CLASSES = 25
# Code influenced by keras application examples
# that from rasta and from ...

def get_autoencoder2(input_shape, add_reg, alpha, dropout, pretrained=False):
    input = Input(shape=input_shape)
    x = Conv2D(64, (3, 3), strides=(1, 1), activation='')

# resnet
def get_autoencoder1(input_shape,  add_reg, alpha, dropout, pretrained=False):
    input = Input(shape=input_shape)

    x = Conv2D(64, (7, 7), strides=(2, 2), activation='relu', data_format='channels_last',
               kernel_initializer=VarianceScaling(scale=2.0))(input)
    x = BatchNormalization(axis=3)
    x = ZeroPadding2D()
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = Conv2D(64, (1, 1), strides=(1, 1), activation='relu', kernel_initializer=VarianceScaling(scale=2.0))(x)
    x = BatchNormalization(axis=3)
    x = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same', kernel_initializer=VarianceScaling(scale=2.0))(x)
    x = BatchNormalization(axis=3)
    x = Conv2D(256, (1, 1), strides=(1, 1), activation='relu', kernel_initializer=VarianceScaling(scale=2.0))(x)
    encoded = MaxPooling2D((2, 2))(x)

    # at this point the representation is (8, 4, 4) i.e. 128-dimensional

    x = Conv2D(256, (1, 1), strides=(1, 1), activation='relu', kernel_initializer=VarianceScaling(scale=2.0))(encoded)
    x = UpSampling2D((2, 2))(x)
    x = BatchNormalization(axis=3)
    x = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same', kernel_initializer=VarianceScaling(scale=2.0))(x)
    x = BatchNormalization(axis=3)
    x = Conv2D(64, (1, 1), strides=(1, 1), activation='relu', kernel_initializer=VarianceScaling(scale=2.0))(x)
    x = UpSampling2D((3, 3))(x)
    decoded = Conv2D(3, 3, 3, activation='sigmoid', border_mode='same')(x)

    return input, decoded


def get_test1(input_shape, add_reg, alpha, dropout=0.0, pretrained=False):
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
    if dropout > 0:
        base_model.add(Dropout(dropout))
    base_model.add(Dense(56, activation='relu', kernel_initializer=glorot_uniform(0)))
    base_model.add(Dense(N_CLASSES, activation='softmax', kernel_initializer=glorot_uniform(0)))

    return base_model, base_model.output


def get_test2(input_shape, add_reg=l2, alpha=0.01, dropout=0.2, pretrained=False):
    # fixed seed, use of l2 regularisation, dropout rate of 0.2, initial Conv layer with kernel size 5x5 followed by
    # 2 layers of 3x3 each
    base_model = Sequential()
    base_model.add(InputLayer(input_shape))
    base_model.add(Conv2D(64, (5, 5), activation='relu', padding='valid', kernel_initializer=glorot_uniform(0),
                          kernel_regularizer=add_reg(alpha)))
    base_model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_last'))
    base_model.add(Conv2D(128, (3, 3), strides=(2, 2), activation='relu', kernel_initializer=glorot_uniform(0),
                          kernel_regularizer=add_reg(alpha)))
    base_model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_last'))
    base_model.add(Conv2D(256, (3, 3), strides=(2, 2), activation='relu', kernel_initializer=glorot_uniform(0),
                          kernel_regularizer=add_reg(alpha)))
    base_model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_last'))

    # out = Sequential()
    base_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    if dropout > 0:
        base_model.add(Dropout(dropout))
    base_model.add(Dense(28, activation='relu', kernel_initializer=glorot_uniform(0)))
    base_model.add(Dense(N_CLASSES, activation='softmax', kernel_initializer=glorot_uniform(0)))

    return base_model, base_model.output

def get_test3(input_shape, add_reg=l2, alpha=0.01, dropout=0.2, pretrained=False):
    # fixed seed, use of l2 regularisation, dropout rate of 0.2, initial Conv layer with kernel size 5x5 followed by
    # 2 layers of 3x3 each
    base_model = Sequential()
    base_model.add(InputLayer(input_shape))
    base_model.add(Conv2D(64, (5, 5), activation='relu', padding='valid', kernel_initializer=glorot_uniform(0),
                          kernel_regularizer=add_reg(alpha)))
    base_model.add(Conv2D(64, (3, 3), strides=(2, 2), activation='relu', padding='valid', kernel_initializer=glorot_uniform(0)))
    base_model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_last'))
    base_model.add(Conv2D(128, (3, 3), strides=(2, 2), activation='relu', padding='valid', kernel_initializer=glorot_uniform(0)))
    base_model.add(Conv2D(128, (3, 3), strides=(2, 2), activation='relu', padding='valid',  kernel_initializer=glorot_uniform(0)))
    base_model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_last'))
    base_model.add(Conv2D(128, (3, 3), strides=(2, 2), activation='relu', padding='valid', kernel_initializer=glorot_uniform(0)))
    base_model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), data_format='channels_last'))

    # out = Sequential()
    base_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    base_model.add(Dense(56, activation='relu', kernel_initializer=glorot_uniform(0), kernel_constraint=MaxNorm(max_value=4)))
    if dropout > 0:
        base_model.add(Dropout(dropout))
    base_model.add(Dense(N_CLASSES, activation='softmax', kernel_initializer=glorot_uniform(0)))

    return base_model, base_model.output


def get_vgg16(input_shape, add_reg, alpha, dropout, pretrained=True):
    if not pretrained:
        base_model = VGG16(include_top=False, weights=None, input_shape=input_shape)
        # return base_model
    else:
        base_model = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)

    x = Sequential()
    x.add(Flatten(input_shape=base_model.output_shape[1:]))
    x.add(Dense(56, activation='relu'))
    if dropout > 0:
        x.add(Dropout(dropout))
    #x.add(Dense(512, activation='relu'))
    #x.add(Dropout(0.5))
    x.add(Dense(N_CLASSES, activation='softmax'))
    return base_model, x


def get_inceptionv3(input_shape, add_reg, alpha, dropout, pretrained=True):
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


def get_resnet50(input_shape, add_reg, alpha, dropout=0.2, pretrained=True):
    if not pretrained:
        base_model = ResNet50(input_shape=input_shape, weights=None, include_top=True)
    else:
        base_model = ResNet50(input_shape=input_shape, weights='imagenet', include_top=False)

    x = base_model.output
    x = GlobalAveragePooling2D(data_format='channels_last')(x)
    # added below
    x = Dense(128, activation='relu')(x) # , kernel_regularizer=add_reg(alpha))(x)
    if dropout > 0:
        x = Dropout(dropout)(x)
    # up to above
    output = Dense(N_CLASSES, activation='softmax')(x)
    return base_model, output


get_model = {
    "inceptionv3": get_inceptionv3,
    "vgg16": get_vgg16,
    "resnet50": get_resnet50,
    "test1": get_test1,
    'test2': get_test2,
    'test3': get_test3,
    "auto1": get_autoencoder1
}

def get_new_model(model_type, input_shape, reg, alpha, drop, pretrained):
    data_type = True
    base_model, output = get_model[model_type](input_shape, reg, alpha, drop, pretrained)
    if re.search('test*', model_type):
        model = Model(inputs=base_model.input, outputs=base_model.output)
    elif model_type == 'auto1':
        data_type = False
        model = Model(base_model, output)
    elif model_type == 'vgg16':
        model = Model(inputs=base_model.input, outputs=output(base_model.output))
    else:
        model = Model(inputs=base_model.input, outputs=output)
    return model, data_type

def copy_weights(old_model, new_model, layer_no):
    i = 0
    j = layer_no
    if layer_no > 0:
        j = len(old_model.layers) - layer_no
    for layer in old_model.layers:
        if i == j:
            break
        new_model.layers[i].set_weights(old_model.layers[i].get_weights())
        i += 1
    return new_model

def get_model_name(sample_no, type='empty', model_type='test1', n_tune=0, **kwargs):
    if type == 'empty':
        now = datetime.datetime.now()
        name = model_type + '_' + str(now.month) + '-' + str(now.day) + '-' + str(now.hour) + '-' + str(now.minute)
        name = name + "_empty"
    else:
        name = kwargs["name"].rsplit("_", 1)[0] + '_' + type  # just get model_type_time form
    if n_tune == 0:
        tune = 'full1'
    else:
        tune = str(n_tune)
    name = name + '_layers-' + tune
    name = name + "-s-" + str(sample_no)
    return name

def save_summary(dir_path, name, model):
    file_name = name + '_' + "summary.txt"
    file_path = join(dir_path, file_name)
    with open(file_path, 'w+') as f:  # os.path.join(".", "models", file_name)
        with redirect_stdout(f):
            model.summary()





def get_generator(path, batch_size, target_size, horizontal_flip, train_type, function):
    if function == 'vgg16' or function == 'resnet50':
        datagen = ImageDataGenerator(horizontal_flip=horizontal_flip, preprocessing_function=imagenet_preprocess_input)
    elif function == 'auto1':
        datagen = ImageDataGenerator(horizontal_flip=horizontal_flip, preprocessing_function=scale_id_preprocess_input)
    else:
        datagen = ImageDataGenerator(horizontal_flip=horizontal_flip, preprocessing_function=scale_preprocess_input)

    if train_type:
        generator = datagen.flow_from_directory(
            path,
            target_size=target_size,
            batch_size=batch_size,
            seed=0,
            class_mode='categorical')
    else:
        generator = datagen.flow_from_directory(
            path,
            target_size=target_size,
            batch_size=batch_size,
            seed=0,
            class_mode=None)

    return generator


# learning rate schedule
def step_decay(epoch):
    initial_lrate = 0.1
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate


# learning rate schedule
def exp_decay(epoch):
   initial_lrate = 0.01
   k = 0.1
   lrate = initial_lrate * math.exp(-k * epoch)
   return lrate


optimiser = {
    'adam': Adam,
    'rmsprop': RMSprop,
    'adadelta': Adadelta,
    'sgd': SGD
}


def get_optimiser(opt, lr, decay, mom, n_epoch):
    if decay.isdigit():
        if opt == 'sgd':
            return optimiser[opt](lr=lr, decay=float(decay), momentum=mom, nesterov=True)
        else:
            return optimiser[opt](lr=lr, decay=float(decay))
    else:
        if opt != 'sgd':
            if decay == 'rate':
                return optimiser[opt](lr=lr, decay=lr/n_epoch)
            else:
                return optimiser[opt](lr=lr)
        else:
            if decay == 'rate':
                return optimiser[opt](lr=lr, momentum=mom, decay=lr/n_epoch)
            else:
                return optimiser[opt](lr=lr, momentum=mom)


def imagenet_preprocess_input(x):
    # 'RGB'->'BGR'
    x = x[:, :, ::-1]
    # Zero-center by mean pixel
    x[:, :, 0] -= 103.939
    x[:, :, 1] -= 116.779
    x[:, :, 2] -= 123.68
    return x


def scale_preprocess_input(x):
    x[:, :, 0] -= 133.104
    x[:, :, 0] -= 119.973
    x[:, :, 0] -= 104.432
    x *= 1./255
    return x

# zero-center by mean pixel calculated for id_medium_train
def scale_id_preprocess_input(x):
    x[:, :, 0] -= 115.247
    x[:, :, 1] -= 104.962
    x[:, :, 2] -= 91.913
    x *= 1./255
    return x


def wp_preprocess_input(x):
    # 'RGB'->'BGR'
    x = x[:, :, ::-1]

    x[:, :, 0] -= 133.104
    x[:, :, 1] -= 119.973
    x[:, :, 2] -= 104.432
    return x



if __name__=='__main__':
    DATA_PATH = 'data/wikipaintings_full/wikipaintings_train/Abstract_Art/ad-reinhardt_collage-1938.jpg'
    x = load_img(DATA_PATH, target_size=(224, 224))
    print("current")
    x = img_to_array(x)
    print(x)
    temp = x
    #print("using imagenet ")
    #x = imagenet_preprocess_input(x)
    #print(x)
    print("using wp")
    temp = wp_preprocess_input(temp)
    print(temp)






