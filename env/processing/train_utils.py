from keras.models import Sequential, Model
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications.resnet50 import ResNet50
from keras.initializers import glorot_uniform, VarianceScaling
from keras.regularizers import l2
#from keras.preprocessing.image import ImageDataGenerator
from processing.keras_image_ext import ImageDataGenerator
from keras.constraints import MaxNorm
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, InputLayer, Input, \
    GlobalAveragePooling2D, UpSampling2D, BatchNormalization, Activation, ZeroPadding2D, AveragePooling2D
from keras.optimizers import Adam, RMSprop, Adadelta, SGD
from keras import backend as K
from rasta.custom_resnets import *
from rasta.alexnet import decaf,alexnet
from contextlib import redirect_stdout
import numpy as np
import datetime
import os
from os.path import join
import math
from keras.callbacks import LearningRateScheduler
import re
import sys
import pandas, json

dirname = os.path.dirname(__file__)
MODEL_DIR = "models/logs"
N_CLASSES = 25

def fixed_generator(generator):
    for batch in generator:
        yield (batch[0], batch[0])
        #yield batch
# Code influenced by keras application examples
# that from rasta and from ...

def get_autoencoder2(input_shape, add_reg, alpha, dropout, pretrained=False):
    input = Input(shape=input_shape)
    x = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same')(input)
    x = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    #x = MaxPooling2D((2,2), padding='same')(x)
   # x = Conv2D(128, (3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    #x = Conv2D(128, (3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    #x = Conv2D(128, (3,3), strides=(1,1), activation='relu', padding = 'same')(encoded)
    #x = Conv2D(128, (3,3), strides=(1,1), activation='relu', padding = 'same')(x)
    #x = UpSampling2D((2,2))(x)
    x = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding = 'same')(encoded)#x)
    x = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding = 'same')(x)
    x = UpSampling2D((2,2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    return input, decoded
# resnet
def get_autoencoder1(input_shape,  add_reg, alpha, dropout, pretrained=False):
    input = Input(shape=input_shape)
    x = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same')(input)
    x = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    x = MaxPooling2D((2,2), padding='same')(x)
    x = Conv2D(128, (3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    x = Conv2D(128, (3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    #x = MaxPooling2D((2, 2), padding='same')(x)
    #x = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    #x = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    #x = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    #x = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same')(encoded)
    #x = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    #x = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    #x = UpSampling2D((2,2))(x)
    x = Conv2D(128, (3,3), strides=(1,1), activation='relu', padding = 'same')(encoded)
    x = Conv2D(128, (3,3), strides=(1,1), activation='relu', padding = 'same')(x)
    x = UpSampling2D((2,2))(x)
    x = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same')(x)  # x)
    x = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    return input, decoded

def get_vgg16(input_shape, add_reg, alpha, dropout, pretrained=True):
    if not pretrained:
        base_model = VGG16(include_top=False, weights=None, input_shape=input_shape)
        # return base_model
    else:
        base_model = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)

    x = base_model.output
    x = GlobalAveragePooling2D(data_format='channels_last')(x)
    # added below
    if add_reg is not None and dropout > 0:
        x = Dense(128, activation='relu', kernel_regularizer=add_reg(alpha))(x)
    elif add_reg is None and dropout > 0:
        x = Dense(128, activation='relu')(x)
    if dropout > 0:
        x = Dropout(dropout)(x)
    #else:
    #    x = Dense(128, activation='relu')(x)
    # up to above
    if add_reg is not None:
        output = Dense(N_CLASSES, activation='softmax', kernel_regularizer=add_reg(alpha))(x)
    else:
        output = Dense(N_CLASSES, activation='softmax')(x)
    return base_model, output


def get_inceptionv3(input_shape, add_reg, alpha, dropout, pretrained=True):
    if not pretrained:
        base_model = InceptionV3(include_top=True, weights=None, input_shape=input_shape)

    else:
        base_model = InceptionV3(include_top=False, weights='imagenet', input_shape=input_shape)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    # added below
    if add_reg is not None and dropout > 0:
        x = Dense(128, activation='relu', kernel_regularizer=add_reg(alpha))(x)
    elif add_reg is None and dropout > 0:
        x = Dense(128, activation='relu')(x)
    if dropout > 0:
        x = Dropout(dropout)(x)
    #else:
    #    x = Dense(128, activation='relu')(x)
    # up to above
    if add_reg is not None:
        output = Dense(N_CLASSES, activation='softmax', kernel_regularizer=add_reg(alpha))(x)
    else:
        output = Dense(N_CLASSES, activation='softmax')(x)

    return base_model, output


def get_resnet50(input_shape, add_reg, alpha, dropout=0.2, pretrained=True):
    if not pretrained:
        base_model = ResNet50(input_shape=input_shape, weights=None, include_top=False, pooling='avg')
    else:
        base_model = ResNet50(input_shape=input_shape, weights='imagenet', include_top=False, pooling='avg')

    x = base_model.output
    #x = GlobalAveragePooling2D(data_format='channels_last')(x)
    # added below
    if add_reg is not None and dropout > 0:
        x = Dense(128, activation='relu', kernel_regularizer=add_reg(alpha))(x)
    elif add_reg is None and dropout > 0:
        x = Dense(128, activation='relu')(x)
    if dropout > 0:
        x = Dropout(dropout)(x)
    #else:
    #    x = Dense(128, activation='relu')(x)
    # up to above
    if add_reg is not None:
        output = Dense(N_CLASSES, activation='softmax', kernel_regularizer=add_reg(alpha))(x)
    else:
        output = Dense(N_CLASSES, activation='softmax')(x)
    return base_model, output

get_model = {
    "inceptionv3": get_inceptionv3,
    "vgg16": get_vgg16,
    "resnet50": get_resnet50,
    "auto1": get_autoencoder1,
    'auto2': get_autoencoder2
}

def get_rasta_model(model_name, n_train, reg, alpha, init, drop):
    K.set_image_data_format('channels_last')
    base_model = None
    if model_name =='alexnet_empty':
        model = alexnet(weights=None)
        for layer in model.layers:
            layer.trainable = True
    elif model_name=='custom_resnet':
        model = custom_resnet(add_reg=reg, alpha=alpha, init=init, n=n_train, dp_rate=drop)
    return model

def get_new_model(model_type, input_shape, reg, alpha, init, drop, pretrained, n_train):
    data_type = True
    if int(pretrained)==1:
        tr = True
    else:
        tr = False
    if model_type in get_model:
        base_model, output = get_model[model_type](input_shape, reg, alpha, drop, tr)
        if re.search('auto*', model_type):
            data_type = False
            model = Model(inputs=base_model, outputs=output)
        else:
            model = Model(inputs=base_model.input, outputs=output)
    else: # rasta
        model = get_rasta_model(model_type, n_train, reg, alpha, init, drop)
    if int(pretrained)==2:
        WEIGHTS_PATH_NO_TOP = 'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnet50_imagenet_11586_no_top.h5'
        weights_path = get_file('resnet50_imagenet_11586_no_top.h5',#'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                WEIGHTS_PATH_NO_TOP,
                                cache_subdir='models',
                                md5_hash='a268eb855778b3df3c7506639542a6af')
        model.load_weights(weights_path,by_name=True)
    return model, data_type

def copy_range(old_model, new_model, ran):
    try:
        start = int(ran.split('-')[0])
        end = int(ran.split('-')[1])
        for i in range(start, end + 1):
            new_model.layers[i].set_weights(old_model.layers[i].get_weights())
    except ValueError:
        sys.exit("ValueError: incorrect range of layers specified")

    return new_model

def copy_layer(old_model, new_model, layer_no):
    try :
        if int(layer_no) > 0:
            index = int(layer_no)
            if index >= len(new_model.layers):
                return None
            else:
                new_model.layers[index].set_weights(old_model.layers[index].get_weights())
    except ValueError:
        sys.exit("ValueError: incorrect index of layer specified")
    return new_model

copy_type = {
    'range':copy_range,
    'layer':copy_layer
}

def copy_weights(old_model, new_model, layer_no):
    if '-' in layer_no:
        if ',' in layer_no:
            splits = layer_no.split(',')
            for o in splits:
                if '-' in o:
                    new_model = copy_type['range'](old_model, new_model, o)
                else:
                    new_model = copy_type['layer'](old_model, new_model, o)

        else:
            new_model = copy_type['range'](old_model, new_model, layer_no)
    elif ',' in layer_no:
        splits = layer_no.split(',')
        for o in splits:
            new_model = copy_type['layer'](old_model, new_model, o)
    else:
        new_model = copy_type['layer'](old_model, new_model, layer_no)
    return new_model

def set_trainable_layers(model, layers):
    changed = False
    if '-' not in layers:
        try:
            if ',' not in layers:
                v = int(layers)
                if v > 0 and v < len(model.layers):
                    for layer in model.layers[:len(model.layers) - v + 1]:
                        if layer.trainable == True:
                            changed = True
                        layer.trainable = False
                    for layer in model.layers[len(model.layers) - v + 1:]:
                        if layer.trainable == False:
                            changed = True
                        layer.trainable = True
                elif v == 0:
                    for layer in model.layers:
                        if layer.trainable == False:
                            changed = True
                        layer.trainable = True
            else:
                ls = layers.split(',')
                for n, i in zip(range(len(ls)), ls):
                    ii = int(i)
                    if n == 0:
                        for layer in model.layers[:ii]:
                            if layer.trainable == True:
                                changed = True
                            layer.trainable = False
                    elif n == len(ls)-1:
                        for layer in model.layers[ii+1:]:
                            if layer.trainable == True:
                                changed = True
                            layer.trainable = False
                    else:
                        if model.layers[ii].trainable == False:
                            changed = True
                        model.layers[ii].trainable = True
        except ValueError:
            sys.exit("Error in input of the number of trainable layers")
    else:
        try:
            if ',' in layers:
                multi = layers.split(',')
                if '-' in multi[0]:
                    start = int(multi[0].split('-')[0])
                else:
                    start = multi[0]
                if '-' in multi[len(multi)-1]:
                    end = int(multi[len(multi)-1].split('-')[1])
                else:
                    end = multi[len(multi)-1]
                for layer in model.layers[0:start]:
                    if layer.trainable == True:
                        changed = True
                    layer.trainable = False
                for layer in model.layers[end + 1:]:
                    if layer.trainable == True:
                        changed = True
                    layer.trainable = False
                for i in range(len(multi)):
                    if '-' in multi[i]:
                        start = int(multi[i].split('-')[0])
                        end = int(multi[i].split('-')[1])
                        if start >= len(model.layers) or end >= len(
                                model.layers) or start < 0 or end < 0 or start >= end:
                            raise ValueError
                        for layer in model.layers[start:end + 1]:
                            if layer.trainable == False:
                                changed = True
                            layer.trainable = True
                    else:
                        if model.layers[multi[i]].trainable == False:
                            changed = True
                        model.layers[multi[i]].trainable = True

            else:
                start = int(layers.split('-')[0])
                end = int(layers.split('-')[1])
                if start >= len(model.layers) or end >= len(model.layers) or start < 0 or end < 0 or start >= end:
                    raise ValueError
                for layer in model.layers[start:end + 1]:
                    if layer.trainable == False:
                        changed = True
                    layer.trainable = True
                for layer in model.layers[0:start]:
                    if layer.trainable == True:
                        changed = True
                    layer.trainable = False
                for layer in model.layers[end + 1:]:
                    if layer.trainable == True:
                        changed = True
                    layer.trainable = False
        except ValueError:
            sys.exit("Error in input of the number of trainable layers")
    return model, changed

def get_model_name(sample_no, type='empty', multi=False, model_type='resnet50', n_tune=0, **kwargs):
    if type == 'empty':
        if not multi:
            now = datetime.datetime.now()
            name = model_type + '_' + str(now.month) + '-' + str(now.day) + '-' + str(now.hour) + '-' + str(now.minute)
            name = name + "_empty"
        else:
            pass
    else:
        name = kwargs["name"].rsplit("_", 1)[0] + '_' + type  # just get model_type_time form
    if n_tune == 0:
        tune = 'full'
    else:
        if ',' in str(n_tune):
            tune = str(n_tune).replace(',', '_')
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

def center_crop(x, center_crop_size, **kwargs):
    centerw, centerh = x.shape[1]//2, x.shape[2]//2
    halfw, halfh = center_crop_size[0]//2, center_crop_size[1]//2
    return x[:, centerw-halfw:centerw+halfw,centerh-halfh:centerh+halfh]

def random_crop(x, random_crop_size, sync_seed=None, **kwargs):
    np.random.seed(sync_seed)
    w, h = x.shape[1], x.shape[2]
    rangew = (w - random_crop_size[0]) // 2
    rangeh = (h - random_crop_size[1]) // 2
    offsetw = 0 if rangew == 0 else np.random.randint(rangew)
    offseth = 0 if rangeh == 0 else np.random.randint(rangeh)
    return x[:, offsetw:offsetw+random_crop_size[0], offseth:offseth+random_crop_size[1]]



def get_generator(path, batch_size, target_size, horizontal_flip, train_type, function):
    if function == 'vgg16' or function == 'resnet50':
        datagen = ImageDataGenerator(width_shift_range=.15, height_shift_range=.15, horizontal_flip=horizontal_flip, preprocessing_function=imagenet_preprocess_input)
    elif re.search('auto*', function):
        datagen = ImageDataGenerator(horizontal_flip=horizontal_flip, preprocessing_function=id_preprocess_input)
    else:
        datagen = ImageDataGenerator(horizontal_flip=horizontal_flip, preprocessing_function=scale_preprocess_input)
    datagen.config['random_crop_size'] = (224, 224)
    datagen.config['center_crop_size'] = (224, 224)
    datagen.set_pipeline([random_crop, center_crop])
    dgdx = datagen.flow_from_directory(path, class_mode='categorical', batch_size=batch_size)

    # you can now fit a generator as well
    datagen.fit_generator(dgdx, nb_iter=10)

    # here we sychronize two generator and combine it into one
    generator = dgdx
    '''
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
            class_mode='input')
    print("total size : ", generator.n)'''
    return generator


# learning rate schedule
def lr_decay_callback(lr_init, lr_decay):
    def step_decay(epoch):
        drop = 5
        return lr_init * math.pow(lr_decay, math.floor((1 + epoch) / drop))
        #return lr_init * math.pow(lr_decay, (epoch + 1))
    return LearningRateScheduler(step_decay)


def lr_decay_callback2(lr_init, lr_decay):
    def exp_decay(epoch):
        return lr_init * math.exp(-lr_decay * epoch)
    return LearningRateScheduler(exp_decay)

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
def id_preprocess_input(x):
    # 'RGB'->'BGR'
    x = x[:, :, ::-1]
    x[:, :, 0] -= 115.247
    x[:, :, 1] -= 104.962
    x[:, :, 2] -= 91.913
    #x *= 1./255
    return x


def wp_preprocess_input(x):
    # 'RGB'->'BGR'
    x = x[:, :, ::-1]

    x[:, :, 0] -= 133.104
    x[:, :, 1] -= 119.973
    x[:, :, 2] -= 104.432
    return x

def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z

cmds= {
    "t": "train_type",
    "b": "batch_size",
    "e": "epochs",
    "f": "horizontal_flip",
    "n": "n_layers_trainable",
    "opt": "optimiser",
    "decay": "add_decay",
    "r": "add_reg",
    "alp": "alpha",
    "dropout": "add_drop",
    "mom": "add_mom",
    "w": "add_wei"
}

def save_ordered(csv_path):
    data = pandas.read_csv(csv_path, encoding='utf-8-sig')
    headers = list(data[:0])
    sorted = data.sort_values("max_val_acc", ascending=False)
    sorted.to_csv(join(csv_path.rsplit('/', 1)[0], '_output_ordered.csv'), header=headers, index=True)
    return sorted

def get_best_comb_from_csv(csv_path, sorted, params, top=3, save=False):
    print(sorted)
    best = "Best Combination of "
    for p in params:
        best += p.replace("-", "")
        best += "-"
    best = best.rsplit("-", 1)[0] + ": " + "\n"

    combs = {}
    for i in range(top):
        #r = sorted.iloc(i)
        dic = {}
        for p in params:
            dic[p] = sorted.loc[i, cmds[p]]#.iloc(i)
            if i == 0:
                best += p + ": " + str(dic[p]) + "\n"
        dic["max_val_acc"] = round(sorted.loc[i,"max_val_acc"],4)#.iloc(i)
        combs[str(i+1)] = dic
    best += "Highest Validation Accuracy: " + str(round(combs["1"]["max_val_acc"],4)) + "\n"
    best += ''.join('{}{}\n'.format(key, val) for key, val in combs.items())
    print(best)

    if save:
        with open(join(csv_path.rsplit('/', 1)[0], '_output_top_' + str(top) + '.json'), 'w') as f:
            json.dump(combs, f)
    return best

if __name__=='__main__':
    '''DATA_PATH = 'data/wikipaintings_full/wikipaintings_train/Abstract_Art/ad-reinhardt_collage-1938.jpg'
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
    print(temp)'''

    get_best_comb_from_csv("../models/train_hyp_2-11-23-11/_output.csv", ['decay', 'mom'], save=True)
