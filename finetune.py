import shutil
import tensorflow as tf
import keras
from keras_tqdm import TQDMCallback
from keras.models import Sequential, Model, load_model
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, InputLayer, Input, GlobalAveragePooling2D
from keras.optimizers import Adam
from rasta.python.models.processing import count_files
from contextlib import redirect_stdout
import datetime
import os
from os.path import join
from cleaning.read_images import create_dir

dirname = os.path.dirname(__file__)
MODEL_DIR = "models/logs"
N_CLASSES = 25
# Code influenced by keras application examples
# that from rasta and from ...


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

    base_model = Sequential()
    base_model.add(InputLayer(input_shape))
    base_model.add(Conv2D(64, (7, 7), strides=(4, 4), activation='relu', padding='valid'))
    base_model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_last'))
    base_model.add(Conv2D(128, (5, 5), strides=(2, 2), activation='relu'))
    base_model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_last'))
    base_model.add(Conv2D(256, (3, 3), strides=(2, 2), activation='relu'))
    base_model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_last'))

    #out = Sequential()
    base_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    base_model.add(Dropout(0.5))
    base_model.add(Dense(56, activation='relu'))
    base_model.add(Dense(N_CLASSES, activation='softmax'))

    return base_model


def get_vgg16(input_shape, pretrained=True):
    if not pretrained:
        base_model = VGG16(include_top=False, weights=None, input_shape=input_shape)
        # return base_model
    else:
        base_model = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)

    x = Sequential()
    x.add(Flatten(input_shape=base_model.output_shape[1:]))
    x.add(Dense(512, activation='relu'))
    # x.add(Dropout(0.5))
    x.add(Dense(512, activation='relu'))
    # x.add(Dropout(0.5))
    x.add(Dense(N_CLASSES, activation='softmax'))
    return base_model, x

    # model = Model(input=base_model.input, output=x(base_model.output))

    # x = base_model.output
    # x = Dense(4096, activation='relu')(x)
    # x = Dropout(0.5)(x)
    # x = Dense(4096, activation='relu')(x)
    # x = Dropout(0.5)(x)
    # output = Dense(n_classes, activation='softmax')(x)
    # model.outputs = [model.layers[-1].output]
    # model.layers[-1].outbound_nodes = []
    # model.add(Dense(num_class, activation='softmax'))

    # return base_model, x


def get_inceptionv3(input_shape, pretrained=True):
    if not pretrained:
        base_model = InceptionV3(include_top=True, weights=None, input_shape=input_shape)

    else:
        base_model = InceptionV3(include_top=False, weights='imagenet', input_shape=input_shape)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    output = Dense(N_CLASSES, activation='softmax')(x)

    return base_model, output


get_model = {
    "inceptionv3": get_inceptionv3,
    "vgg16": get_vgg16,
    "test1": get_test1
}


def get_model_name(model_type, sample_no, empty=True, **kwargs):
    now = datetime.datetime.now()
    name = model_type + '_' + str(now.month) + '-' + str(now.day) + '-' + str(now.hour) + '-' + str(now.minute)
    if empty:
        name = name + "_empty"
    else:
        name = kwargs["name"].rsplit("_", 1)[0]  # just get model_type_time form
        n_tune = kwargs["n_tune"]
        name = name + '_tune-' + str(n_tune)
    name = name + "-no-" + str(sample_no)
    return name

    # train_type: T for empty, F for retrain existing
def fit_model(model_type, input_shape, epochs, train_path, val_path, batch_size, sample_no, train_type, horizontal_flip=True, save=True, **kwargs):
    if model_type == "vgg":
        pre_type = True
    else:
        pre_type = False
    if not train_type:
        path = kwargs["dir_path"]
        name = kwargs["name"]
        model = load_model(path)
        n_tune = kwargs["n_tune"]
        name = get_model_name(model_type, sample_no, empty=False, name=name, n_tune=n_tune)
        for layer in model.layers[:len(model.layers) - n_tune]:
            layer.trainable = False
        for layer in model.layers[len(model.layers) - n_tune:]:
            layer.trainable = True
        dir_path = path
    else:
        base_model = get_model[model_type](input_shape, pretrained=False)
        model = Model(inputs=base_model.input, outputs=base_model.output)
        name = get_model_name(model_type, sample_no)
        dir_path = join("models", name)
        create_dir(dir_path)
    train_generator = get_generator(train_path, batch_size, target_size=(input_shape[0], input_shape[1]),
                                    horizontal_flip=horizontal_flip, pre_type=pre_type)
    model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    if save:
        save_summary(dir_path, name, model)
    val_generator = get_generator(val_path, batch_size, target_size=(input_shape[0], input_shape[1]),
                                  horizontal_flip=horizontal_flip, pre_type=pre_type)
    tb = TensorBoard(log_dir=MODEL_DIR + "/graph", histogram_freq=0, write_graph=True, write_images=True)
    earlyStop = EarlyStopping(monitor='val_acc', patience=5)
    if val_path != None:

        checkpoint = ModelCheckpoint(join("models", name, "{epoch:02d}-{val_acc:.3f}.hdf5"), monitor='val_acc',
                                     verbose=1, save_best_only=True, mode='max')
        history = model.fit_generator(train_generator, steps_per_epoch=count_files(train_path) // batch_size,
                                      epochs=epochs, callbacks=[tb, checkpoint, earlyStop], validation_data=val_generator,
                                      validation_steps=count_files(val_path) // batch_size)
    else:
        history = model.fit_generator(train_generator, steps_per_epoch=count_files(train_path) // batch_size,
                                      epochs=epochs)

    model.save(join(dir_path, name + ".h5py"))
    return name, dir_path


def save_summary(dir_path, name, model):
    file_name = name + '_' + "summary.txt"
    file_path = join(dir_path, file_name)
    with open(file_path, 'w+') as f:  # os.path.join(".", "models", file_name)
        with redirect_stdout(f):
            model.summary()


def vgg_preprocess_input(x):
    # 'RGB'->'BGR'
    x = x[:, :, ::-1]

    x[:,:,0] -=  133.104
    x[:,:,0] -=  119.973
    x[:,:,0] -=  104.432

    return x


def wp_preprocess_input(x):

    x[:, :, 0] -= 133.104
    x[:, :, 0] -= 119.973
    x[:, :, 0] -= 104.432

    return x


def get_generator(path, batch_size, target_size, horizontal_flip, pre_type):
    if pre_type:
        datagen = ImageDataGenerator(horizontal_flip=horizontal_flip, preprocessing_function=vgg_preprocess_input)
    else:
        datagen = ImageDataGenerator(horizontal_flip=horizontal_flip, preprocessing_function=wp_preprocess_input)

    generator = datagen.flow_from_directory(
        path,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical')
    return generator


