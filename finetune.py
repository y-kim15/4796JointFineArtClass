import shutil

from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Flatten, Conv2D, MaxPooling2D, ZeroPadding2D, Dropout, Activation
from rasta.python.utils.utils import wp_preprocess_input
from rasta.python.models.processing import count_files
from contextlib import redirect_stdout
import datetime
import os
from os.path import join

dirname = os.path.dirname(__file__)
MODEL_DIR = "models/logs"

# Code influenced by keras application examples
# that from rasta and from ...
def create_dir(file_path):
    if os.path.exists(file_path):
        shutil.rmtree(file_path)
    os.makedirs(file_path)

def get_alexnet(input_shape, n_classes=25):
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
        x.add(Dense(n_classes, activation='softmax'))

        alexnet = x
        alexnet.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print(alexnet.summary())

def get_test1(input_shape, n_classes, pretrained=True):
    from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, Input

    base_model = Sequential()
    base_model.add(Conv2D(64, (7, 7), strides=(4, 4), input_shape=input_shape, activation='relu', padding='valid'))
    base_model.add(MaxPool2D(pool_size=(2, 2), data_format='channels_last'))
    base_model.add(Conv2D(128, (5, 5), strides=(2, 2), activation='relu'))
    base_model.add(MaxPool2D(pool_size=(2, 2), data_format='channels_last'))
    base_model.add(Conv2D(256, (3, 3), strides=(2, 2), activation='relu'))
    base_model.add(MaxPool2D(pool_size=(2, 2), data_format='channels_last'))

    out = Sequential()
    out.add(Flatten(input_shape=base_model.output_shape[1:]))
    out.add(Dense(56, activation='relu'))
    out.add(Dense(n_classes, activation='softmax'))

    return base_model, out


def get_vgg16(input_shape, n_classes, pretrained=True):
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
    x.add(Dense(n_classes, activation='softmax'))
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


def get_inceptionv3(input_shape, n_classes, pretrained=True):
    if not pretrained:
        base_model = InceptionV3(include_top=True, weights=None, input_shape=input_shape)

    else:
        base_model = InceptionV3(include_top=False, weights='imagenet', input_shape=input_shape)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    output = Dense(n_classes, activation='softmax')(x)

    return base_model, output


get_model = {
    "inceptionv3": get_inceptionv3,
    "vgg16": get_vgg16,
    "test1": get_test1
}


def get_model_name(model_type, empty=False):
    now = datetime.datetime.now()
    name = model_type + '_'
    if empty:
        name = name + "empty_"
    name = name + str(now.month) + '_' + str(now.day) + '-' + str(now.hour) + '_' + str(now.minute) + '_' + str(
        now.second)
    return name


"""
def train_model(model_type, input_shape, n_classes, n_tune_layers, pretrained=True):
    if not pretrained:
        train_empty(model_type, )
    else:
        model = tune_output_layer(model_type, input_shape, n_classes, epochs)
        tune_layers(n_tune_layers)
    # method to tune top (n_tune_layers) layers of a model.
    # get model with already trained new top output layer.
    return null
"""


def fine_tune_trained_model_load(name, model_path, input_shape, n_tune_layers, train_path, val_path, horizontal_flip,
                                 batch_size, epochs, save=True):
    model = load_model(model_path)
    train_generator = get_generator(train_path, batch_size, target_size=(input_shape[0], input_shape[1]),
                                    horizontal_flip=horizontal_flip)
    val_generator = get_generator(val_path, batch_size, target_size=(input_shape[0], input_shape[1]),
                                  horizontal_flip=horizontal_flip)
    for layer in model.layers[:len(model.layers) - n_tune_layers]:
        layer.trainable = False
    for layer in model.layers[len(model.layers) - n_tune_layers:]:
        layer.trainable = True
    from tensorflow.keras.optimizers import SGD
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])

    # we train our model again (this time fine-tuning the top 2 inception blocks
    # alongside the top Dense layers
    model.fit_generator(
        train_generator,
        steps_per_epoch=count_files(train_path) // batch_size,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=count_files(val_path) // batch_size
    )
    model.save(name + "_tuned_" + str(n_tune_layers) + ".h5py")
    return


def finetune_model_last_layer(model_type, input_shape, n_classes, train_path, horizontal_flip, batch_size,
                              epochs, save=True):
    # tune output layer
    base_model, output = get_model[model_type](input_shape, n_classes, pretrained=True)
    train_generator = get_generator(train_path, batch_size, target_size=(input_shape[0], input_shape[1]),
                                    horizontal_flip=horizontal_flip)
    model = Model(inputs=base_model.input, outputs=output(base_model.output))  # output)

    print("The total number of layers is", len(model.layers))
    # for layer in base_model.layers:
    #    layer.trainable = False

    for layer in model.layers[:18]:
        layer.trainable = False

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    name = get_model_name(model_type)
    if save:
        save_summary(name, model)
    model.fit_generator(train_generator, steps_per_epoch=count_files(train_path) // batch_size, epochs=epochs)
    model_path = name + "_last_layer.h5py"
    model.save(model_path)
    # either make a separate function tune_layers so just write it here
    return name, model_path


def train_empty(model_type, input_shape, n_classes, epochs, train_path, val_path, batch_size, horizontal_flip=True,
                save=True):
    base_model, output = get_model[model_type](input_shape, n_classes, pretrained=False)
    train_generator = get_generator(train_path, batch_size, target_size=(input_shape[0], input_shape[1]),
                                    horizontal_flip=horizontal_flip)
    model = Model(inputs=base_model.input, outputs=output(base_model.output))
    print("created model")
    #for layer in model.layers[:18]:
     #   layer.trainable = False
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    name = get_model_name(model_type, empty=True)
    if save:
        save_summary(name, model)
    val_generator = get_generator(val_path, batch_size, target_size=(input_shape[0], input_shape[1]),
                                  horizontal_flip=horizontal_flip)
    create_dir(MODEL_DIR)
    tb = TensorBoard(log_dir=MODEL_DIR+"/{}", histogram_freq=0, write_graph=True, write_images=True)
    earlyStop = EarlyStopping(monitor='val_acc', patience=2)
    if val_path!=None:
        create_dir(join("models", name))
        checkpoint = ModelCheckpoint(join("models", name, "{epoch:02d}-{val_acc:.2f}.hdf5"), monitor='val_acc',
                                     verbose=1, save_best_only=True, mode='max')
        history  = model.fit_generator(train_generator, steps_per_epoch=count_files(train_path) // batch_size,
                                       epochs=epochs, callbacks=[checkpoint, earlyStop], validation_data=val_generator,
                                       validation_steps=count_files(val_path) // batch_size)
    else:
        history  = model.fit_generator(train_generator, steps_per_epoch=count_files(train_path) // batch_size,
                                       epochs=epochs)

    #model.fit_generator(train_generator, steps_per_epoch=count_files(train_path) // batch_size, epochs=epochs,)
     #                  # validation_data=val_generator,
      #                  # validation_steps=count_files(val_path) // batch_size)
    file_path = os.path.join(dirname, "models", name)
    model.save(name+"_empty.h5py")
    return file_path


# def tune_output_layer(model_type, input_shape, n_classes, epochs, train_path, val_path, horizontal_flip, batch_size, save):


def save_summary(name, model):
    file_name = name + '_' + "summary.txt"
    file_path = "models/" + file_name
    with open(file_path, 'w+') as f:  # os.path.join(".", "models", file_name)
        with redirect_stdout(f):
            model.summary()


def get_generator(path, batch_size, target_size, horizontal_flip):
    datagen = ImageDataGenerator(horizontal_flip=horizontal_flip,
                                 preprocessing_function=wp_preprocess_input)
    generator = datagen.flow_from_directory(
        path,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical')
    return generator
