from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Flatten, Dropout
from rasta.python.utils.utils import wp_preprocess_input
from rasta.python.models.processing import count_files
from contextlib import redirect_stdout
import datetime
import os

dirname = os.path.dirname(__file__)
# Code influenced by keras application examples
# that from rasta and from ...

def get_vgg16(input_shape, n_classes, pretrained=True):
    input = Input(shape=input_shape)
    if not pretrained:
        base_model = VGG16(include_top=True, weights='None', input_shape=input_shape)
        return base_model
    else:
        base_model = VGG16(include_top=True, weights='imagenet', input_shape=input_shape)

    x = Sequential()
    #x.add(Flatten(input_shape=base_model.output_shape[1:]))
    #x.add(Dense(512, activation='relu'))
    #x.add(Dropout(0.5))
    #x.add(Dense(256, activation='relu'))
    #x.add(Dropout(0.5))
    #x.add(Dense(n_classes, activation='softmax'))
    return base_model, x

    #model = Model(input=base_model.input, output=x(base_model.output))

    #x = base_model.output
    #x = Dense(4096, activation='relu')(x)
    #x = Dropout(0.5)(x)
    #x = Dense(4096, activation='relu')(x)
    #x = Dropout(0.5)(x)
    #output = Dense(n_classes, activation='softmax')(x)
    #model.outputs = [model.layers[-1].output]
    #model.layers[-1].outbound_nodes = []
    #model.add(Dense(num_class, activation='softmax'))

    #return base_model, x

def get_inceptionv3(input_shape, n_classes, pretrained=True):
    if not pretrained:
        base_model = InceptionV3(include_top=True, weights='None', input_shape=input_shape)
        return base_model
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
    "vgg16": get_vgg16
}

def get_model_name(model_type, empty=False):
    now = datetime.datetime.now()
    name = model_type + '_'
    if empty:
        name = name + "empty_"
    name = name + str(now.month) + '_' + str(now.day) + '-' + str(now.hour) +'_'+ str(now.minute) +'_'+ str(now.second)
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

def finetune_model(model_type, input_shape, n_classes, n_tune_layers, train_path, val_path, horizontal_flip, batch_size, epochs=20, save=True):
    # tune output layer
    base_model, output = get_model[model_type](input_shape, n_classes, pretrained=True)
    train_generator = get_generator(train_path, batch_size, target_size=(input_shape[0], input_shape[1]), horizontal_flip=horizontal_flip)
    model = Model(inputs=base_model.input, outputs=base_model.output)#output(base_model.output))#output)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])
    name = get_model_name(model_type)
    if save:
        save_summary(name, model)
    """
    #for layer in base_model.layers:
    #    layer.trainable = False
    for layer in model.layers[:13]:
        layer.trainable = False

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])
    name = get_model_name(model_type)
    if save:
        save_summary(name, model)
    model.fit_generator(train_generator, steps_per_epoch=count_files(train_path)//batch_size, epochs=epochs)
    val_generator = get_generator(val_path, batch_size, target_size=(input_shape[0], input_shape[1]), horizontal_flip=horizontal_flip)
"""
    # either make a separate function tune_layers so just write it here
    return

def train_empty(model_type, input_shape, n_classes, epochs, train_path, val_path, horizontal_flip, batch_size, save=True):
    model = get_model[model_type](input_shape, n_classes, pretrained=False)
    train_generator = get_generator(train_path, batch_size, target_size=(input_shape[0], input_shape[1]), horizontal_flip=horizontal_flip)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])
    name = get_model_name(model_type, empty=True)
    if save:
        save_summary(name, model)
    val_generator = get_generator(val_path, batch_size, target_size=(input_shape[0], input_shape[1]), horizontal_flip=horizontal_flip)
    model.fit_generator(train_generator, steps_per_epoch=count_files(train_path)//batch_size, epochs=epochs, validation_data=val_generator,
        validation_steps=count_files(val_path)//batch_size)
    file_path = os.path.join(dirname, "models", name)
    model.save(file_path)
    return file_path

#def tune_output_layer(model_type, input_shape, n_classes, epochs, train_path, val_path, horizontal_flip, batch_size, save):


def save_summary(name, model):
    file_name = name + '_' + "summary.txt"
    file_path = "models/" + file_name
    with open(file_path, 'w+') as f: #os.path.join(".", "models", file_name)
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
