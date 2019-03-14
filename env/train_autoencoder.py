from keras.layers import Dense, Conv2D, MaxPooling2D, Input, GlobalAveragePooling2D, InputLayer, Flatten
from keras.models import Sequential, Model
from keras.optimizers import SGD
import matplotlib
matplotlib.use("agg")
from matplotlib import pyplot
import os
from os.path import join
from processing.read_images import count_files
from processing.train_utils import fixed_generator, get_generator

PATH = "/cs/tmp/yk30/data"#os.path.dirname(__file__)
# parameter definition
TRAIN_PATH = join(PATH, "id_medium_train")#join(PATH, "data/id_medium", "id_train")
VAL_PATH = join(PATH, "id_medium_val")#join(PATH, "data/wikipaintings_full", "wikipaintings_val")
TEST_PATH = join(PATH, "data/id_medium", "id_test")
INPUT_SHAPE = (224, 224, 3)
BATCH_SIZE = 30
N_EPOCHS = 20
DATA_TYPE = False
MODEL_TYPE ='auto1'
flip = False

# data generators
train_generator = get_generator(TRAIN_PATH, BATCH_SIZE, target_size=(INPUT_SHAPE[0], INPUT_SHAPE[1]),
                                horizontal_flip=flip, train_type=DATA_TYPE, function=MODEL_TYPE)
val_generator = get_generator(VAL_PATH, BATCH_SIZE, target_size=(INPUT_SHAPE[0], INPUT_SHAPE[1]),
                             horizontal_flip=flip, train_type=True, function=MODEL_TYPE)
# define model
# input = Input(shape=INPUT_SHAPE)
# x = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same')(input)
# x = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same')(x)
# x = MaxPooling2D((2, 2), padding='same')(x)
# x = GlobalAveragePooling2D(data_format='channels_last')(x)
# output = Dense(25, activation='linear')(x)
#model = Model(inputs=input, outputs=output)
base_model = Sequential()
base_model.add(InputLayer(INPUT_SHAPE))
base_model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same'))
base_model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same'))
base_model.add(MaxPooling2D((2, 2), data_format='channels_last'))
# compile model
base_model.add(Dense(25, activation='softmax'))
model = Model(inputs=base_model.input, outputs=base_model.output)
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01, momentum=0.9), metrics=['accuracy'])
print("Model summary")
print(model.summary())
# fit model
history = model.fit_generator(fixed_generator(train_generator), steps_per_epoch=count_files(TRAIN_PATH) // BATCH_SIZE, epochs=N_EPOCHS)

# evaluate reconstruction loss
train_mse = model.evaluate_generator(train_generator,verbose=0)
test_mse = model.evaluate_generator(val_generator, verbose=0)
print('> reconstruction error train=%.3f, test=%.3f' % (train_mse, test_mse))


# evaluate the autoencoder as a classifier
def evaluate_autoencoder_as_classifier(model):
    # remember the current output layer
    output_layer = model.layers[-2:]
    # remove the output layer
    model.pop()
    # mark all remaining layers as non-trainable
    for layer in model.layers:
        layer.trainable = False
    # add new output layer
    model.add(Dense(25, activation='softmax'))
    # compile model
    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01, momentum=0.9), metrics=['acc'])
    # fit model
    history = model.fit_generator(fixed_generator(train_generator),
                                  steps_per_epoch=count_files(TRAIN_PATH) // BATCH_SIZE,
                                  epochs=N_EPOCHS)
    # evaluate model
    _, train_acc_1 = model.evaluate_generator(train_generator, verbose=0)
    _, test_acc_1 = model.evaluate(val_generator, verbose=0)
    # put the model back together
    model.pop()
    model.add(output_layer)
    model.compile(loss='mse', optimizer=SGD(lr=0.01, momentum=0.9))
    return train_acc_1, test_acc_1


# add one new layer and re-train only the new layer
# add the number of filters in filter_n, and the number of conv layers by count
def add_layer_to_autoencoder(model, filter_n, count):
    # remember the current output layer
    output_layer = model.layers[-2:]
    # remove the output layer
    model.pop()
    # mark all remaining layers as non-trainable
    for layer in model.layers:
        layer.trainable = False
    # add a new hidden layer
    for i in range(count):
        model.add(Conv2D(filter_n, (3, 3), strides=(1, 1), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), padding='same'))
    # re-add the output layer
    model.add(output_layer)
    # compile model
    model.compile(loss='mse', optimizer=SGD(lr=0.01, momentum=0.9))
    # fit model
    history = model.fit_generator(fixed_generator(train_generator),
                                  steps_per_epoch=count_files(TRAIN_PATH) // BATCH_SIZE,
                                  epochs=N_EPOCHS)
    # evaluate reconstruction loss
    train_mse = model.evaluate(train_generator, verbose=0)
    test_mse = model.evaluate(val_generator, verbose=0)
    print('> reconstruction error train=%.3f, test=%.3f' % (train_mse, test_mse))


struct = {
    1: [128, 2],
    2: [256, 3],
    3: [512, 3],
    4: [512, 3]
}

scores = dict()
train_acc, test_acc = evaluate_autoencoder_as_classifier(model)
print('> classifier accuracy layers=%d, train=%.3f, test=%.3f' % (len(model.layers), train_acc, test_acc))
scores[len(model.layers)] = (train_acc, test_acc)
# add layers and evaluate the updated model
n_layers = 4

for i in range(n_layers):
    # add layer
    add_layer_to_autoencoder(model, struct[i][0], struct[i][1])
    # evaluate model
    train_acc, test_acc = evaluate_autoencoder_as_classifier(model)
    print('> classifier accuracy layers=%d, train=%.3f, test=%.3f' % (len(model.layers), train_acc, test_acc))
    # store scores for plotting
    scores[len(model.layers)] = (train_acc, test_acc)
# plot number of added layers vs accuracy
keys = scores.keys()
pyplot.plot(keys, [scores[k][0] for k in keys], label='train', marker='.')
pyplot.plot(keys, [scores[k][1] for k in keys], label='test', marker='.')
pyplot.legend()
pyplot.show()
