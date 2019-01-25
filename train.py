import tensorflow as tf
import os
from os.path import join
import argparse
import re
import pickle
from keras.models import Sequential, Model, load_model
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, LearningRateScheduler
from keras.regularizers import l1, l2
from keras.utils import multi_gpu_model
from train_utils import get_model_name, get_model, save_summary, get_generator, step_decay, get_optimiser
from rasta.python.models.processing import count_files
from processing.clean_csv import create_dir
from keras import backend as K

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.40
config.gpu_options.allow_growth = True

PATH = os.path.dirname(__file__)

# PARSING ARGUMENTS

parser = argparse.ArgumentParser(description='Description')

parser.add_argument('-t', action="store", default='empty', dest='train_type', help='Training type [empty|retrain|tune]')
parser.add_argument('-m', action="store", dest='model_path',help='Path of the model file')
parser.add_argument('--new_m', action="store", default='N', dest='new_path', help='Save in a new directory [Y|N]')
parser.add_argument('--model_type', action='store', default='test1', dest='model_type', help='Type of model [test1|test2|test3|auto1|vgg16|inceptionv3|resnet50]')
parser.add_argument('-b', action="store", default=30, type=int, dest='batch_size',help='Size of the batch.')
parser.add_argument('-e', action="store",default=10, type=int, dest='epochs',help='Number of epochs')
parser.add_argument('-f', action="store", default=False, type=bool,dest='horizontal_flip',help='Set horizontal flip or not [True|False]')
parser.add_argument('-n', action="store", default=0, type=int,dest='n_layers_trainable',help='Set the number of last trainable layers')
parser.add_argument('-d', action="store", default=0, type=int, dest='sample_n', choices=range(0, 10), metavar='[0-9]', help='Sample Number to use [0-9]')
parser.add_argument('--opt', action="store", default='adam', dest='optimiser', help='Optimiser [adam|rmsprop|adadelta|sgd]')
parser.add_argument('-lr', action="store", default=0.001, type=float, dest='lr', help='Learning Rate for Optimiser')
parser.add_argument('--decay', action="store", default='none', dest='add_decay', help='Add decay to Learning Rate for Optimiser [none|rate_v|step]')
parser.add_argument('-r', action="store", default='l2', dest='add_reg', help='Add regularisation in Conv layers [none|l1|l2]')
parser.add_argument('--alp', action="store", default=0.01, type=float, dest='alpha', help='Value of Alpha for regularizer')
parser.add_argument('--dropout', action="store", default=0.5, type=float, dest='add_drop', help='Add dropout rate [0-1]')
parser.add_argument('--mom', action="store", default=0.0, type=float, dest='add_mom', help='Add momentum to SGD')

args = parser.parse_args()
print("Args: ", args)
MODEL_DIR = "models/logs"

MODEL_TYPE = args.model_type
train_type = args.train_type
BATCH_SIZE = args.batch_size
N_EPOCHS = args.epochs
flip = args.horizontal_flip
N_TUNE = args.n_layers_trainable
SAMPLE_N = args.sample_n
OPT = args.optimiser
LR = args.lr
DECAY = args.add_decay
MOM = args.add_mom
DATA_TYPE = True
if OPT != 'sgd':
    print("Warning: chosen optimiser is not SGD, momentum value is ignored.")
if args.add_reg == 'none':
    REG = None
elif args.add_reg == 'l1':
    REG = l1
else:
    REG = l2

TRAIN_PATH = join(PATH, "data/wiki_small_2_" + str(SAMPLE_N), "small_train")#join(PATH, "data/wiki_small" + str(SAMPLE_N), "smalltrain")
VAL_PATH = join(PATH, "data/wiki_small_2_" + str(SAMPLE_N), "small_val")

INPUT_SHAPE = (224, 224, 3)

if train_type != 'empty':
    # MODEL_TYPE = ''
    if args.model_path is None:
        print("Error: model path should be provided for retraining/tuning.")
    else:
        MODEL_PATH = args.model_path
    name = MODEL_PATH.rsplit('/', 1)[1].replace('hdf5', '')
    model = load_model(MODEL_PATH)
    name = get_model_name(SAMPLE_N, empty=False, model_type=MODEL_TYPE, name=name, n_tune=N_TUNE)
    if args.new_path == 'Y':
        dir_path = join(MODEL_PATH.split('/')[1], train_type + 'tune-' + str(N_TUNE) + '-no-' + str(SAMPLE_N), name)
    else:
        dir_path = join(MODEL_PATH.rsplit('/', 1)[0], name)
    create_dir(dir_path)
else:
    MODEL_TYPE = args.model_type
    MODEL_PATH = None
    base_model, output = get_model[MODEL_TYPE](INPUT_SHAPE, REG, args.alpha, args.add_drop, pretrained=True)
    if re.search('test*', MODEL_TYPE):
        model = Model(inputs=base_model.input, outputs=base_model.output)
    elif MODEL_TYPE == 'auto1':
        DATA_TYPE = False
        model = Model(base_model, output)
    elif MODEL_TYPE == 'vgg16':
        model = Model(inputs=base_model.input, outputs=output(base_model.output))
    else:
        model = Model(inputs=base_model.input, outputs=output)
    name = get_model_name(SAMPLE_N, model_type=MODEL_TYPE, n_tune=N_TUNE)
    dir_path = join("models", name)
    create_dir(dir_path)
changed = False
if N_TUNE > 0:
    for layer in model.layers[:len(model.layers) - N_TUNE]:
        if layer.trainable == True:
            changed = True
        layer.trainable = False
    for layer in model.layers[len(model.layers) - N_TUNE:]:
        if layer.trainable == False:
            changed = True
        layer.trainable = True

params = vars(args)

if MODEL_TYPE == "vgg":
    pre_type = True
else:
    pre_type = False
train_generator = get_generator(TRAIN_PATH, BATCH_SIZE, target_size=(INPUT_SHAPE[0], INPUT_SHAPE[1]),
                                horizontal_flip=flip, pre_type=pre_type, train_type=DATA_TYPE)
try:
    model = multi_gpu_model(model)
    print("multi gpu enabled")
except:
    pass

if train_type == 'empty' or changed:
    model.compile(optimizer=get_optimiser(OPT, LR, DECAY, MOM), loss='categorical_crossentropy', metrics=['accuracy'])
elif LR != 0.001:
    K.set_value(model.optimizer.lr, LR)

save_summary(dir_path, name, model)
with open(join(dir_path, name + '.json'), 'w') as json_file:
    json_file.write(model.to_json())

val_generator = get_generator(VAL_PATH, BATCH_SIZE, target_size=(INPUT_SHAPE[0], INPUT_SHAPE[1]),
                              horizontal_flip=flip, pre_type=pre_type, train_type=DATA_TYPE)
tb = TensorBoard(log_dir=MODEL_DIR + "/" + name, histogram_freq=0, write_graph=True, write_images=True)
earlyStop = EarlyStopping(monitor='val_acc', patience=5)
if VAL_PATH != None:

    checkpoint = ModelCheckpoint(join(dir_path, "{epoch:02d}-{val_acc:.3f}.hdf5"), monitor='val_acc',
                                 verbose=1, save_best_only=True, mode='max')
    if DECAY == 'step':
        lr_decay = LearningRateScheduler(step_decay)
        history = model.fit_generator(train_generator, steps_per_epoch=count_files(TRAIN_PATH) // BATCH_SIZE,
                                      epochs=N_EPOCHS, callbacks=[tb, checkpoint, earlyStop, lr_decay],
                                      validation_data=val_generator,
                                      validation_steps=count_files(VAL_PATH) // BATCH_SIZE)
    else:
        history = model.fit_generator(train_generator, steps_per_epoch=count_files(TRAIN_PATH) // BATCH_SIZE,
                                      epochs=N_EPOCHS, callbacks=[tb, checkpoint, earlyStop], validation_data=val_generator,
                                      validation_steps=count_files(VAL_PATH) // BATCH_SIZE)
else:
    history = model.fit_generator(train_generator, steps_per_epoch=count_files(TRAIN_PATH) // BATCH_SIZE,
                                  epochs=N_EPOCHS)

model.save(join(dir_path, name + ".hdf5"))
# model.save_weights(join(dir_path,'model_weights.h5'))
#model.save(join(dir_path,'final_model.hdf5'))
with open(join(dir_path,'history.pck'), 'wb') as f:
    pickle.dump(history.history, f)
    f.close()


