from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, TensorBoard
from keras.wrappers.scikit_learn import KerasClassifier
from keras.wrappers.scikit_learn import to_categorical
from keras.models import Sequential, load_model, Model
from keras.regularizers import l1,l2
from keras import backend as K
from keras.utils import multi_gpu_model
from processing.train_utils import get_generator, get_model_name, get_new_model, copy_weights, get_optimiser, save_summary
from processing.clean_csv import create_dir
from processing.read_images import count_files
from sklearn.model_selection import GridSearchCV
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt import gp_minimize
import argparse
from os.path import join
import os
import tensorflow as tf
from time import time, sleep
import pickle
import json

import types

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.40
config.gpu_options.allow_growth = True

PATH = os.path.dirname(__file__)

parser = argparse.ArgumentParser(description='Description')

parser.add_argument('-t', action="store", default='empty', dest='train_type', help='Training type [empty|retrain|tune]')
parser.add_argument('-m', action="store", dest='model_path',help='Path of the model file')
parser.add_argument('--new_m', action="store", default='N', dest='new_path', help='Save in a new directory [Y|N]')
parser.add_argument('--model_type', action='store', default='test1', dest='model_type', required=True, help='Type of model [test1|test2|test3|auto1|vgg16|inceptionv3|resnet50]')
parser.add_argument('-b', action="store", default=30, type=int, dest='batch_size',help='Size of the batch.')
parser.add_argument('-e', action="store",default=10, type=int, dest='epochs',help='Number of epochs')
parser.add_argument('-f', action="store_true", default=False, dest='horizontal_flip',help='Set horizontal flip or not')
parser.add_argument('-n', action="store", default=0, type=int,dest='n_layers_trainable',help='Set the number of last trainable layers')
parser.add_argument('-d', action="store", default=0, type=int, dest='sample_n', choices=range(0, 5), metavar='[0-4]', help='Sample Number to use [0-4]')
parser.add_argument('--opt', action="store", default='adam', dest='optimiser', help='Optimiser [adam|rmsprop|adadelta|sgd]')
parser.add_argument('-lr', action="store", default=0.001, type=float, choices=range(0,1),dest='lr', help='Learning Rate for Optimiser')
parser.add_argument('--decay', action="store", default='none', dest='add_decay', choices=['none', 'rate', 'step', 'rate', 'dec'], help='Add decay to Learning Rate for Optimiser')
parser.add_argument('-r', action="store", default='none', dest='add_reg', choices=['none', 'l1', 'l2'], help='Add regularisation in Conv layers')
parser.add_argument('--alp', action="store", default=0.0, type=float, dest='alpha',  metavar='[0.0-1.0]', help='Value of Alpha for regularizer')
parser.add_argument('--dropout', action="store", default=0.0, type=float, dest='add_drop', metavar='[0.0-1.0]', help='Add dropout rate')
parser.add_argument('--mom', action="store", default=0.0, type=float, dest='add_mom',metavar='[0.0-1.0]', help='Add momentum to SGD')
parser.add_argument('-ln', action="store", type=int, dest='layer_no', help='Select the layer to replace')

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

changed = False
TRAIN_PATH = join(PATH, "data/wiki_small_2_" + str(SAMPLE_N), "small_train")#join(PATH, "data/wiki_small" + str(SAMPLE_N), "smalltrain")
VAL_PATH = join(PATH, "data/wiki_small_2_" + str(SAMPLE_N), "small_val")


INPUT_SHAPE = (224, 224, 3)


class KerasBatchClassifier(KerasClassifier):

    def fit(self, X, y, **kwargs):

        # taken from keras.wrappers.scikit_learn.KerasClassifier.fit ###################################################
        if self.build_fn is None:
            self.model = self.__call__(**self.filter_sk_params(self.__call__))
        elif isinstance(self.build_fn, Model):
            self.model = self.build_fn
        elif not isinstance(self.build_fn, types.FunctionType) and not isinstance(self.build_fn, types.MethodType):
            self.model = self.build_fn(**self.filter_sk_params(self.build_fn.__call__))
        else:
            self.model = self.build_fn(**self.filter_sk_params(self.build_fn))

        loss_name = self.model.loss
        if hasattr(loss_name, '__name__'):
            loss_name = loss_name.__name__

        #if loss_name == 'categorical_crossentropy' and len(y.shape) != 2:
         #   y = to_categorical(y)

        ################################################################################################################

        train_generator = get_generator(VAL_PATH, BATCH_SIZE, target_size=(INPUT_SHAPE[0], INPUT_SHAPE[1]),
                              horizontal_flip=flip, train_type=DATA_TYPE, function=MODEL_TYPE)

        tb = TensorBoard(log_dir=MODEL_DIR + "/" + name, histogram_freq=0, write_graph=True, write_images=True)
        earlyStop = EarlyStopping(monitor='val_acc', patience=10)


        if VAL_PATH != None:

            val_generator = get_generator(VAL_PATH, BATCH_SIZE, target_size=(INPUT_SHAPE[0], INPUT_SHAPE[1]),
                                          horizontal_flip=flip, train_type=DATA_TYPE, function=MODEL_TYPE)

            checkpoint = ModelCheckpoint(join(dir_path, "{epoch:02d}-{val_acc:.3f}.hdf5"), monitor='val_acc',
                                     verbose=5, save_best_only=True, mode='max')
            start = time()
            self.__history = self.model.fit_generator(train_generator, steps_per_epoch=count_files(TRAIN_PATH) // BATCH_SIZE,
                                      epochs=N_EPOCHS, callbacks=[tb, checkpoint, earlyStop], validation_data=val_generator,
                                      validation_steps=count_files(VAL_PATH) // BATCH_SIZE)
        else:
            start = time()
            self.__history = self.model.fit_generator(train_generator,
                                                      steps_per_epoch=count_files(TRAIN_PATH) // BATCH_SIZE,
                                                      epochs=N_EPOCHS, callbacks=[tb, earlyStop])
        end = time()
        print("time elapsed: ", str(end-start))
        self.model.save(join(dir_path, name + ".hdf5"))
        add = {'time': str(end-start)}
        with open(join(dir_path, '_param.json')) as f:
            data = json.load(f)

        data.update(add)

        with open(join(dir_path, '_param.json'), 'w') as f:
            json.dump(data, f)
        self.model.save_weights(join(dir_path, '_end_weights.h5'))
        with open(join(dir_path, 'history.pck'), 'wb') as f:
            pickle.dump(self.__.history, f)
            f.close()
        return self.__history

    def score(self, X, y, **kwargs):
        kwargs = self.filter_sk_params(Sequential.evaluate, kwargs)

        loss_name = self.model.loss
        if hasattr(loss_name, '__name__'):
            loss_name = loss_name.__name__
        if loss_name == 'categorical_crossentropy' and len(y.shape) != 2:
            y = to_categorical(y)
        outputs = self.model.evaluate(X, y, **kwargs)
        if type(outputs) is not list:
            outputs = [outputs]
        for name, output in zip(self.model.metrics_names, outputs):
            if name == 'acc':
                return output
        raise Exception('The model is not configured to compute accuracy. '
                        'You should pass `metrics=["accuracy"]` to '
                        'the `model.compile()` method.')

    @property
    def history(self):
        return self.__history


if train_type != 'empty':
    # MODEL_TYPE = ''
    MODEL_PATH = args.model_path
    if MODEL_PATH is None:
        print("Error: model path should be provided for retraining/tuning.")
    else:
        MODEL_PATH = args.model_path

    name = MODEL_PATH.rsplit('/', 1)[1].replace('hdf5', '')
    model = load_model(MODEL_PATH)
    name = get_model_name(SAMPLE_N, type=train_type, model_type=MODEL_TYPE, name=name, n_tune=N_TUNE)
    if train_type == 'tune':
        new_model, DATA_TYPE = get_new_model(MODEL_TYPE, INPUT_SHAPE, REG, args.alpha, args.add_drop, pretrained=True)
        model = copy_weights(model, new_model, args.layer_no)
        changed = True
    if args.new_path == 'Y':
        dir_path = join(MODEL_PATH.split('/')[0], MODEL_PATH.split('/')[1], train_type +'-tune-'+str(N_TUNE), name)
    else:
        dir_path = join(MODEL_PATH.rsplit('/', 1)[0], name)
    create_dir(dir_path)

else:
    MODEL_TYPE = args.model_type
    MODEL_PATH = None
    model, DATA_TYPE = get_new_model(MODEL_TYPE, INPUT_SHAPE, REG, args.alpha, args.add_drop, pretrained=True)
    name = get_model_name(SAMPLE_N, type=train_type, model_type=MODEL_TYPE, n_tune=N_TUNE)
    dir_path = join("models", name)
    create_dir(dir_path)

if N_TUNE > 0:
    for layer in model.layers[:len(model.layers) - N_TUNE]:
        if layer.trainable == True:
            changed = True
        layer.trainable = False
    for layer in model.layers[len(model.layers) - N_TUNE:]:
        if layer.trainable == False:
            changed = True
        layer.trainable = True
else:
    for layer in model.layers:
        if layer.trainable == False:
            changed = True
        layer.trainable = True
try:
    model = multi_gpu_model(model)
    print("multi gpu enabled")
except:
    pass

if train_type == 'empty' or changed:
    if MODEL_TYPE == 'auto1':
        model.compile(optimizer=get_optimiser(OPT, LR, DECAY, MOM, N_EPOCHS), loss='mean_squared_error')
    else:
        model.compile(optimizer=get_optimiser(OPT, LR, DECAY, MOM, N_EPOCHS), loss='categorical_crossentropy', metrics=['accuracy'])
elif LR != 0.001:
    K.set_value(model.optimizer.lr, LR)

save_summary(dir_path, name, model)
with open(join(dir_path, '_model.json'), 'w') as json_file:
    json_file.write(model.to_json())
with open(join(dir_path, '_param.json'), 'w') as json_file:
    json.dump(vars(args), json_file)  # json_file.write(vars(args))
model.save_weights(join(dir_path, '_ini_weights.h5'))

new_model = KerasBatchClassifier(build_fn=model, epochs=N_EPOCHS)

learning_rate = [0.001, 0.01, 0.1]
dropout_rate = [0.25, 0.5]
epsilon = [None, 1e-1, 1e-3]
param_grid = dict(learning_rate=learning_rate, epsilon=epsilon, dropout_rate=dropout_rate)

space = [Real(10**-3, 10**-1, "log-uniform", name='learning_rate'),
         Integer(30, 50, name='batch_size'),
         Real(0.25, 0.30, name='dropout_rate'),
         Integer(30, 35, name='epochs')]

@use_named_args(space)
def optimise(**params):
    __history = new_model.fit(X=None, y=None, batch_size=BATCH_SIZE, epochs=N_EPOCHS)
    return __history['val_loss'][-1]

res_gp = gp_minimize(optimise, space, n_calls=10, n_jobs=-1, random_state=0)








