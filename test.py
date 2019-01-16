from finetune import fit_model
import tensorflow as tf
import os
from os.path import join
import argparse


config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allocator_type = 'BFC'
# config.gpu_options.per_process_gpu_memory_fraction = 0.40
config.gpu_options.allow_growth = True

PATH = os.path.dirname(__file__)

# PARSING ARGUMENTS

parser = argparse.ArgumentParser(description='Description')

parser.add_argument('-t', action="store", default='empty', dest='train_type', help='Training type [empty|retrain|tune]')
parser.add_argument('-m', action="store", dest='model_path',help='Path of the model file')
parser.add_argument('--model_type', action='store', default='test1', dest='model_type', help='Type of model [test1|vgg|inception]')
parser.add_argument('-b', action="store", default=30, type=int, dest='batch_size',help='Size of the batch.')
parser.add_argument('-e', action="store",default=10, type=int, dest='epochs',help='Number of epochs')
parser.add_argument('-f', action="store", default=False, type=bool,dest='horizontal_flip',help='Set horizontal flip or not [True|False]')
parser.add_argument('-n', action="store", default=0, type=int,dest='n_layers_trainable',help='Set the number of last trainable layers')
parser.add_argument('-d', action="store", default=0, type=int, dest='sample_n', choices=range(0, 10), metavar='[0-9]', help='Sample Number to use [0-9]')

args = parser.parse_args()
print("Args: ", args)

train_type = args.train_type
BATCH_SIZE = args.batch_size
N_EPOCHS = args.epochs
flip = args.horizontal_flip
N_TUNE = args.n_layers_trainable
SAMPLE_N = args.sample_n

if train_type != 'empty':
    TRAIN_TYPE = False
    if args.model_path is None:
        print("Error: model path should be provided for retraining/tuning.")
    else:
        MODEL_PATH = args.model_path
    MODEL_TYPE = None
else:
    TRAIN_TYPE = True
    MODEL_TYPE = args.model_type
    MODEL_PATH = None

params = vars(args)

TRAIN_PATH = join(PATH, "data/wiki_small" + str(SAMPLE_N), "smalltrain")
VAL_PATH = join(PATH, "data/wiki_small" + str(SAMPLE_N), "smallval")

INPUT_SHAPE = (224, 224, 3)
fit_model(MODEL_TYPE, INPUT_SHAPE, N_EPOCHS, TRAIN_PATH, VAL_PATH, BATCH_SIZE, SAMPLE_N, TRAIN_TYPE, horizontal_flip=flip)

#name, model_path = finetune.train_empty(model_type=MODEL_TYPE, input_shape=INPUT_SHAPE, epochs=N_EPOCHS, train_path=TRAIN_PATH, val_path=VAL_PATH, batch_size=BATCH_SIZE)
name = "test1_empty_1_12-15_0_10_empty.h5py_tuned_5.h5py"
model_path = "./" + name
#finetune.fine_tune_trained_model_load(name, model_path, INPUT_SHAPE, N_TUNE_LAYERS, TRAIN_PATH, VAL_PATH, HORIZONTAL_FLIP, BATCH_SIZE, epochs=N_EPOCHS)
#name, dir_path = fit_model("test1", INPUT_SHAPE, N_EPOCHS, TRAIN_PATH, VAL_PATH, BATCH_SIZE, SAMPLE_N, True)
