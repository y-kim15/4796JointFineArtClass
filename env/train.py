import tensorflow as tf
import os
from os.path import join
import argparse
import time
import pickle, json
from keras.models import load_model
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.regularizers import l1, l2
from processing.train_utils import get_model_name, get_new_model, save_summary, get_generator, step_decay, \
    get_optimiser, exp_decay, copy_weights, fixed_generator, set_trainable_layers, merge_two_dicts, lr_decay_callback,lr_decay_callback2
from processing.clean_csv import create_dir
from processing.read_images import count_files
from keras import backend as K
import re
from collections import Counter
import sys, csv
from collections import OrderedDict
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.40
config.gpu_options.allow_growth = True

PATH = os.path.dirname(__file__)
lab = "/cs/tmp/yk30/data/wikipaintings_full/wikipaintings_train!/cs/tmp/yk30/data/wikipaintings_full/wikipaintings_val"
lap = "data/wikipaintings_full/wikipaintings_train!data/wikipaintings_full/wikipaintings_val"
# PARSING ARGUMENTS

parser = argparse.ArgumentParser(description='Description')

# TODO:
# upload wiki small dataset sep later on
#default="/cs/tmp/yk30/data/wikipaintings_full/wikipaintings_train!/cs/tmp/yk30/data/wikipaintings_full/wikipaintings_val"
parser.add_argument('-t', action="store", default='empty', dest='train_type', help='Training type [empty|retrain|tune|hyp]')
parser.add_argument('-m', action="store", dest='model_path',help='Path of the model file')
parser.add_argument('--new_p', action="store", dest='new_path', help='Save in a new directory')
parser.add_argument('--model_type', action='store', dest='model_type', help='Type of model [auto1|auto2|vgg16|inceptionv3|resnet50]')
parser.add_argument('--rasta_model', action='store', dest='rasta_model', help='Type of rasta models [alexnet_empty|custom_resnet')
parser.add_argument('-b', action="store", default=30, type=int, dest='batch_size',help='Size of the batch.')
parser.add_argument('-e', action="store",default=10, type=int, dest='epochs',help='Number of epochs')
parser.add_argument('-f', action="store_true", default=False, dest='horizontal_flip',help='Set horizontal flip or not')
parser.add_argument('-n', action="store", default='0', dest='n_layers_trainable',help='Set the number of trainable layers, range [a-b] or [csv] or single value [x] for last x layers only')
parser.add_argument('-dp', action="store", default='lab', dest='data_path',help='Optional Full path to dataset')
parser.add_argument('-d', action="store", default=0, type=int, dest='sample_n', choices=range(0, 5), metavar='[0-4]', help='Sample Number to use [0-4]')
parser.add_argument('--opt', action="store", default='adam', dest='optimiser', help='Optimiser [adam|rmsprop|adadelta|sgd]')
parser.add_argument('-lr', action="store", default=0.001, type=float, dest='lr', help='Learning Rate for Optimiser')
parser.add_argument('--decay', action="store", default='none', dest='add_decay', choices=['none', 'rate', 'step', 'exp', 'dec'], help='Add decay to Learning Rate for Optimiser')
parser.add_argument('-r', action="store", default='none', dest='add_reg', choices=['none', 'l1', 'l2'], help='Add regularisation in Conv layers')
parser.add_argument('--alp', action="store", default=0.0, type=float, dest='alpha', metavar='[0.0-1.0]', help='Value of Alpha for Regularizer')
parser.add_argument('--dropout', action="store", default=0.0, type=float, dest='add_drop', metavar='[0.0-1.0]', help='Add dropout rate')
parser.add_argument('--init', action="store", dest='add_init', help='Define type of Initialiser')
parser.add_argument('--mom', action="store", default=0.9, type=float, dest='add_mom', metavar='[0.0-1.0]', help='Add momentum to SGD')
parser.add_argument('-ln', action="store", default=None, dest='rep_layer_no', help='Select a layer/range/point onwards to copy to new model (keep)')
parser.add_argument('-tr', action="store", type=int, default=1, choices=[0,1,2], dest='pretrained', help="Get pretrained model [0:None, 1: keras, 2: classification_models]")
parser.add_argument('-path', action="store", default="", dest="path", help='Path to save the train output file for train_hyp case')
parser.add_argument('-w', action="store_true", default=False, dest='add_wei', help='Add class weight for imbalanaced data')

args = parser.parse_args()
print("Args: ", args)
MODEL_DIR = "models/logs"

try:
    if args.model_type is not None:
        MODEL_TYPE = args.model_type
    elif args.rasta_model is not None:
        MODEL_TYPE = args.rasta_model
        print("Warning: chosen model type is rasta, training type -t is ignored.")
    else:
        raise ValueError()
except ValueError:
        sys.exit("ValueError: either model_type or rasta_model should be defined.")

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
if args.data_path != None:
    if args.data_path == 'lap':
        args.data_path = lap
    else:
        args.data_path = lab
    TRAIN_PATH = args.data_path.rsplit('!',1)[0]
    VAL_PATH = args.data_path.rsplit('!',1)[1]
else:
    TRAIN_PATH = join(PATH, "data/wikipaintings_full", "wikipaintings_train")#join(PATH, "data", "id_medium_small_" + str(SAMPLE_N), "small_train")##join(PATH, "data/wiki_small" + str(SAMPLE_N), "smalltrain")
    VAL_PATH = join(PATH, "data/wikipaintings_full", "wikipaintings_val")# join(PATH, "data", "id_medium_small_" + str(SAMPLE_N), "small_val")#

if args.rasta_model is not None and (args.rasta_model=='alexnet_empty' or args.rasta_model=='decaf'):
    INPUT_SHAPE = (227, 227, 3)
else:
    INPUT_SHAPE = (224, 224, 3)

if train_type != 'empty':
    MODEL_PATH = args.model_path
    if MODEL_PATH is None:
        sys.exit("Error: model path should be provided for retraining/tuning.")
    else:
        MODEL_PATH = args.model_path

    name = MODEL_PATH.rsplit('/', 1)[1].replace('hdf5', '')
    model = load_model(MODEL_PATH)
    print("no of layers : ", len(model.layers))
    name = get_model_name(SAMPLE_N, type=train_type, model_type=MODEL_TYPE, name=name, n_tune=N_TUNE)
    if train_type == 'tune':
        new_model, DATA_TYPE = get_new_model(MODEL_TYPE, INPUT_SHAPE, REG, args.alpha, args.add_init, args.add_drop, args.pretrained, N_TUNE)
        if args.rep_layer_no!= None:
            # give this value 0 if you want to copy the whole model to the new with end weights
            if args.rep_layer_no.isdigit() and int(args.rep_layer_no) == 0:
                model = model.load_weights(join(MODEL_PATH.rsplit('/',1)[0], '_end_weights.h5'))
            else:
                model = copy_weights(model, new_model, args.rep_layer_no)
            if model == None:
                sys.exit('Incorrect command line attribute of replacing layer weights!')
            changed = True
    if args.new_path is not None:
        dir_path = join(args.new_path, name)
	#dir_path = join(MODEL_PATH.rsplit('/', 2)[0], name)
    else:
        dir_path = join(MODEL_PATH.rsplit('/', 1)[0], name)
    create_dir(dir_path)

else:
    #MODEL_TYPE = args.model_type
    print("MODEL TYPE IS ", MODEL_TYPE)
    MODEL_PATH = None
    model, DATA_TYPE = get_new_model(MODEL_TYPE, INPUT_SHAPE, REG, args.alpha, args.add_init, args.add_drop, args.pretrained, N_TUNE)
    name = get_model_name(SAMPLE_N, type=train_type, model_type=MODEL_TYPE, n_tune=N_TUNE)
    if args.path != "":
        dir_path = args.path
    else:
        dir_path = join("/cs/scratch/yk30/models",name)
	#dir_path = join("models", name)
        create_dir(dir_path)

if args.model_type is not None:
    model, changed = set_trainable_layers(model, N_TUNE)
if  args.rasta_model=='cust_resnet':
    model, changed = set_trainable_layers(model, N_TUNE)

params = vars(args)
train_generator = get_generator(TRAIN_PATH, BATCH_SIZE, target_size=(INPUT_SHAPE[0], INPUT_SHAPE[1]),
                                horizontal_flip=flip, train_type=DATA_TYPE, function=MODEL_TYPE)
WEIGHTS = None
if args.add_wei:
   counter = Counter(train_generator.classes)
   #max_val = float(max(counter.values()))
   #class_weights = {class_id: max_val / num_images for class_id, num_images in counter.items()}
   class_weights = {class_id: 1.0 / num_images for class_id, num_images in counter.items()}
   WEIGHTS = class_weights



print("Total number of layers is ", str(len(model.layers)))

if train_type == 'empty' or changed:
    if re.search('auto*', MODEL_TYPE):
        model.compile(optimizer=get_optimiser(OPT, LR, DECAY, MOM, N_EPOCHS), loss='mean_squared_error')
    else:
        model.compile(optimizer=get_optimiser(OPT, LR, DECAY, MOM, N_EPOCHS), loss='categorical_crossentropy', metrics=['accuracy'])
elif LR != 0.001:
    K.set_value(model.optimizer.lr, LR)

if args.path == "":
    save_summary(dir_path, name, model)
    with open(join(dir_path, '_model.json'), 'w') as json_file:
        json_file.write(model.to_json())
    with open(join(dir_path, '_param.json'), 'w') as json_file:
        json.dump(vars(args), json_file)  # json_file.write(vars(args))
    model.save_weights(join(dir_path, '_ini_weights.h5'))

tb = TensorBoard(log_dir=MODEL_DIR + "/" + name, histogram_freq=0, write_graph=True, write_images=True)
earlyStop = EarlyStopping(monitor='val_loss', patience=3)
start = time.time()
add = ""
if VAL_PATH != None:
    val_generator = get_generator(VAL_PATH, BATCH_SIZE, target_size=(INPUT_SHAPE[0], INPUT_SHAPE[1]),
                                  horizontal_flip=flip, train_type=DATA_TYPE, function=MODEL_TYPE)
    if re.search('auto*', MODEL_TYPE):
        checkpoint = ModelCheckpoint(join(dir_path, add+"{epoch:02d}-{val_loss:.3f}.hdf5"), monitor='val_loss',
                                     verbose=1, save_best_only=True, mode='min')
        history = model.fit_generator(fixed_generator(train_generator), steps_per_epoch=count_files(TRAIN_PATH) // BATCH_SIZE,
                                      epochs=N_EPOCHS, callbacks=[checkpoint], validation_data=fixed_generator(val_generator), validation_steps=count_files(VAL_PATH) // BATCH_SIZE)

    else:
        if args.path != "":
            add = name + "-"
        checkpoint = ModelCheckpoint(join(dir_path, add+"{epoch:02d}-{val_acc:.3f}-{val_loss:.3f}.hdf5"), monitor='val_loss',
                                     verbose=1, save_best_only=True, mode='min')
        callbacks = [tb, earlyStop, checkpoint]
        #if args.path == "":
        #    callbacks.append(checkpoint)
        if not DECAY.isdigit() or DECAY != None or DECAY != 'rate':
            if DECAY == 'step':
                decay_rate = 0.5
                lr_decay = lr_decay_callback(LR, decay_rate)
                #lr_decay = LearningRateScheduler(step_decay)
            else:
                decay_rate = 0.1
                lr_decay = lr_decay_callback2(LR, decay_rate)
                #lr_decay = LearningRateScheduler(exp_decay)
            callbacks.append(lr_decay)

        history = model.fit_generator(train_generator, steps_per_epoch=count_files(TRAIN_PATH) // BATCH_SIZE,
                                      epochs=N_EPOCHS, callbacks=callbacks,
                                      validation_data=val_generator,
                                      validation_steps=count_files(VAL_PATH) // BATCH_SIZE, class_weight=WEIGHTS)
else:
    history = model.fit_generator(train_generator, steps_per_epoch=count_files(TRAIN_PATH) // BATCH_SIZE,
                                  epochs=N_EPOCHS, class_weight=WEIGHTS)
end =time.time()
model.save(join(dir_path, name + ".hdf5"))

extra1 = {'train_loss': history.history['loss'][-1],'val_loss':
    history.history['val_loss'][-1]}
if args.path == "":
    extra2 = {'train_acc':history.history['acc'][-1],'val_acc': history.history['val_acc'][-1], 'e':history.history['val_acc'].index(max(history.history['val_acc'])),
         'max_val_acc': max(history.history['val_acc']),'run_time': str(end-start)}
    extra = merge_two_dicts(extra1, extra2)
else:
    extra = extra1
orig = vars(args)
orig['path'] = dir_path + '/' + name
data = OrderedDict()
data = merge_two_dicts(orig, extra)
with open(join(dir_path, add + '_history.pck'), 'wb') as f:
    pickle.dump(history.history, f)
    f.close()
if args.path == "":
    model.save_weights(join(dir_path, add + '_end_weights.h5'))
    #with open(join(dir_path, '_param.json')) as f:
    #    data = json.load(f)
    #data.update(extra)
    with open(join(dir_path, '_param.json'), 'w') as f:
        json.dump(data, f)
else:
    if not os.path.isfile(join(dir_path, '_output.csv')):
        head = True
    else:
        head = False
    FIELDNAMES = list(data.keys()).sort()
    with open(join(dir_path, '_output.csv'), 'a', newline='') as f:
        w = csv.DictWriter(f, FIELDNAMES)
        if head:
            w.writeheader()
        w.writerow(row for row in zip(*(data[key] for key in FIELDNAMES)))
