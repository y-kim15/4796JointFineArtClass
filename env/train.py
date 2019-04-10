import tensorflow as tf
import os
from os.path import join
import argparse, time, pickle, json
from keras.models import load_model
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.regularizers import l1, l2
from processing.train_utils import get_model_name, get_new_model, save_summary, get_generator, get_optimiser, copy_weights, fixed_generator, set_trainable_layers, merge_two_dicts, lr_decay_callback, lr_decay_callback2
from processing.clean_csv import create_dir
from processing.read_images import count_files
from keras import backend as K
import re, sys, csv
from collections import Counter, OrderedDict

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.40
config.gpu_options.allow_growth = True

PATH = os.path.dirname(__file__)
# options for data path
lab = "/cs/tmp/yk30/data/wikipaintings_full/wikipaintings_train#/cs/tmp/yk30/data/wikipaintings_full/wikipaintings_val"
lap = "data/wikipaintings_full/wikipaintings_train#data/wikipaintings_full/wikipaintings_val"
# PARSING ARGUMENTS

parser = argparse.ArgumentParser(description='Description')

### basic training by fit, use of callbacks, command line arguments are inspired from RASTA

parser.add_argument('-t', action="store", default='empty', dest='train_type', help='Training type [empty|retrain|tune]')
parser.add_argument('-m', action="store", dest='model_path',help='Path of the model file')
parser.add_argument('--new_p', action="store", dest='new_path', help='New Path to save the train directory')
parser.add_argument('--model_type', action='store', required=True, dest='model_type', help='Type of model [auto1|auto2|vgg16|inceptionv3|resnet50]')
parser.add_argument('-b', action="store", default=30, type=int, dest='batch_size',help='Size of the batch')
parser.add_argument('-e', action="store",default=10, type=int, dest='epochs',help='Number of epochs')
parser.add_argument('-f', action="store_true", default=False, dest='horizontal_flip',help='Set horizontal flip')
parser.add_argument('-n', action="store", default='0', dest='n_layers_trainable',help='Set the number of trainable layers, as range [a-b] or [csv] or single value [x] for last x layers or nested')
parser.add_argument('-dp', action="store", default='lab', dest='data_path',help='# separated path to dataset for train and val')
parser.add_argument('--opt', action="store", default='adam', dest='optimiser', help='Optimiser [adam|rmsprop|adadelta|sgd|nadam]')
parser.add_argument('-lr', action="store", default=0.001, type=float, dest='lr', help='Initial learning rate')
parser.add_argument('--decay', action="store", default='none', dest='add_decay', help='Add decay to Learning rate [rate, step, exp, decimal value]')
parser.add_argument('-r', action="store", default='none', dest='add_reg', choices=['none', 'l1', 'l2'], help='Add regularisation in penultimate Dense layer')
parser.add_argument('--alp', action="store", default=0.0, type=float, dest='alpha', metavar='[0.0-1.0]', help='Value of Alpha hyperparameter for Regularizer')
parser.add_argument('--dropout', action="store", default=0.0, type=float, dest='add_drop', metavar='[0.0-1.0]', help='Add dropout rate')
parser.add_argument('--init', action="store", dest='add_init', help='Initialiser [he_normal|glorot_uniform]')
parser.add_argument('--mom', action="store", default=0.9, type=float, dest='add_mom', metavar='[0.0-1.0]', help='Add momentum to SGD')
parser.add_argument('-ln', action="store", default=None, dest='rep_layer_no', help='Set indices of a layer/range/point onwards to copy to new model (keep)')
parser.add_argument('-tr', action="store", type=int, default=1, choices=[0,1,2], dest='pretrained', help="Get pretrained model [0:random weights, 1: keras without top layers, 2: keras full model]")
parser.add_argument('-path', action="store", default="", dest="path", help='Path to save the train output file for train_hyp case')
parser.add_argument('-w', action="store_true", default=False, dest='add_wei', help='Add class weight for imbalanced data')

args = parser.parse_args()
print("Args: ", args)

####### Decode cmd line arguments #########
# TODO
MODEL_DIR = join(PATH, 'models', 'logs')#"/cs/tmp/yk30/models/logs"
MODEL_TYPE = args.model_type
train_type = args.train_type
BATCH_SIZE = args.batch_size
N_EPOCHS = args.epochs
flip = args.horizontal_flip
N_TUNE = args.n_layers_trainable
SAMPLE_N = 'full'
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

####### Get data path #################
if args.data_path != None:
    if args.data_path == 'lap':
        args.data_path = lap
    elif args.data_path == 'lab':
        args.data_path = lab
    TRAIN_PATH = args.data_path.rsplit('#',1)[0]
    VAL_PATH = args.data_path.rsplit('#',1)[1]
else:
    TRAIN_PATH = join(PATH, "data/wikipaintings_full", "wikipaintings_train")#join(PATH, "data", "id_medium_small_" + str(SAMPLE_N), "small_train")##join(PATH, "data/wiki_small" + str(SAMPLE_N), "smalltrain")
    VAL_PATH = join(PATH, "data/wikipaintings_full", "wikipaintings_val")# join(PATH, "data", "id_medium_small_" + str(SAMPLE_N), "small_val")#

if args.model_type=='inceptionv3':
    INPUT_SHAPE = (299, 299, 3)
else:
    INPUT_SHAPE = (224, 224, 3)

######## Load model ##################
if train_type != 'empty':
    MODEL_PATH = args.model_path
    if MODEL_PATH is None:
        sys.exit("Error: model path should be provided for retraining/tuning.")
    else:
        MODEL_PATH = args.model_path

    name = MODEL_PATH.rsplit('/', 1)[1].replace('hdf5', '')
    # if weights file, again load new model instance
    try:
        model = load_model(MODEL_PATH)
    except ValueError:
        model, _ = get_new_model(MODEL_TYPE, INPUT_SHAPE, REG, args.alpha, args.add_init, args.add_drop, args.pretrained, N_TUNE)
        model.load_weights(MODEL_PATH)
        changed = True
    name = get_model_name(SAMPLE_N, type=train_type, model_type=MODEL_TYPE, name=name, n_tune=N_TUNE)


    if train_type == 'tune':
        new_model, DATA_TYPE = get_new_model(MODEL_TYPE, INPUT_SHAPE, REG, args.alpha, args.add_init, args.add_drop, args.pretrained, N_TUNE)
        if args.rep_layer_no!= None:
            # give this value 0 if you want to copy the whole model to the new with end weights
            if args.rep_layer_no.isdigit() and int(args.rep_layer_no) == 0:
                model.load_weights(join(MODEL_PATH.rsplit('/',1)[0], '_end_weights.h5'))
            else:
                model = copy_weights(model, new_model, args.rep_layer_no)
            if model == None:
                sys.exit('Incorrect command line attribute of replacing layer weights!')
            changed = True
    if args.new_path is not None:
        dir_path = join(args.new_path, name)
    else:
        dir_path = join(MODEL_PATH.rsplit('/', 1)[0], name)
    create_dir(dir_path)

else:
    MODEL_PATH = None
    model, DATA_TYPE = get_new_model(MODEL_TYPE, INPUT_SHAPE, REG, args.alpha, args.add_init, args.add_drop, args.pretrained, N_TUNE)
    name = get_model_name(SAMPLE_N, type=train_type, model_type=MODEL_TYPE, n_tune=N_TUNE)
    if args.path != "":
        dir_path = args.path
    else:
        dir_path = join("/cs/tmp/yk30/models",name)
        create_dir(dir_path)

if args.model_type is not None:
    model, changed = set_trainable_layers(model, N_TUNE)
    if not changed and train_type=='tune':
        changed = True


params = vars(args)
train_generator = get_generator(TRAIN_PATH, BATCH_SIZE, target_size=(INPUT_SHAPE[0], INPUT_SHAPE[1]),
                                horizontal_flip=flip, train_type=DATA_TYPE, function=MODEL_TYPE)
WEIGHTS = None
if args.add_wei:
   counter = Counter(train_generator.classes)
   max_val = float(max(counter.values()))
   class_weights = {class_id: max_val / num_images for class_id, num_images in counter.items()}
   #class_weights = {class_id: 1.0 / num_images for class_id, num_images in counter.items()}
   WEIGHTS = class_weights

# Compile for new model and recompile for any models with changed number of tainable layers
if train_type == 'empty' or changed:
    if re.search('auto*', MODEL_TYPE):
        model.compile(optimizer=get_optimiser(OPT, LR, DECAY, MOM, N_EPOCHS), loss='mean_squared_error')
    else:
        model.compile(optimizer=get_optimiser(OPT, LR, DECAY, MOM, N_EPOCHS), loss='categorical_crossentropy', metrics=['accuracy'])

###### Saves initial model summary and user input params ####
if args.path == "":
    save_summary(dir_path, name, model)
    with open(join(dir_path, '_model.json'), 'w') as json_file:
        json_file.write(model.to_json())
    with open(join(dir_path, '_param.json'), 'w') as json_file:
        json.dump(vars(args), json_file)

####### Model fitting with callbacks #################
tb = TensorBoard(log_dir=MODEL_DIR + "/" + name, histogram_freq=0, write_graph=True, write_images=True)
earlyStop = EarlyStopping(monitor='val_loss', patience=5)
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
        if not DECAY.isdigit() or DECAY != None or DECAY != 'rate':
            if DECAY == 'step':
                decay_rate = 0.5
                lr_decay = lr_decay_callback(LR, decay_rate)
            else:
                decay_rate = 0.1
                lr_decay = lr_decay_callback2(LR, decay_rate)
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

# if it is not for grid serach save the rest
if args.path == "":
    extra2 = {'train_acc':history.history['acc'][-1],'val_acc': history.history['val_acc'][-1], 'e':history.history['val_acc'].index(max(history.history['val_acc'])),
         'max_val_acc': max(history.history['val_acc']),'run_time': str(end-start)}
    extra = merge_two_dicts(extra1, extra2)
else:
    extra = extra1
orig = vars(args)
orig['path'] = dir_path + '/' + name
data = merge_two_dicts(orig, extra)
ordered_dict = OrderedDict(sorted(data.items(), key=lambda t: t[0]))

####### Saves training history, the final model #############
with open(join(dir_path, '_history.pck'), 'wb') as f:
    pickle.dump(history.history, f)
    f.close()
if args.path == "":
    model.save_weights(join(dir_path, add + '_end_weights.h5'))
    with open(join(dir_path, '_param.json'), 'w') as f:
        json.dump(ordered_dict, f)
else:
    # if it is for gridsearch, write the param output to the output file collating
    # the result of all sub traiings.
    if not os.path.isfile(join(dir_path, '_output.csv')):
        head = True
    else:
        head = False
    FIELDNAMES = sorted(list(ordered_dict.keys()))
    with open(join(dir_path, '_output.csv'), 'a', newline='') as f:
        w = csv.DictWriter(f, FIELDNAMES)
        if head:
            w.writeheader()
        w.writerow(ordered_dict)
