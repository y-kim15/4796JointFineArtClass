import finetune
import tensorflow as tf

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allocator_type = 'BFC'
# config.gpu_options.per_process_gpu_memory_fraction = 0.40
config.gpu_options.allow_growth = True

TRAIN_PATH = "data/wiki_small0/smalltrain"  # rasta/data/wikipaintings_full/wikipaintings_train"
VAL_PATH =  "data/wiki_small0/smallval"  # "rasta/data/wikipaintings_full/wikipaintings_val"

N_CLASSES = 25
BATCH_SIZE = 20
MODEL_TYPE = "vgg16"
INPUT_SHAPE = (224, 224, 3)
N_TUNE_LAYERS = 5
HORIZONTAL_FLIP = False
N_EPOCHS = 10
name, model_path = finetune.train_empty(model_type=MODEL_TYPE, input_shape=INPUT_SHAPE, epochs=N_EPOCHS, train_path=TRAIN_PATH, val_path=VAL_PATH, batch_size=BATCH_SIZE)
name = "test1_empty_1_12-15_0_10_empty.h5py_tuned_5.h5py"
model_path = "./" + name
#finetune.fine_tune_trained_model_load(name, model_path, INPUT_SHAPE, N_TUNE_LAYERS, TRAIN_PATH, VAL_PATH, HORIZONTAL_FLIP, BATCH_SIZE, epochs=N_EPOCHS)
