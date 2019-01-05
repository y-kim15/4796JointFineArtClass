import finetune
import tensorflow as tf

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allocator_type = 'BFC'
# config.gpu_options.per_process_gpu_memory_fraction = 0.40
config.gpu_options.allow_growth = True

TRAIN_PATH = "data/wikipaintings_small/wikipaintings_train" #"data/wiki_small2/wiki_train"  # rasta/data/wikipaintings_full/wikipaintings_train"
VAL_PATH =  "data/wikipaintings_small/wikipaintings_val" #""data/wiki_small2/wiki_val"  # "rasta/data/wikipaintings_full/wikipaintings_val"

N_CLASSES = 25
BATCH_SIZE = 40
MODEL_TYPE = "test1"
INPUT_SHAPE = (224, 224, 3)
N_TUNE_LAYERS = 5
HORIZONTAL_FLIP = False
N_EPOCHS = 10
name, model_path = finetune.train_empty(model_type=MODEL_TYPE, input_shape=INPUT_SHAPE, n_classes=N_CLASSES, epochs=N_EPOCHS,
                                        train_path=TRAIN_PATH, val_path=VAL_PATH, batch_size=BATCH_SIZE)

name = "vgg16_empty_11_26-16_4_53_s3"
model_path = "vgg16_empty_11_26-16_4_53_empty.h5py"
#finetune.fine_tune_trained_model_load(name, model_path, INPUT_SHAPE, N_TUNE_LAYERS, TRAIN_PATH, VAL_PATH,
 #HORIZONTAL_FLIP, BATCH_SIZE, epochs=N_EPOCHS)

