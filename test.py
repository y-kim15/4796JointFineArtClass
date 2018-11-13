from fine_tune import *
import tensorflow as tf

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allocator_type = 'BFC'
#config.gpu_options.per_process_gpu_memory_fraction = 0.40
config.gpu_options.allow_growth = True


TRAIN_PATH="rasta/data/wiki_small/wiki_train"#"rasta/data/wikipaintings_full/wikipaintings_train"
VAL_PATH="rasta/data/wiki_small/wiki_val"#"rasta/data/wikipaintings_full/wikipaintings_val"

N_CLASSES=25
BATCH_SIZE=25
MODEL_TYPE="vgg16"
INPUT_SHAPE=(224,224,3)
N_TUNE_LAYERS=5
HORIZONTAL_FLIP=True
name, model_path = finetune_model_last_layer(model_type=MODEL_TYPE, input_shape=INPUT_SHAPE, n_classes=N_CLASSES, n_tune_layers=N_TUNE_LAYERS, train_path=TRAIN_PATH,
    val_path=VAL_PATH, horizontal_flip=HORIZONTAL_FLIP, batch_size=BATCH_SIZE)
fine_tune_trained_model_load(name, model_path, INPUT_SHAPE, N_TUNE_LAYERS, TRAIN_PATH, VAL_PATH, HORIZONTAL_FLIP, BATCH_SIZE)
