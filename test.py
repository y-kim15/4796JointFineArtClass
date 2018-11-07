from fine_tune import *
import tensorflow as tf

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allocator_type = 'BFC'
#config.gpu_options.per_process_gpu_memory_fraction = 0.40
config.gpu_options.allow_growth = True


TRAIN_PATH="rasta/data/wikipaintings_full/wikipaintings_train"
VAL_PATH="rasta/data/wikipaintings_full/wikipaintings_val"

N_CLASSES=25
BATCH_SIZE=25
finetune_model(model_type="vgg16", input_shape=(224,224,3), n_classes=N_CLASSES, n_tune_layers=1, train_path=TRAIN_PATH,
    val_path=VAL_PATH, horizontal_flip=True, batch_size=BATCH_SIZE)
