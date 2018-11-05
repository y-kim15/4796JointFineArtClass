from fine_tune import *

TRAIN_PATH="rasta/data/wikipaintings_full/wikipaintings_train"
VAL_PATH="rasta/data/wikipaintings_full/wikipaintings_val"

N_CLASSES=25
BATCH_SIZE=10
finetune_model(model_type="vgg16", input_shape=(224,244,3), n_classes=N_CLASSES, n_tune_layers=1, train_path=TRAIN_PATH,
    val_path=VAL_PATH, horizontal_flip=True, batch_size=BATCH_SIZE)
