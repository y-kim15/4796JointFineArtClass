import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Flatten
import os

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

def count_files(folder):
    s = 0
    for t in list(os.walk(folder)):
        s += len(t[2])
    return s
def wp_preprocess_input(x):
    # 'RGB'->'BGR'
    x = x[:, :, ::-1]

    x[:,:,0] -=  133.104
    x[:,:,0] -=  119.973
    x[:,:,0] -=  104.432

    return x
_NUM_CLASSES = 25
_BATCH_SIZE = 25
_PATH = "../wikipaintings_small/wikipaintings_train"#"../rasta/data/wikipaintings_full/wikipaintings_train"
_VAL_PATH = "../wikipaintings_small/wikipaintings_val"#"../rasta/data/wikipaintings_full/wikipaintings_val"
def tfdata_generator(images, labels, is_training, batch_size=_BATCH_SIZE):
  '''Construct a data generator using `tf.Dataset`. '''

  def preprocess_fn(image, label):
      '''Preprocess raw data to trainable input. '''
      x = tf.reshape(tf.cast(image, tf.float32), (299, 299, 3))

      y = tf.one_hot(tf.cast(label, tf.uint8), _NUM_CLASSES)
      return x, y

  dataset = tf.data.Dataset.from_tensor_slices((images, labels))

  if is_training:
    dataset = dataset.shuffle(_BATCH_SIZE)  # depends on sample size
    dataset = dataset.apply(tf.contrib.data.map_and_batch(
        preprocess_fn, batch_size,
        num_parallel_batches=4,  # cpu cores
        drop_remainder=True if is_training else False))
    dataset = dataset.repeat()
    dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)

  return dataset
# Load mnist training data
#(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
#training_set = tfdata_generator(x_train, y_train, is_training=True)
#testing_set = tfdata_generator(x_test, y_test, is_training=False)

train_datagen = image.ImageDataGenerator(
        preprocessing_function=wp_preprocess_input)
test_datagen = image.ImageDataGenerator(preprocessing_function=wp_preprocess_input)
train_generator = train_datagen.flow_from_directory(
        _PATH,
        target_size=(299,299),
        batch_size=_BATCH_SIZE,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        _VAL_PATH,
        target_size=(299, 299),
        batch_size=_BATCH_SIZE,
        class_mode='categorical')
# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299,299,3))

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Flatten()(x)
# let's add a fully-connected layer
#x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(_NUM_CLASSES, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()
# train the model on the new data for a few epochs
#model.fit_generator(...)
model.fit_generator(
        train_generator,
        steps_per_epoch=count_files(_PATH)//_BATCH_SIZE,
        epochs=20)
#model.fit(training_set.make_one_shot_iterator(), steps_per_epoch = len(x_train)//_BATCH_SIZE, epochs=5, validation_data=testing_set.make_one_shot_iterator(),
#validation_steps=len(x_test) // _BATCH_SIZE, verbose=1)
#model.save("inception_v3_wp_small.h5py")
"""
# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)
"""
# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in model.layers[:201]:
   layer.trainable = False
for layer in model.layers[201:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from tensorflow.keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=["accuracy"])

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
model.fit_generator(
        train_generator,
        steps_per_epoch=count_files(_PATH)//_BATCH_SIZE,
        epochs=20,
        validation_data=validation_generator,
        validation_steps=count_files(_VAL_PATH)//_BATCH_SIZE
)
model.save("inception_v3_wp_small_50layers.h5py")
