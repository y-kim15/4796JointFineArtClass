# Fine-art Paintings Classification with Transfer Learning

### To set up the environment
    ./build.sh
    source venv/bin/activate
To be run in GPU supported Scientific Linux lab clients
Note that this automatically installs tensorflow-gpu

### To download image files
Image files for training was downloaded from Lecoutre et al. source of RASTA. Full dataset can be downloaded:
Note it has a total size of 18GB so would recommend to store in the scratch space.
* Full test set (large) was used for all the test results reported. Testing takes around 8-10 minutes from the lab machine

  cd data
  wget www.lamsade.dauphine.fr/~bnegrevergne/webpage/software/rasta/wikipaintings_full.tgz
  tar xzvf wikipaintings_full.tgz
  cd ../

### Model Evaluation
#### Evaluate the accuracy of model on small test set
The small wikipaintings test data set from RASTA are stored in data/wikipaintings_small/wikipaintings_test and are set as default test path.

    python3 evaluate_result.py -t acc -m <model_path> [-cm --report --roc --show -s]

To use the large wikipaintings test set for full evaluation, pass the path to wikipaintings_full/wikipaintings_test using -d

    python3 evaluate_result.py -t acc -m <model_path> [-cm --report --roc -show -s]

#### Predict from a given image
There are example images which can be used for prediction stored in data/images which the models have not seen before from train/val/test.

    python3 evaluate_result.py -t pred -m <model_path> -d data/images/<image_file_name>

#### Plot history plot
    python3 evaluate_result.py --his b -f <history_file_path> [--show -s]
#### Plot activation maps
    python3 evaluate_result.py -m <model_path> -d <image_path> --act <layer_no/name> [--show -s]

* Default model_path is the saved latest optimal model
* Default image path for an image to predict on, generate activation map from

````
  usage: evaluate_result.py [-h] [-t TYPE] [-cv CV] [-m MODEL_PATH]
                            [-d DATA_PATH] [-ds {f,s}] [-dp] [-k TOP_K] [-cm]
                            [--report] [--show] [-s] [--save SAVE_PATH]
                            [--his PLOT_HIS] [-f FILE] [--model_name MODEL_NAME]
                            [--act ACT] [--roc]

  Description

  optional arguments:
    -h, --help            show this help message and exit
    -t TYPE               Type of Evaluation [acc-predictive accuracy of model,
                          pred-predict an image][acc|pred]
    -cv CV                Evaluate Cross Validation Output and Save [path to csv
                          to save] to be used by train_hyp
    -m MODEL_PATH         Path of the model file
    -d DATA_PATH          Path of test data
    -ds {f,s}             Choose the size of test set, full or small
    -dp                   Set to test in lab
    -k TOP_K              Top-k accuracy to compute
    -cm                   Get Confusion Matrix
    --report              Get Classification Report
    --show                Display graphs
    -s                    Save graphs
    --save SAVE_PATH      Specify save location
    --his PLOT_HIS        Plot history, choose which to plot [l|a|b (default)]
    -f FILE               Name of history file to plot: Reqruied for --his
    --model_name MODEL_NAME
                          Model types/name: Required for --his
    --act ACT             Visualise activation function of layer [layer name or
                          index]
    --roc                 Get Roc Curve
````

### Model Training
For all training, the average time taken for 1 epoch is 50 minutes.
Should have dataset saved under env/data/as <wikipaintings_full>/<wikipaintings_train> and <wikipaintings_full>/<wikipaintings_val>
or add a new dataset with the same file hierarchy and pass on cmd line as:
    -dp data/<>/<\_train>#data/<>/<\_val>
Each train/val/test directory should contain 25 subdirectories for classes with corresponding images

#### Train a new model
    python3 train.py --model_type <model_type> -dp <# separated path to train and val set>
#### Retrain an existing model
    python3 train.py -t retrain --model_type <model_type> -m <model_path>
#### Tune an existing model
    python3 train.py -t tune --model_type <model_type> -m <model_path> -n <n_tune> -ln <layers_to_copy>

To view tensorboard for monitoring the training process use:
    tensorboard --logdir <path>


````
usage: train.py [-h] [-t TRAIN_TYPE] [-m MODEL_PATH] [--new_p NEW_PATH]
                --model_type MODEL_TYPE [-b BATCH_SIZE] [-e EPOCHS] [-f]
                [-n N_LAYERS_TRAINABLE] [-dp DATA_PATH] [--opt OPTIMISER]
                [-lr LR] [--decay ADD_DECAY] [-r {none,l1,l2}]
                [--alp [0.0-1.0]] [--dropout [0.0-1.0]] [--init ADD_INIT]
                [--mom [0.0-1.0]] [-ln REP_LAYER_NO] [-tr {0,1,2}]
                [-path PATH] [-w]

Description

optional arguments:
  -h, --help            show this help message and exit
  -t TRAIN_TYPE         Training type [empty|retrain|tune]
  -m MODEL_PATH         Path of the model file
  --new_p NEW_PATH      New Path to save the train directory
  --model_type MODEL_TYPE
                        Type of model [auto1|auto2|vgg16|inceptionv3|resnet50]
  -b BATCH_SIZE         Size of the batch
  -e EPOCHS             Number of epochs
  -f                    Set horizontal flip
  -n N_LAYERS_TRAINABLE
                        Set the number of trainable layers, as range [a-b] or
                        [csv] or single value [x] for last x layers or nested
  -dp DATA_PATH         # separated path to dataset for train and val
  --opt OPTIMISER       Optimiser [adam|rmsprop|adadelta|sgd|nadam]
  -lr LR                Initial learning rate
  --decay ADD_DECAY     Add decay to Learning rate [rate, step, exp, decimal
                        value]
  -r {none,l1,l2}       Add regularisation in penultimate Dense layer
  --alp [0.0-1.0]       Value of Alpha hyperparameter for Regularizer
  --dropout [0.0-1.0]   Add dropout rate
  --init ADD_INIT       Initialiser [he_normal|glorot_uniform]
  --mom [0.0-1.0]       Add momentum to SGD
  -ln REP_LAYER_NO      Set indices of a layer/range/point onwards to copy to
                        new model (keep)
  -tr {0,1,2}           Get pretrained model [0:random weights, 1: keras
                        without top layers, 2: keras full model]
  -path PATH            Path to save the train output file for train_hyp case
  -w                    Add class weight for imbalanced data
deprecated:
  -d [0-4]              Sample Number to use [0-4] (previously used for small scale training)
````
#### Example
    python3 train.py --model_type vgg16 -tr 1 -e 3 -b 30 -n 3


Note the below guideline is described for completeness only for cross validation as the implementation
was used extensively due to time constraints. Cross validation implementation also requires predefined smaller dataset
which should be created using read_images - get_small_from_dataset

### More Training - GridSearch
#### GridSearch on different parameter combination
    python3 train_hyp.py -t params [-j <json_file>] [-s]
#### Train one model config via CV
    python3 train_hyp.py -t cv -cv <k> [-s] -m <model_name>

```
usage: train_hyp.py [-h] [-t TYPE] [-cv CV] [-j JSON_FILE] [-s]
                    [--save SAVE_PATH] [-m MODEL]

Description

optional arguments:
  -h, --help        show this help message and exit
  -t TYPE           Type of Training [params-GridSearch|cv-Cross Validation]
  -cv CV            Set K for K-fold CV
  -j JSON_FILE      Path to json file containing parameter values
  -s                Save the output in directory
  --save SAVE_PATH  Path to save the directory
  -m MODEL          Name of architecture to use [vgg16|inceptionv3|resnet50]
```
