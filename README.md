# Fine-art Paintings Classification with Transfer Learning

### To set up the environment
    ./build.sh
    source venv/bin/activate
To be run in GPU supported Scientific Linux lab clients.
Note that this automatically installs tensorflow-gpu

### To download image files
Image files for training was downloaded from Lecoutre et al. source of [RASTA](https://github.com/bnegreve/rasta) where full credits are given as stated in the report,
that the project has taken the existing work as a basis to extend on.

The full dataset can be downloaded using the below command. Note it has a total size of 18GB so would recommend to store in the scratch space.

Full test set (large) was used for all the test results reported. It contains around 8000 images, on average 300 images per class.
Testing takes around 8-10 minutes from the lab machine.

````
  cd data
  wget www.lamsade.dauphine.fr/~bnegrevergne/webpage/software/rasta/wikipaintings_full.tgz
  tar xzvf wikipaintings_full.tgz
  cd ../
````

### Model Evaluation
For quick run of all tests below, use:
````
    ./test.sh
````

This will evaluate the model accuracy on small test set with confusion matrices and classification report, make prediction on an image, plot history and activation maps.

N/B Please ensure tkinter is available for plot generation.

#### Evaluate the accuracy of model on small test set
The small wikipaintings test data set from RASTA are stored in data/wikipaintings_small/wikipaintings_test and are set as default test path.
The small test set contains around 10 images per class and takes around 15 seconds to test. Note that model evaluation results analysed in the report are using the **full** test set
- hence the outcome in terms of accuracy, confusion matrix may differ.

Adding a *-s* flag with *-t acc* will also save the generated accuracy results to json file.

Categorical accuracy for Small *wikipaintings_test* set:

| small | ResNet50 | VGG16 |
|-------|----------|-------|
| Top-1 | 40.8     | 39.6  |
| Top-3 | 66.0     | 68.4  |
| Top-5 | 82.0     | 81.6  |

Categorical accuracy for Large *wikipaintings_test* set:

| full  | ResNet50 | VGG16 |
|-------|----------|-------|
| Top-1 | 49.1     | 50.3  |
| Top-3 | 75.4     | 78.4  |
| Top-5 | 86.3     | 88.3  |


    python3 evaluate_result.py -t acc -m <model_path> [-cm --report --roc --show] -s

To use the large wikipaintings test set for full evaluation, pass the path to *wikipaintings_full/wikipaintings_test* using *-d*

    python3 evaluate_result.py -t acc -m <model_path> [-cm --report --roc -show -s]

#### Predict from a given image
There are example images which can be used for prediction stored in data/images which the models have not seen before from train/val/test.
The models' predictive ability on unseen images are analysed in detail in the report and in the supported material *Model Prediction Report*.

    python3 evaluate_result.py -t pred -m <model_path> -d data/images/<image_file_name>

Example run:
    python3 evaluate_result.py -t pred --m_type vgg -d gustav-klimt_the-sunflower-1907.jpg

#### Plot history plot
Some example history files generated from sample training are saved to test this command.

General run:

    python3 evaluate_result.py --his <b,l,a> -f <history_file_path> [--show -s]

Example run:

    python3 evaluate_result.py --his b -f models/resnet50_models/resnet_eg_history.pck --show -s

#### Plot activation maps
As stated in the report, activation maps can be visualised for convolutional layers by layer index or name.
Layer name is available in \_summary.txt file for each model. Without a specific image passed on cmd line, default image of *Sunflowers* by Vincent van Gogh will be used.
(equivalent to the report).

This is used to generate the maps included in the report.
General run:

    python3 evaluate_result.py -m <model_path> -d <image_path> --act <layer_no/name> [--show -s]

Example run:

    python3 evaluate_result.py --m_type vgg --act block1_conv1 --show
    python3 evaluate_result.py --m_type resnet --act conv1 --show

* Default model_path is the saved latest optimal model
* Default image path for an image to predict on, generate activation map from

````
usage: evaluate_result.py [-h] [-t TYPE] [-cv CV] [-m MODEL_PATH]
                          [--m_type {resnet,vgg}] [-d DATA_PATH] [-ds {f,s}]
                          [-k TOP_K] [-cm] [--cm_type] [--report] [--show]
                          [-s] [--save SAVE_PATH] [--his PLOT_HIS] [-f FILE]
                          [--model_name MODEL_NAME] [--act ACT] [--roc]

Description

optional arguments:
  -h, --help            show this help message and exit
  -t TYPE               Type of Evaluation [acc-predictive accuracy of model,
                        pred-predict an image][acc|pred]
  -cv CV                Evaluate Cross Validation Output and Save [path to csv
                        to save] to be used by train_hyp
  -m MODEL_PATH         Path of the model file
  --m_type {resnet,vgg}
                        Choose the type of ready trained model to use for
                        evaluation/prediction
  -d DATA_PATH          Path of test data
  -ds {f,s}             Choose the size of test set, full or small
  -k TOP_K              Top-k accuracy to compute
  -cm                   Get Confusion Matrix
  --cm_type             Get un-labelled Confusion Matrix
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
Should have dataset saved under env/data/  as

    <wikipaintings_full>/<wikipaintings_train> and <wikipaintings_full>/<wikipaintings_val>

exactly or add a new dataset with the same file hierarchy and pass on cmd line as:

    -dp data/<>/<\_train>#data/<>/<\_val>

Each train/val/test directory should contain 25 subdirectories for classes with corresponding images

#### Train a new model
    python3 train.py --model_type <model_type> -dp <# separated path to train and val set>
#### Retrain an existing model
    python3 train.py -t retrain --model_type <model_type> -m <model_path>
#### Tune an existing model
    python3 train.py -t tune --model_type <model_type> -m <model_path> -n <n_tune> -ln <layers_to_copy>

#### Example
train a new VGG model, pretrained with its top dense layers replaced with newly initialised dense layers,
setting top 3 layers trainable with batch size 30 and epoch 3 and create a directory with all model related files under
/cs/scratch/<id>/models. Add class weights.

        python3 train.py --model_type vgg16 -tr 1 -e 3 -b 30 -n 3 --new_p /cs/scratch/<id>/models -w

Retrain the saved model, setting layers indexed 110 to 120 inclusive to be trainable with initial learning rate of 0.00001 of Adam
optimiser with factor of 10 decay, with batch size 60 and epoch 10.

        python3 train.py -t retrain -m models/resnet50_model/resnet50_06-0.517-2.090.hdf5 -e 10 -b 60 -n 110-120 -lr 0.00001 --decay rate


To view tensorboard for monitoring the training process use:

    tensorboard --logdir <models/logs/path_to_dir_saved_with_events_file>


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
