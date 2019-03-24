# Fine-art Paintings Classification with Transfer Learning

### To set up the environment
    cd env
    virtualenv -p python3 venv
    source venv/bin/activate
    pip3 install -r ../requirements.txt
This will enable using python3.
Note that this automatically installs tensorflow-gpu for GPU support

### Train a new model
    python3 train.py --model_type <model_type> -d <dataset index>
### Retrain an existing model
    python3 train.py -t retrain --model_type <model_type> -m <model_path> -d <dataset index>
### Tune an existing model
    python3 train.py -t tune --model_type <model_type> -m <model_path> -n <n_tune> -d <dataset index>

````
optional arguments:
  -h, --help            show this help message and exit
  -t TRAIN_TYPE         Training type [empty|retrain|tune] - (default: empty)
  -m MODEL_PATH         Path of the model file
  --new_m NEW_PATH      Save in a new directory [new path]
  --model_type MODEL_TYPE
                        Type of model
                        [auto1|vgg16|inceptionv3|resnet50]
  -b BATCH_SIZE         Size of the batch. - (default: 30)
  -e EPOCHS             Number of epochs - (default: 10)
  -f                    Set horizontal flip or not
  -n N_LAYERS_TRAINABLE
                        Set the number of last trainable layers
  --opt OPTIMISER       Optimiser [adam|rmsprop|adadelta|sgd]
  -lr {0}               Learning Rate for Optimiser
  --decay {none,rate,step,exp,decimal}
                        Add decay to Learning Rate for Optimiser
  -r {none,l1,l2}       Add regularisation in Conv layers
  --alp [0.0-1.0]       Value of Alpha for regulariser
  --init INITIALISER    Initialiser [glorot_uniform|he_normal]
  --dropout [0.0-1.0]   Add dropout rate
  --mom [0.0-1.0]       Add momentum to SGD
  -ln LAYER_NO          Select the layer to replace
  -tr [0-2]             Select initial weight type [0:random|1:imagenet keras|2:classification_models package]      
  -w                    Add class weights to resolve imbalanced data
deprecated:
  -d [0-4]              Sample Number to use [0-4]
````


### Evaluate the accuracy of model
    python3 evaluate_result.py -m <model_path> -d <data_path>

* Default model_path is the saved latest optimal model
* Default data_path is data/wikipaintings_small/wikipaintings_test

```
optional arguments:
  -h, --help       show this help message and exit
  -t TYPE          evaluation type [acc: accuracy from test set|pred: predict a single image]
  -m MODEL_PATH    Path of the model file
  -d DATA_PATH     Path of test set - (default: data/wikipaintings_small/wikipaintings_test)
  -ds [f,s]        Set test set type, full or small, in absence of -d
  -k TOP_K         Top-k accuracy to compute - (default: 1,3,5)
  -cm              Get Confusion Matrix
  -pr              Get Precision Recall Curve
  --report         Get Classification Report
  --show            Display graphs
  -s               Save graphs
  --save SAVE_PATH  Give save location - (default: models/eval/<model_name>)
```

```python3
parser.add_argument('--his', action="store", dest='plot_his', help='Plot history, choose which to plot [l|a|b (default)]')
parser.add_argument('-f', action="store", dest="file", help='Name of history file to plot (extension pck)')
parser.add_argument('--model_name', action="store", dest='model_name', help='Model types/name: Mandatory to call --his')
parser.add_argument('--act', action="store", dest='act', help='Visualise activation function of layer (layer name or index)')
parser.add_argument('--roc', action="store_true", dest='get_roc', help='Get Roc Curve')
```
