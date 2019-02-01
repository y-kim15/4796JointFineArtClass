# Fine-art Paintings Classification with Transfer Learning

### To set up the environment
    cd env
    python3 -m venv .
    source bin/activate
    pip install -r ../requirements.txt
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
  --new_m NEW_PATH      Save in a new directory [Y|N]
  --model_type MODEL_TYPE
                        Type of model
                        [test1|test2|test3|auto1|vgg16|inceptionv3|resnet50]
  -b BATCH_SIZE         Size of the batch. - (default: 30)
  -e EPOCHS             Number of epochs - (default: 10)
  -f                    Set horizontal flip or not 
  -n N_LAYERS_TRAINABLE
                        Set the number of last trainable layers
  -d [0-4]              Sample Number to use [0-4]
  --opt OPTIMISER       Optimiser [adam|rmsprop|adadelta|sgd]
  -lr {0}               Learning Rate for Optimiser
  --decay {none,rate,step,rate,dec}
                        Add decay to Learning Rate for Optimiser
  -r {none,l1,l2}       Add regularisation in Conv layers
  --alp [0.0-1.0]       Value of Alpha for regularizer
  --dropout [0.0-1.0]   Add dropout rate
  --mom [0.0-1.0]       Add momentum to SGD
  -ln LAYER_NO          Select the layer to replace
````


### Evaluate the accuracy of model
    python3 evaluate_result.py -m <model_path> -d <data_path>
    
* Default model_path is the saved latest optimal model
* Default data_path is data/wikipaintings_small/wikipaintings_test 

```
optional arguments:
  -h, --help       show this help message and exit
  -m MODEL_PATH    Path of the model file - (default: current trained model)
  -d DATA_PATH     Path of test data - (default: wikipaintings_small/wikipaintings_test)
  -k TOP_K         Top-k accuracy to compute - (default: 1,3,5)
  -cm              Get Confusion Matrix
  -pr              Get Precision Recall Curve
  --report         Get Classification Report
  -show            Display graphs
  -save SAVE_PATH  Save graphs, give save location - (default: models/eval/<model_name>)
```