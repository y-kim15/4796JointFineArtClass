### To set up the environment
    source ./bin/activate
    pip install -r requirements.txt
Note that this automatically installs tensorflow-gpu for gpu support

### Train a new model
    python3 train.py --model_type <model_type> -d <dataset index>
### Retrain an existing model
    python3 train.py -t retrain --model_type <model_type> -m <model_path> -d <dataset index>
### Tune an existing model
    python3 train.py -t tune --model_type <model_type> -m <model_path> -n <n_tune> -d <dataset index>

Required arguments:
* --model_type <>: type of model to initialise (resnet50, vgg16, inceptionv3)
Optional arguments:
* -d <>: ith dataset to use for training (0-4) - default: 0
* -b <>: batch size - default: 30
* -e <>: number of epochs - default: 10
* --opt <>: type of optimiser to use - default: Adam
* --decay <> : adding decay (step|rate|<float>|exp) - default: None
* -f : adding horizontal flip - default: False
* -r <> : adding regularisation in Convolutional layers (none|l1|l2) - default: None
* --alp <> : set value of alpha - default = 0.0
* --dropout <> : adding dropout layer as a penultimate layer, define rate - default: 0.0
* --mom <> : set momentum if SGD is chosen - default: 0.0
* -ln <>: set last N layers to reinitialise with rest copied to a new model


### Evaluate the accuracy of model
    python3 evaluate_result.py -m <model_path> -d <data_path>
    
* Default model_path is the saved latest optimal model
* Default data_path is data/wikipaintings_small/wikipaintings_test 

Optional arguments:
* -k : value of Top-k accuracy to find, can be written in comma separated values (e.g. -k 1,3,5)
* -cm : get confusion matrix
* -pr : get precision and recall curve
* --class : get classification report containing accuracy, precision and recall per class
* -show : display all plots on screen
* -save <directory_path> : save all plots in <directory_path> - default: ../models/eval/<model_name>