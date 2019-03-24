from processing.clean_csv import create_dir
from processing.train_utils import merge_two_dicts, get_best_comb_from_csv, save_ordered
import os
from os.path import join
import datetime
import subprocess
import time
import argparse
import json
import itertools

# TODO:
#  more options could add (mom, decay etc in common or individually)
models = {
    "common": {
        "--dropout": [0.3, 0.4, 0.5],
        "--opt": ["adam", "rmsprop", "sgd"],
        "-lr": [0.001, 0.01, 0.1]
    },
    "resnet50": {
        "--model_type": "resnet50",
        "-e": 5, #[5,10,15]
        "-b": 60, #[60, 70, 80],
        "-n": 3 #[3, "169-172", "164-168", "159-163"],
    },
    "vgg16": {
        "--model_type": "vgg16",
        "-e": [10],
        "-b": [30, 35, 40],
    }
}


PATH = os.path.dirname(__file__)
FINAL_PATH = join(PATH, 'models')

parser = argparse.ArgumentParser(description='Description')

parser.add_argument('-t', action="store", default="params", dest='type', help='Type of Training [params|series]')
parser.add_argument('-j', action="store", default=None, dest='json_file', help='Path to json file containing parameter values')
parser.add_argument('-s', action="store_true",default=False, dest='save', help='Save the output in directory') #if false, will only report the best on cmd line and json
parser.add_argument('-m', action="store", dest='model', help='Name of architecture to use [vgg16|inceptionv3|resnet50]')
args = parser.parse_args()
TYPE = args.type
if TYPE == 'params':
    if args.save:
        now = datetime.datetime.now()
        name = 'train_hyp' + '_' + str(now.month) + '-' + str(now.day) + '-' + str(now.hour) + '-' + str(now.minute)
        PATH = os.path.dirname(__file__)
        print(PATH)
        FINAL_PATH = join(PATH, 'models', name)
        create_dir(join(PATH, 'models', name))
        path = ['-path', FINAL_PATH]
    else:
        path = []

    command = []
    command += ["python", "train.py"]
    if args.model is not None:
        MODEL = args.model
        command += ["--model_type", MODEL]

    if args.json_file != None:
        JSON_PATH = args.json_file
        params = {}
        with open(JSON_PATH) as f:
            output = json.load(f)
        for k, v in output.items():
            params[k] = v
            #if isinstance(v, list):
             #   params[k] = v
            #else:
             #   params[k] = [v]
    else:
        params = models["common"].copy()  # start with x's keys and values
        params.update(models[MODEL])
        params = merge_two_dicts(models["common"], models[MODEL])
    # execute
    start = time.time()
    lists = {}
    for k,v in params.items():
        if isinstance(v, list):
            lists[k] = v
        else:
            if k == v:
                command += [k]
            else:
                command += [k, str(v)]
    if lists:
        vals = [x for x in list(lists.values())]
        copy = command.copy()
        for comb in itertools.product(*vals):
            cmd = None
            cmd = copy + []
            for k, v in zip(lists.keys(), comb):
                cmd += [k, str(v)]
            cmd += path
            cmd += ["-tr", "1"]
            print("print command: \n", cmd)
            subprocess.call(cmd)
    else:
        print("print command just one: \n", command)
        subprocess.call(command)
    print("time elapsed: ", time.time() - start)

#get_best_comb_from_csv(csv_path, params,
    if args.save:
        csv_path = join(FINAL_PATH, "_output.csv")
        sorted = save_ordered(csv_path)
        get_best_comb_from_csv(csv_path, sorted, params.keys(), save=True)
#TODO:
# return conduct a series of training by reloading the model trained previously
else:
    if args.save:
        now = datetime.datetime.now()
        name = 'train_hyp' + '_' + str(now.month) + '-' + str(now.day) + '-' + str(now.hour) + '-' + str(now.minute)
        PATH = os.path.dirname(__file__)
        print(PATH)
        FINAL_PATH = join(PATH, 'models', name)
        create_dir(join(PATH, 'models', name))
        path = ['-path', FINAL_PATH]

'''
EPOCHS = 5
#BATCH_SIZE = 60 #30
DROPOUT = 0.3
#dropouts = [0.3,0.4,0.5]
batch_size = [60, 70, 80]
LR = 0.001
#lrs = [0.001, 0.01, 0.1]
OPT = 'sgd'
DECAY = 'exp'
#mom = [0.85, 0.90, 0.95]
#decay = ['rate', 'step', 'exp']
#opts = ['rmsprop', 'sgd', 'adam'] #'adam', 'adadelta']  #
reg = ['none','l1', 'l2']
types = ['vgg16','inceptionv3', 'resnet50']

#for t in types:
i = 0
start = time.time()
for b in batch_size:
    subprocess.call(
        ["python", "train.py", "--model_type", "resnet50", "-b", str(b), "-e", str(EPOCHS), "--opt",
         str(OPT),
         "-lr", str(LR), "--decay", str(b), "--dropout", str(DROPOUT), "-n", "3", "-tr", "-path", FINAL_PATH])
    #for m in mom:

    i += 1

print("time elapsed: ", time.time() - start)'''
