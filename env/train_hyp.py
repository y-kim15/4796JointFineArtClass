from processing.clean_csv import create_dir
from processing.train_utils import merge_two_dicts, get_best_comb_from_csv, save_ordered
import os
from os.path import join, exists
import datetime
import subprocess
import time
import argparse
import json
import itertools
from shutil import rmtree

# example hyperparameters to search for each architecture types
# these will be used if user defined json file is not part of script call
# the json file should be same format as this.
models = {
    "common": {
        "--dropout": [0.3, 0.4, 0.5],
        "--opt": ["adam", "sgd"],
        "-lr": [0.001, 0.01, 0.1]
    },
    "resnet50": {
        "--model_type": "resnet50",
        "-e": 5,
        "-b": 60, #[60, 70, 80],
        "-n": 3,
        "-lr": 0.001
    },
    "vgg16": {
        "--model_type": "vgg16",
        "-e": 10,
        "-b": [30, 40],
        "-n": 3,
        "-lr": [0.0001, 0.00001]
    },
    "inceptionv3": {
        "--model_type": "inceptionv3",
        "-e": 5,
        "-b": [60, 70, 80],
        "-n": 3,
        "-lr": [0.001, 0.01, 0.1]
    }
}


PATH = os.path.dirname(__file__)
FINAL_PATH = join(PATH, 'models')

parser = argparse.ArgumentParser(description='Description')

parser.add_argument('-t', action="store", default="params", dest='type', help='Type of Training [params-GridSearch|cv-Cross Validation]')
parser.add_argument('-cv', action="store", default=5, dest='cv', help='Set K for K-fold CV')
parser.add_argument('-j', action="store", default=None, dest='json_file', help='Path to json file containing parameter values')
parser.add_argument('-s', action="store_true",default=False, dest='save', help='Save the output in directory') #if false, will only report the best on cmd line and json
parser.add_argument('--save', action="store", dest='save_path', help='Path to save the directory')
parser.add_argument('-m', action="store", dest='model', help='Name of architecture to use [vgg16|inceptionv3|resnet50]')
args = parser.parse_args()
TYPE = args.type

if args.save:
    now = datetime.datetime.now()
    name = 'train_hyp' + '_' + str(now.month) + '-' + str(now.day) + '-' + str(now.hour) + '-' + str(now.minute)
    PATH = os.path.dirname(__file__)
    print(PATH)
    if args.save_path is not None:
        PATH = args.save_path
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

#### Get parameters ########
if args.json_file is not None:
    JSON_PATH = args.json_file
    params = {}
    with open(JSON_PATH) as f:
        output = json.load(f)
    for k, v in output.items():
        params[k] = v
else:
    params = models["common"].copy()  # start with x's keys and values
    params.update(models[MODEL])
    params = merge_two_dicts(models["common"], models[MODEL])

#### Type of training #######
if TYPE == 'params':
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
        # train for every combination
        for comb in itertools.product(*vals):
            cmd = None
            cmd = copy + []
            for k, v in zip(lists.keys(), comb):
                cmd += [k, str(v)]
            cmd += path
            print("print command: \n", cmd)
            subprocess.call(cmd)
    else:
        print("print command just one: \n", command)
        subprocess.call(command)
    print("time elapsed: ", time.time() - start)

    if args.save:
        csv_path = join(FINAL_PATH, "_output.csv")
        sorted = save_ordered(csv_path)
        get_best_comb_from_csv(csv_path, sorted, params.keys(), save=True)
else:
    # CV training
    val_path = join(PATH, 'data', 'wikipaintings_full', 'wikipaintings_val')#"/cs/tmp/yk30/data/wikipaintings_full/wikipaintings_val"
    csv_path = join(FINAL_PATH, "_cv_fold_" + str(args.cv) + ".csv")
    for i in range(args.cv):
        print("Fold " + str(i+1) + "....")
        train_path = make_sub_train(i)
        train_path = join(PATH, 'train_exp_'+str(i))#'/cs/tmp/yk30/data/train_exp_'+str(i)
        if not exists(train_path):
            train_path = make_sub_train(i)
        test_path = join(PATH, 'data', 'wiki_small_2_' + str(i), 'small_train')
        sub_dir = join(FINAL_PATH, 'train_' + str(i))
        os.mkdir(sub_dir)
        # execute
        start = time.time()
        lists = {}
        command += ['-dp', train_path+'!'+val_path]
        for k, v in params.items():
            if isinstance(v, list):
                command += [k, v[0]]
            else:
                if k == v:
                    command += [k]
                else:
                    command += [k, str(v)]
        command += ['-path', sub_dir]
        print("Training....")
        print("print command: \n", command)
        subprocess.call(command)
        print("time elapsed: ", time.time() - start)

        rmtree(train_path)
        model_path = get_opt_model(sub_dir)

        print("Evaluate....")
        start2 = time.time()
        cmd2 = ['python', 'evaluate_result.py', '-t', 'acc']
        cmd2 += ['-cv', csv_path, '-m', model_path, '-d', test_path]
        print("print command: \n", command)
        subprocess.call(command)
        print("time elapsed: ", time.time() - start2)

    res = cross_val_score(csv_path, cv=args.cv)


##### FUNCTIONS FOR CV TRAINING (extension) ####################################
DIR_PATH = join(PATH, 'data')

# make sub training set to fit for CV, n index of sub test set
# functions used for cv training

def make_sub_train(n):
    # given original full train set and (1/n)th set, creates a new train set
    # which is the full set excluding the (1/n)th set
    full_path = join(DIR_PATH, 'wikipaintings_full/wikipaintings_train')
    train_path = join(DIR_PATH, 'train_exp_'+str(n))
    test_path = join(DIR_PATH, 'wiki_small_2_'+str(n), 'small_train')
    # if already exists ignore
    create_dir(train_path, remove=False)
    styles = os.listdir(full_path)
    for style in styles:
        create_dir(join(train_path, style))
        s_full_works = os.listdir(join(full_path, style))
        s_test_works = os.listdir(join(test_path, style))
        for s_full in s_full_works:
            if s_full not in s_test_works:
                src = join(full_path, style, s_full)
                dest = join(train_path, style, s_full)
                copy(src, dest)
    return train_path


def cross_val_score(path, cv, save=True):
    # computes validation score having computed all folds
    # this is done by examining the output file and save this in json
    data = pd.read_csv(path, encoding='utf-8-sig')
    res = "Result\n"
    combs = {}
    for i in range(3): # 1,3,5
        vals = list(data["Accuracy_"+str(i)])
        dic = {}
        dic['ave'] = round(statistics.mean(vals),4)
        dic['conf'] = round(statistics.stdev(vals)*2,4)
        combs[str(i)] = dic
    res += ''.join('{}{}\n'.format(key, val) for key, val in combs.items())
    print(res)

    if save:
        with open(join(path.rsplit('/', 1)[0], '_output_cv_' + str(cv) + '_fold' + '.json'), 'w') as f:
                json.dump(combs, f)
    return res


def get_opt_model(dir_path):
    # return the file path to the model with the lowest loss value for a specific fold
    # lowest found by examining the history file saved
    try:
        files = os.listdir(dir_path)
        his = pickle.load(open(join(dir_path, '_history.pck'), 'rb'))
        index = his['loss'].index(min(his['loss']))
        f = [x for x in files if x.startswith(str(index)+'-') and x.endswith('.hdf5')]

        if len(f) >= 1:
            return join(dir_path,f[0])
        else:
            raise ValueError()
    except ImportError:
        sys.exit("Error in finding history!")
    except ValueError:
        sys.exit("Error in finding model file!")
