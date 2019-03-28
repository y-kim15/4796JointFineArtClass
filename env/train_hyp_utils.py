from os.path import join
import os
from shutil import copyfile
import statistics
import pandas as pd
import json
import pickle
import sys

DIR_PATH = "data"

# make sub training set to fit for CV, n index of sub test set


def make_sub_train(n):
    full_path = join(DIR_PATH, 'wikipaintings_full/wikipaintings_train')
    train_path = join(DIR_PATH, 'train_exp_'+str(n))
    test_path = join(DIR_PATH, 'wiki_small2_'+str(n), 'small_train')
    os.mkdir(train_path)
    styles = os.listdir(full_path)
    for style in styles:
        s_full_works = os.listdir(join(full_path, style))
        s_test_works = os.listdir(join(test_path, style))
        for s_full in s_full_works:
            if s_full not in s_test_works:
                src = join(full_path, style, s_full)
                dest = join(train_path, style, s_full)
                copyfile(src, dest)
    return train_path


def cross_val_score(path, cv, save=True):
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

