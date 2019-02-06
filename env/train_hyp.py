from processing.clean_csv import create_dir
import os
from os.path import join
import datetime
import subprocess

now = datetime.datetime.now()
name = 'train_hyp' + '_' + str(now.month) + '-' + str(now.day) + '-' + str(now.hour) + '-' + str(now.minute)
PATH = os.path.dirname(__file__)
FINAL_PATH = join(PATH, 'models', name)
create_dir(join(PATH, 'models', name))
file_name = name+"_sum.csv"
f = open(join(FINAL_PATH, file_name), "w+")
f.write("------------"+name+"_output------------" + "\n")
f.close()

EPOCHS = 5
BATCH_SIZE = 60
dropouts = [0.3, 0.4, 0.5]
lrs = [0.001, 0.01, 0.1]
opts = ['adam', 'rmsprop', 'adadelta', 'sgd']
reg = ['none', 'l1', 'l2']
types = ['vgg16', 'inceptionv3', 'resnet50']

for t in types:
    for d in dropouts:
        for opt in opts:
            for lr in lrs:
                subprocess.call(["python", "train.py", "--model_type", t, "-b", BATCH_SIZE, "-e", EPOCHS, "--opt", opt,
                                 "-lr", lr, "--dropout", d, "-n", 3, "-path", FINAL_PATH, " >> ", file_name])



