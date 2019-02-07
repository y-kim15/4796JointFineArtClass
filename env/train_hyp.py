from processing.clean_csv import create_dir
import os
from os.path import join
import datetime
import subprocess
import time

now = datetime.datetime.now()
name = 'train_hyp' + '_' + str(now.month) + '-' + str(now.day) + '-' + str(now.hour) + '-' + str(now.minute)
PATH = os.path.dirname(__file__)
print(PATH)
FINAL_PATH = join(PATH, 'models', name)
create_dir(join(PATH, 'models', name))
#file_name = name + "_vgg16_sum.csv"
#f = open(join(FINAL_PATH, file_name), "w+")
#f.write("opt, lr, dropout, train_loss, train_acc, val_loss, val_acc, time" + "\n")
#f.close()

EPOCHS = 5
BATCH_SIZE = 60
dropouts = [0.3, 0.4, 0.5]
lrs = [0.001, 0.01, 0.1]
opts = ['adam', 'rmsprop']  # 'adadelta', 'sgd']
reg = ['none', 'l1', 'l2']
types = ['vgg16']  #, 'inceptionv3', 'resnet50']

#for t in types:
i = 0
start = time.time()
for d in dropouts:
    for opt in opts:
        for lr in lrs:


            subprocess.call(
                ["python", "train.py", "--model_type", "resnet50", "-b" , str(BATCH_SIZE) ,"-e", str(EPOCHS), "--opt", opt,
                 "-lr",str(lr),"--dropout",str(d),"-n", "3", "-tr", "-path", FINAL_PATH])
            i += 1

print("time elapsed: ", time.time()-start)