import os
import re
import pandas
import csv
from processing.clean_csv import Clean
import imageio
import math
import random
from os.path import join, exists
from random import shuffle
from shutil import copyfile,move, copy, rmtree

N_CLASSES = 25



def count_works(dict):
    # method to count total number of items in dict
    total = 0
    for _, n in dict.items():
        total += n
    return total


def get_n_per_artist(class_dir_path, summary_path):
    # method to return a dictionary with artist name and number of works done by each artist
    # dict key : artist name, value : no of works
    artist_dict = {}
    f = open(summary_path, "a")
    f.write("Size by artist\n")
    for item in os.listdir(class_dir_path):
        artist = item.split('_')[0]
        if artist in artist_dict:
            artist_dict[artist] += 1
        else:
            artist_dict[artist] = 1
    f.write(str(artist_dict) + "\n")
    f.close()
    return artist_dict


def get_sample_size_per_artist(class_dir_path, proportion, summary_path):
    # method to count average number of works per artist per sample
    artist_dict = get_n_per_artist(class_dir_path, summary_path)
    for a, n in artist_dict.copy().items():
        sample_size = math.trunc(n * proportion)
        if sample_size == 0:
            sample_size = 1
        artist_dict[a] = sample_size
    f = open(summary_path, "a")
    f.write("Average Number of works per sample\n")
    f.write(str(artist_dict) + "\n")
    f.close()
    return artist_dict


def get_small_dataset(large_path, proportion, target_path, dir_name, summary_path):
    # method to generate subsets of large dataset (train/test/val) according to given proportion
    # dir_name denote (train/test/val)
    n_dest = int(round(1 / proportion))
    style_dirs = os.listdir(large_path)
    total = 0
    dir_count = [0]*n_dest
    for style in style_dirs:
        f = open(summary_path, "a")
        f.write("Style: " + style)
        for i in range(n_dest):
            os.mkdir(join(target_path + str(i), dir_name, style))
        dir_path = join(large_path, style)
        n = count_files(dir_path)
        f.write(" Size: " + str(n) + "\n")
        total += n
        ave = int(round(n * proportion))
        works = os.listdir(dir_path)
        f.write("Average work size per Sample: " + str(ave) + "\n")
        for i in range(n_dest):
            count = 0
            for x in range(ave):
                index = i * ave + x
                if index >= len(works):
                    break
                copy(join(dir_path, works[index]),
                            join(target_path + str(i), dir_name, style,
                                         os.path.basename(os.path.normpath(works[index]))))
                count += 1
            dir_count[i] += count
    f.write("Total Per Sample Count: \n")
    for item in dir_count:
        f.write("%s\n" % item)
    f.write("Total " + str(total))
    f.close()
    return


def get_small_from_large_dataset(large_path, proportion, target_path, summary_path):
    # method to generate subsets of wikipaintings_full by calling get_small_dataset
    f = open(summary_path, "w+")
    n_dest = int(round(1 / proportion))
    f.write("*********Summary by Directory*********\nProportion: " + str(proportion) + "\nNo dirs: " + str(n_dest) + "\n")
    f.close()
    for i in range(n_dest):
        if exists(target_path + str(i)):
            rmtree(target_path + str(i))
        os.makedirs(target_path + str(i))
    dirs = os.listdir(large_path)
    for d in dirs:
        f = open(summary_path, "a")
        large_split_path = join(large_path, d)
        d_type = d.split('_', 1)[1]
        f.write("split type: " + str(d_type) + "\n")
        f.close()
        for i in range(n_dest):
            dir_name = "small_" + d_type
            to_path = join(target_path + str(i), dir_name)
            Clean.create_dir(to_path)
        get_small_dataset(large_split_path, proportion, target_path, dir_name, summary_path)


def count_files(path):
    n = 0
    for t in list(os.walk(path)):
        n += len(t[2])
    return n


def get_image_matrix(file_path, id, col_name):
    # read train_data_small type file, find the row with matching id
    # get the image and return the matrix form of the image (np format)
    # col_name : col name of where the path exists
    df = pandas.read_csv(file_path)
    row = df.loc[df["id"] == str(id)]
    path = row[col_name]
    im = imageio.imread(path)
    return im


# preprocessing required for wikipaintings dataset
def enumerate_class_names(name, path):
    # given path to the directory, writes all names of class labels to a file.
    # overwrite if exists else create one : "w+"
    f = open(name, "w+")
    dirs = sorted(os.listdir(path))
    i = 0
    for file in dirs:
        f.write(file + "\n")
        i += 1
    f.close()

def get_image_details(file_name):
    splits = file_name.split('_')
    artist = splits[0]
    date = ''
    title = splits[1].replace('.jpg', '')
    sub = title.rsplit("-", 1)
    if len(sub) > 1:
        # if for case xxxx.jpg
        if re.match('^\d{4}$', sub[1]):
            date = sub[1]
            title = sub[0]
        # elif for case where there is xxxx-x.jpg format where year is former
        elif len(sub[0].rsplit("-", 1)) > 1 and re.match('^\d{4}$', sub[0].rsplit("-", 1)[1]):
            sub_sub = sub[0].rsplit("-", 1)
            date = sub_sub[1]
            title = sub_sub[0] + '-' + sub[1]

    return artist, date, title

def generate_image_id_file(name, path, class_file_path, id=True):
    # class_file_path: path where class label file is
    # path: path of parent directory containing all data (e.g. wikipaintings_full)
    # name: path of output file
    # generates file with list of absolute path of images with its class label as csv with code included
    # class label order indexed from 0 determined from class_label.txt file generated
    # print id by default, set as True

    class_f = open(class_file_path, "r")
    lines = class_f.read().split('\n')
    if not id:
        headers = ["agent_display", "date_display", "title_display", "label", "path"]
    else:
        headers = ["id_agent", "id_count", "agent_display", "date_display", "title_display", "label", "class", "path"]
    new_f = pandas.DataFrame(columns=headers)

    upper_dirs = sorted(os.listdir(path))
    for middle in upper_dirs:
        middle_path = join(path, middle)
        dirs = sorted(os.listdir(middle_path))
        i = 0
        for dir in dirs:
            files = sorted(os.listdir(join(middle_path, dir)))
            print("In Directory: ", str(dir))
            for file in files:
                artist, date, title = get_image_details(file)
                file_path = join(middle_path, dir, file)
                if not id:
                    new_f.loc[i] = [artist, date, title, str(lines.index(dir)), file_path]
                else:
                    new_f.loc[i] = ["NaN", "NaN", artist, date, title, str(lines.index(dir)), str(dir), file_path]
                i += 1

    new_f = Clean.assign_id(new_f, "agent_display")
    if not id:
        new_f.to_csv(name, sep=" ", quoting=csv.QUOTE_NONE, escapechar=" ", header=False, index=False)
    else:
        new_f.to_csv(name, header=headers, index=False)


def shuffle_data(train_path, new_path):
    with open(train_path, 'r') as source:
        data = [(random.random(), line) for line in source]
        data.sort()
    with open(new_path, 'w') as target:
        for _, line in data:
            target.write(line)

# creates file system where images are grouped by artist dir
def generate_artist_file_system(path, dest_path):
    if exists(dest_path):
        rmtree(dest_path)
    os.mkdir(dest_path)
    upper_dirs = os.listdir(path)
    for mid_dir in upper_dirs:
        dirs = os.listdir(join(path, mid_dir))

        for dir in dirs:  # for every class dir
            files = os.listdir(join(path, mid_dir, dir))

            for index, item in enumerate(files):  # file in files:
                dest_files = os.listdir(dest_path)  # dest_mid_dir)
                current_item_path = join(path, mid_dir, dir, item)
                new_name = item.split('_')[0]
                if not new_name in dest_files:
                    os.mkdir(join(dest_path, new_name))  # dest_mid_dir, new_name))
                copy(current_item_path,
                            join(dest_path, new_name, item))  # dest_mid_dir, new_name, item))
    print("File system created under data")

DIR_PATH = 'C:/Users/Kira Kim/Documents/cs4796'#os.path.dirname(os.path.realpath(__file__))
# methods applied existing functions from rasta.python.utils.utils

def split_test_training(ratio_test=0.1, name='medium'):
    FULL_PATH = join(DIR_PATH,'data/id_database_' + name)
    TRAIN_PATH = join(DIR_PATH, 'data',  name + '_train')
    TEST_PATH = join(DIR_PATH, 'data',  name + '_test')
    os.mkdir(TRAIN_PATH)
    os.mkdir(TEST_PATH)
    list_full = os.listdir(FULL_PATH)
    n = len(list_full)
    shuffle(list_full)
    split_value = int(n * ratio_test)
    list_test = list_full[:split_value]
    list_train = list_full[split_value:]
    print(len(list_train))
    print(len(list_test))
    for f in list_train:
        SRC_PATH = join(FULL_PATH, f)
        DEST_PATH = join(TRAIN_PATH, f)
        copyfile(SRC_PATH, DEST_PATH)
    for f in list_test:
        SRC_PATH = join(FULL_PATH, f)
        DEST_PATH = join(TEST_PATH, f)
        copyfile(SRC_PATH, DEST_PATH)

def split_val_training(ratio_val=0.1, name='medium'):
    TRAIN_PATH = join(DIR_PATH, 'data',  name + '_train')
    VAL_PATH = join(DIR_PATH, 'data',  name + '_val')
    os.mkdir(VAL_PATH)
    list_train = os.listdir(TRAIN_PATH)
    n = len(list_train)
    shuffle(list_train)
    split_value = int(n * ratio_val)
    list_val = list_train[:split_value]
    print(len(list_val))
    for f in list_val:
        SRC_PATH = join(TRAIN_PATH, f)
        DEST_PATH = join(VAL_PATH, f)
        move(SRC_PATH, DEST_PATH)



if __name__ == '__main__':
    # path = "../../../../../scratch/yk30/wikipaintings_full/wikipaintings_train"
    """path = "../rasta/data/wikipaintings_full/wikipaintings_train"
    class_path = "../data/wikipaintings_class_labels.txt"
    #enumerate_class_names(class_path,path)
    name = "../data/train.txt"
    generate_image_id_file(name, path, class_path, id=False)
    new_train_path = "../data/train_mixed.txt"
    shuffle_data(name, new_train_path)
    generate_image_id_file("../data/val.txt", "../rasta/data/wikipaintings_full/wikipaintings_val", class_path, id=False )
    shuffle_data("../data/val.txt", "../data/val_mixed.txt")"""
    #path = "../data/wikipaintings_full"
    #dest_path = "../data/wiki_small_2_"
    #cur_time = time.strftime("%d%m%y_%H%M")
    #get_small_from_large_dataset(path, 0.2, dest_path, "../summary_"+cur_time+".txt")
    #class_path = "../data/wikipaintings_class_labels.txt"
    #generate_image_id_file("../data/wikipaintings_full_image.csv", "../data/wikipaintings_full", class_path)
    #print(DIR_PATH)
    #split_test_training()
    #split_val_training()
    import numpy as np
    from keras.preprocessing.image import load_img
    DATA_PATH = join(DIR_PATH,'data/medium_train')
    s = np.array([0.,0.,0.])
    t=0
    for file in os.listdir(DATA_PATH):
        x = load_img(join(DATA_PATH, file), target_size=(224, 224))
        s += np.mean(x, axis=(0, 1))
        t += 1

    mean = s/t
    print(mean)
    # path_val = "../../../../../scratch/yk30/wikipaintings_full/wikipaintings_val"
    # val_file_name = "val.txt"
    # generate_image_id_file(val_file_name, path_val, name)
    # path = "./wikipaintings_small"
    # dest_path = "./data/wikipaintings_artist"

    # path = "../../../../scratch/yk30/wikipaintings_full"
    # dest_path = "../../../../scratch/yk30/wikipaintings_full_artist"
    # generate_artist_file_system(path, dest_path)
