import shutil, os
import time
import pandas
import csv
from cleaning.clean_csv import Clean
import imageio
import math
import glob
import random

N_CLASSES = 25

def create_dir(file_path):
    # method to create directory if doesn't exist, overwrite current if exists
    if os.path.exists(file_path):
        shutil.rmtree(file_path)
    os.makedirs(file_path)

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
            os.mkdir(os.path.join(target_path + str(i), dir_name, style))
        dir_path = os.path.join(large_path, style)
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
                shutil.copy(os.path.join(dir_path, works[index]),
                            os.path.join(target_path + str(i), dir_name, style,
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
        if os.path.exists(target_path + str(i)):
            shutil.rmtree(target_path + str(i))
        os.makedirs(target_path + str(i))
    dirs = os.listdir(large_path)
    for d in dirs:
        f = open(summary_path, "a")
        large_split_path = os.path.join(large_path, d)
        d_type = d.split('_', 1)[1]
        f.write("split type: " + str(d_type) + "\n")
        f.close()
        for i in range(n_dest):
            dir_name = "small" + d_type
            to_path = os.path.join(target_path + str(i), dir_name)
            create_dir(to_path)
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


def generate_image_id_file(name, path, class_file_path, id=True):
    # class_file_path: path where class label file is
    # path: path of parent directory containing all data (e.g. wikipaintings_train)
    # name: path of output file
    # generates file with list of absolute path of images with its class label as csv with code included
    # class label order indexed from 0 determined from class_label.txt file generated
    # print id by default, set as True

    class_f = open(class_file_path, "r")
    lines = class_f.read().split('\n')
    if not id:
        headers = ["path", "label"]
    else:
        headers = ["id", "path", "label", "class_name"]
    new_f = pandas.DataFrame(columns=headers)

    dirs = sorted(os.listdir(path))
    i = 0
    for dir in dirs:
        files = sorted(os.listdir(os.path.join(path, dir)))
        print("In Directory: ", str(dir))
        for file in files:
            file_path = os.path.join("..", "rasta", "data", "wikipaintings_full", "wikipaintings_train", dir, file)
            if not id:
                new_f.loc[i] = [file_path, str(lines.index(dir))]
            else:
                new_f.loc[i] = ["NaN", file_path, str(lines.index(dir)), str(dir)]
            i += 1

    new_f = Clean.assign_id(new_f, "label")
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


def generate_artist_file_system(path, dest_path):
    if os.path.exists(dest_path):
        shutil.rmtree(dest_path)
    os.mkdir(dest_path)
    upper_dirs = os.listdir(path)
    for mid_dir in upper_dirs:
        dirs = os.listdir(os.path.join(path, mid_dir))

        for dir in dirs:  # for every class dir
            files = os.listdir(os.path.join(path, mid_dir, dir))

            for index, item in enumerate(files):  # file in files:
                dest_files = os.listdir(dest_path)  # dest_mid_dir)
                current_item_path = os.path.join(path, mid_dir, dir, item)
                new_name = item.split('_')[0]
                if not new_name in dest_files:
                    os.mkdir(os.path.join(dest_path, new_name))  # dest_mid_dir, new_name))
                shutil.copy(current_item_path,
                            os.path.join(dest_path, new_name, item))  # dest_mid_dir, new_name, item))
    print("File system created under data")


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
    path = "../data/wikipaintings_full"
    dest_path = "../data/wiki_small"
    cur_time = time.strftime("%d%m%y_%H%M")
    get_small_from_large_dataset(path, 0.1, dest_path, "../summary_"+cur_time+".txt")

    # path_val = "../../../../../scratch/yk30/wikipaintings_full/wikipaintings_val"
    # val_file_name = "val.txt"
    # generate_image_id_file(val_file_name, path_val, name)
    # path = "./wikipaintings_small"
    # dest_path = "./data/wikipaintings_artist"

    # path = "../../../../scratch/yk30/wikipaintings_full"
    # dest_path = "../../../../scratch/yk30/wikipaintings_full_artist"
    # generate_artist_file_system(path, dest_path)
