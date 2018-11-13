import shutil, os
import random
from os.path import dirname, abspath
import pandas
import csv
from clean_csv import Clean
import imageio
import pprint
import math
import glob
N_CLASSES = 25

def count_works(dict):
    total = 0
    for _, n in dict.items():
        total += n
    return total

def get_n_per_artist(class_dir_path):
    artist_dict = {}
    for item in os.listdir(class_dir_path):
        artist = item.split('_')[0]
        if artist in artist_dict:
            artist_dict[artist] += 1
        else:
            artist_dict[artist] = 1
    return artist_dict

def get_sample_n_per_artist(f, dir_name, class_dir_path, total_n_work, proportion):
    # method to return the list with number of works to get from each artist
    actual = math.trunc(total_n_work*proportion)
    artist_dict = get_n_per_artist(class_dir_path)
    f = open(f, "a")
    f.write("\n=========="+dir_name+"============\n")
    f.write("Total: "+str(count_works(artist_dict))+"\n")
    f.write("Proposed: "+str(actual)+"\n")
    f.write("Number of Artists:"+str(len(artist_dict))+"\n")
    for a, n in artist_dict.copy().items():
        if actual < len(artist_dict):
            del artist_dict[a]
        elif actual == len(artist_dict):
            artist_dict[a] = 1
    ave = math.trunc(actual/len(artist_dict))
    if actual > len(artist_dict):
        large = {} # any artist with n greater than ave
        short = 0
        for a, n in artist_dict.items():
            if n < ave:
                short += (ave-n)
            elif n > ave:
                large[a] = (n-ave)
                artist_dict[a] = ave
        if ave == 1:
            short = actual - len(artist_dict)
        ave_short = math.trunc(short/len(large))
        for a, n in large.copy().items():
            if a not in large:
                continue
            elif short > 0:
                ave_short = math.trunc(short/len(large))
                if n >= ave_short:
                    artist_dict[a] += ave_short
                    del large[a]
                    short -= ave_short
                else:
                    artist_dict[a] += n
                    del large[a]
                    short -= n
    f.write("Final Total: "+str(count_works(artist_dict))+"\n")
    f.close()
    return artist_dict


def get_small_dataset_from_large_by_artist(large_path, proportion, dest_path, summary_path):
    # example of large path would be wikipaintings_train/test/val
    # example of dest path would be wiki_small_train/test/val
    final_total = 0
    if os.path.exists(dest_path):
        shutil.rmtree(dest_path)
    os.makedirs(dest_path)
    dirs = os.listdir(large_path)
    f = open(summary_path, "w+")
    f.write("*********Summary by Directory*********\nProportion: "+str(proportion)+"\n")
    f.close()
    for dir in dirs:
        os.mkdir(os.path.join(dest_path, dir))
        dir_path = os.path.join(large_path, dir)
        dirs = os.listdir(dir_path)
        n = 0
        for t in list(os.walk(dir_path)):
            n += len(t[2])
        artist_dict = get_sample_n_per_artist(summary_path, dir, dir_path, n, proportion)
        for a, n in artist_dict.items():
            name = a+"_*"
            count = n
            current_item_path = os.path.join(dir_path, name)
            for work in glob.iglob(os.path.join(dir_path, name)):
                if count == 0:
                    break
                else:
                    shutil.copy(work, os.path.join(dest_path, dir, os.path.basename(os.path.normpath(work))))
                    count -= 1
                    final_total += 1
        f = open(summary_path, "a")
        f.write("\n******Final Total******"+str(final_total)+"\n")
        f.close()

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
        f.write(file+"\n")
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
    if not id :
        headers = ["path", "label"]
    else:
        headers = ["id", "path", "label", "class_name"]
    new_f = pandas.DataFrame(columns = headers)

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
    with open(train_path,'r') as source:
        data = [ (random.random(), line) for line in source ]
        data.sort()
    with open(new_path,'w') as target:
        for _, line in data:
            target.write( line )

def generate_artist_file_system(path, dest_path):
    if os.path.exists(dest_path):
        shutil.rmtree(dest_path)
    os.mkdir(dest_path)
    upper_dirs = os.listdir(path)
    for mid_dir in upper_dirs:
        dirs = os.listdir(os.path.join(path, mid_dir))

        for dir in dirs: # for every class dir
            files = os.listdir(os.path.join(path, mid_dir, dir))

            for index, item in enumerate(files): # file in files:
                dest_files = os.listdir(dest_path)#dest_mid_dir)
                current_item_path = os.path.join(path, mid_dir, dir, item)
                new_name = item.split('_')[0]
                if not new_name in dest_files:
                    os.mkdir(os.path.join(dest_path, new_name))#dest_mid_dir, new_name))
                shutil.copy(current_item_path, os.path.join(dest_path, new_name, item))#dest_mid_dir, new_name, item))
    print("File system created under data")


if __name__ == '__main__':
    #path = "../../../../../scratch/yk30/wikipaintings_full/wikipaintings_train"
    """path = "../rasta/data/wikipaintings_full/wikipaintings_train"
    class_path = "../data/wikipaintings_class_labels.txt"
    #enumerate_class_names(class_path,path)
    name = "../data/train.txt"
    generate_image_id_file(name, path, class_path, id=False)
    new_train_path = "../data/train_mixed.txt"
    shuffle_data(name, new_train_path)
    generate_image_id_file("../data/val.txt", "../rasta/data/wikipaintings_full/wikipaintings_val", class_path, id=False )
    shuffle_data("../data/val.txt", "../data/val_mixed.txt")"""
    path = "../rasta/data/wikipaintings_full/wikipaintings_train"
    dest_path = "../rasta/data/wiki_small/wiki_train"
    get_small_dataset_from_large_by_artist(path, 0.05, dest_path, "../summary_train.txt")

    #path_val = "../../../../../scratch/yk30/wikipaintings_full/wikipaintings_val"
    #val_file_name = "val.txt"
    #generate_image_id_file(val_file_name, path_val, name)
    #path = "./wikipaintings_small"
    #dest_path = "./data/wikipaintings_artist"

    #path = "../../../../scratch/yk30/wikipaintings_full"
    #dest_path = "../../../../scratch/yk30/wikipaintings_full_artist"
    #generate_artist_file_system(path, dest_path)
