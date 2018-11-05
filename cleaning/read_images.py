import shutil, os
import random
from os.path import dirname, abspath
import pandas
import csv
from clean_csv import Clean
import imageio


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
        #os.mkdir(os.path.join(dest_path, mid_dir))
        #dest_mid_dir = os.path.join(dest_path, mid_dir)
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
    path = "../rasta/data/wikipaintings_full/wikipaintings_train"
    class_path = "../data/wikipaintings_class_labels.txt"
    #enumerate_class_names(class_path,path)
    name = "../data/train.txt"
    generate_image_id_file(name, path, class_path, id=False)
    new_train_path = "../data/train_mixed.txt"
    shuffle_data(name, new_train_path)
    generate_image_id_file("../data/val.txt", "../rasta/data/wikipaintings_full/wikipaintings_val", class_path, id=False )
    shuffle_data("../data/val.txt", "../data/val_mixed.txt")
    #path_val = "../../../../../scratch/yk30/wikipaintings_full/wikipaintings_val"
    #val_file_name = "val.txt"
    #generate_image_id_file(val_file_name, path_val, name)
    #path = "./wikipaintings_small"
    #dest_path = "./data/wikipaintings_artist"

    #path = "../../../../scratch/yk30/wikipaintings_full"
    #dest_path = "../../../../scratch/yk30/wikipaintings_full_artist"
    #generate_artist_file_system(path, dest_path)
