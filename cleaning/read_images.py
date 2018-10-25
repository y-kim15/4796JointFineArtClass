import shutil, os
import pandas
from clean_csv import Clean

class ImageDataGenerator:

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

    def generate_image_id_file(name, path, class_file_path):
        # class_file_path: path where class label file is
        # path: path of parent directory containing all data (e.g. wikipaintings_train)
        # name: path of output file
        # generates file with list of absolute path of images with its class label as csv with code included
        # class label order indexed from 0 determined from class_label.txt file generated

        class_f = open(class_file_path, "r")
        lines = class_f.read().split('\n')
        headers = ["id", "path", "label"]
        new_f = pandas.DataFrame(columns = headers)

        dirs = sorted(os.listdir(path))
        i = 0
        for dir in dirs:
            files = sorted(os.listdir(os.path.join(path, dir)))
            for file in files:
                file_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)), "wikipaintings_small", "wikipaintings_train", file)
                new_f.loc[i] = ["NaN", file_path, str(lines.index(dir))]
                i += 1

        new_f = Clean.assign_id(new_f, "label")
        new_f.to_csv(name, header=headers, index=False)

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
    path = "../wikipaintings_small/wikipaintings_train"
    name = "../data/wikipaintings_class_labels_small.txt"
    ImageDataGenerator.enumerate_class_names(name,path)
    data_file_name = "../data/train_data_small.txt"
    ImageDataGenerator.generate_image_id_file(data_file_name, path, name)
    #path_val = "../../../../../scratch/yk30/wikipaintings_full/wikipaintings_val"
    #val_file_name = "val.txt"
    #ImageDataGenerator.generate_load_image_file(val_file_name, path_val, name)
    #path = "./wikipaintings_small"
    #dest_path = "./data/wikipaintings_artist"

    #path = "../../../../scratch/yk30/wikipaintings_full"
    #dest_path = "../../../../scratch/yk30/wikipaintings_full_artist"
    #ImageDataGenerator.generate_artist_file_system(path, dest_path)
