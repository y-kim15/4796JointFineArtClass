import os

class ImageDataGenerator:
    # preprocessing required for wikipaintings dataset

    def enumerate_class_names(name, path):
        # given path to the directory, writes all names of class labels to a file.
        # overwrite if exists else create one : "w+"
        f = open(name, "w+")
        dirs = os.listdir(path)
        i = 0
        for file in dirs:
            f.write(file+"\n")
            i += 1
        f.close()

    def generate_load_image_file(name, path, class_file):
        # generates file with list of absolute path of images with its class label
        # class label order indexed from 0 determined from class_label.txt file generated
        class_f = open(class_file, "r")
        lines = class_f.read().split('\n')

        f = open(name, "w+")
        dirs = os.listdir(path)
        for dir in dirs:
            files = os.listdir(os.path.join(path, dir))
            for file in files:
                f.write(os.path.abspath(file)+" "+str(lines.index(dir))+"\n")
        f.close()

    def generate_artist_file_system(path):
        

if __name__ == '__main__':
    path = "../../../../../scratch/yk30/wikipaintings_full/wikipaintings_train"
    name = "wikipaintings_class_labels.txt"
    ImageDataGenerator.enumerate_class_names(name,path)
    data_file_name = "train.txt"
    ImageDataGenerator.generate_load_image_file(data_file_name, path, name)
    path_val = "../../../../../scratch/yk30/wikipaintings_full/wikipaintings_val"
    val_file_name = "val.txt"
    ImageDataGenerator.generate_load_image_file(val_file_name, path_val, name)
