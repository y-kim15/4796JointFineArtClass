import shutil, os

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
    #name = "wikipaintings_class_labels.txt"
    #ImageDataGenerator.enumerate_class_names(name,path)
    #data_file_name = "train.txt"
    #ImageDataGenerator.generate_load_image_file(data_file_name, path, name)
    #path_val = "../../../../../scratch/yk30/wikipaintings_full/wikipaintings_val"
    #val_file_name = "val.txt"
    #ImageDataGenerator.generate_load_image_file(val_file_name, path_val, name)
    #path = "./wikipaintings_small"
    #dest_path = "./data/wikipaintings_artist"

    path = "../../../../scratch/yk30/wikipaintings_full"
    dest_path = "../../../../scratch/yk30/wikipaintings_full_artist"
    ImageDataGenerator.generate_artist_file_system(path, dest_path)
