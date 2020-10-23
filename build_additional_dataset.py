import time
from os import listdir
from os.path import isfile, join, isdir

import numpy as np

from run_exec import get_vhf_representation, get_good5_representation, get_good15_representation
from utils import cats

EVAL_DATASET_PATH = "/media/federico/Seagate/rgbd-dataset_eval"
PC_DATASET_PATH = "/media/federico/Seagate/rgbd-dataset_pcd/rgbd-dataset"
OUTPUT_DATASET_PATH = "/media/federico/Seagate/additional_dataset"


def read_object(category_name, object_name):
    """
    :param category_name: The name of the category which is also the directory name e.g. apple, water_bottle
    :param object_name: The name of the object which is also a directory name e.g. apple_1, apple_2, apple_3
    :return: An np array with images of shape (n, 224, 224, 3) and an np array with vfh representations of shape
    (n, 308) where n is the amount of images taken of the instance with the name object name
    """
    object_path = join(EVAL_DATASET_PATH, category_name, object_name)
    # Get the filenames from the txt files
    filenames = [f[:-8] for f in listdir(object_path)
                 if isfile(join(object_path, f)) and f[-8:] == "_loc.txt"]

    vfh_representations = []
    good5_representations = []
    good15_representations = []

    print("{} contains {} angles".format(object_name, len(filenames)))
    for filename in filenames:
        # Calculate the vhf representation
        point_cloud_path = join(PC_DATASET_PATH, category_name, object_name, "{}.pcd".format(filename))
        vhf_rep = get_vhf_representation(point_cloud_path)
        good5_rep = get_good5_representation(point_cloud_path)
        good15_rep = get_good15_representation(point_cloud_path)
        if not vhf_rep:
            continue
        if not good5_rep or not good15_rep:
            print("HOUSTON WE HAVE A FUCKING PROBLEM")
        vfh_representations.append(vhf_rep)
        good5_representations.append(good5_rep)
        good15_representations.append(good15_rep)

    return np.array(vfh_representations), np.array(good5_representations), np.array(good15_representations)


def read_category(category_name):
    """
    :param category_name: The name of the category which is also the directory name e.g. apple, water_bottle
    :return: the function return a np array with images with form (n, 224, 224, 3), np array with vhf representation
    with shape (n, 308) and np array with instances names like ['apple_1', 'apple_1', ..., 'apple_5] with shape (n,)
    in which n is the amount of pictures taken of apples
    """
    category_path = join(EVAL_DATASET_PATH, category_name)
    objects_names = [d for d in listdir(category_path) if isdir(join(category_path, d))]
    print("Found {} instances of {}".format(len(objects_names), category_name))

    object_vfh_representations = ()
    object_good5_representations = ()
    object_good15_representations = ()

    for object_name in objects_names:
        vfh_representations, good5_representations, good15_representations = read_object(category_name, object_name)
        object_vfh_representations += (vfh_representations,)
        object_good5_representations += (good5_representations,)
        object_good15_representations += (good15_representations,)

    return np.concatenate(object_vfh_representations), np.concatenate(object_good5_representations), \
           np.concatenate(object_good15_representations)


# execute only if run as a script
if __name__ == "__main__":

    for category_name in reversed(cats):
        start_time = time.time()
        # Check if file already exist
        try:
            np.load(join(OUTPUT_DATASET_PATH, "{}_good5_reps.npy".format(category_name)))
            print("SKIPPED: npy files for {} already exist".format(category_name))
            continue
        except FileNotFoundError:
            pass

        # Load the data(.pcd and .png) to np arrays
        print("Creating npy files for {}".format(category_name))
        category_vfh_data, category_good5_data, category_good15_data = read_category(category_name)
        print("Total sizes for ", category_name, ": ", category_vfh_data.shape, category_good5_data.shape,
              category_good15_data.shape)

        # Save the np arrays to npy files
        np.save(join(OUTPUT_DATASET_PATH, "{}_vfh_reps.npy".format(category_name)), category_vfh_data)
        np.save(join(OUTPUT_DATASET_PATH, "{}_good5_reps.npy".format(category_name)), category_good5_data)
        np.save(join(OUTPUT_DATASET_PATH, "{}_good15_reps.npy".format(category_name)), category_good15_data)
        total_time = time.time() - start_time
        print("Took {} minutes and {} seconds".format(total_time // 60, total_time % 60))
