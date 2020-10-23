from os import listdir
from os.path import isfile, join, isdir

import numpy as np
from imageio import imread
from skimage.transform import resize

from run_exec import get_vhf_representation
from utils import cats

EVAL_DATASET_PATH = "/media/federico/Seagate/rgbd-dataset_eval"
PC_DATASET_PATH = "/media/federico/Seagate/rgbd-dataset_pcd/rgbd-dataset"
OUTPUT_DATASET_PATH = "/media/federico/Seagate/new_dataset"


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

    resized_images = []
    vfh_representations = []

    print("{} contains {} angles".format(object_name, len(filenames)))
    for filename in filenames:
        # Calculate the vhf representation
        point_cloud_path = join(PC_DATASET_PATH, category_name, object_name, "{}.pcd".format(filename))
        vhf_rep = get_vhf_representation(point_cloud_path)
        if not vhf_rep:
            continue
        vfh_representations.append(vhf_rep)
        # Load the image
        im = imread(join(object_path, '{}_crop.png'.format(filename)))
        resized = resize(im, (224, 224))
        resized_images.append(resized)

    return np.array(resized_images), np.array(vfh_representations)


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

    object_images = ()
    object_vhf_representations = ()
    object_aggregated_names = ()

    for object_name in objects_names:
        resized_images, vhf_representations = read_object(category_name, object_name)
        object_images += (resized_images,)
        object_vhf_representations += (vhf_representations,)
        object_aggregated_names += ([object_name] * resized_images.shape[0],)

    return np.concatenate(object_images), np.concatenate(object_vhf_representations), \
           np.concatenate(object_aggregated_names)


# execute only if run as a script
if __name__ == "__main__":

    for category_name in cats:
        # Check if file already exist
        try:
            np.load(join(OUTPUT_DATASET_PATH, "{}_instance_names.npy".format(category_name)))
            print("SKIPPED: npy files for {} already exist".format(category_name))
            continue
        except FileNotFoundError:
            pass

        # Load the data(.pcd and .png) to np arrays
        print("Creating npy files for {}".format(category_name))
        category_image_data, category_vhf_data, category_name_data = read_category(category_name)
        print("Total sizes for ", category_name, ": ", category_image_data.shape, category_vhf_data.shape,
              category_name_data.shape)

        # Save the np arrays to npy files
        np.save(join(OUTPUT_DATASET_PATH, "{}_smoothed_images.npy".format(category_name)), category_image_data)
        np.save(join(OUTPUT_DATASET_PATH, "{}_vfh_reps.npy".format(category_name)), category_vhf_data)
        np.save(join(OUTPUT_DATASET_PATH, "{}_instance_names.npy".format(category_name)), category_name_data)
