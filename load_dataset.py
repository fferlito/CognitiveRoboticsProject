from utils import cats, mRMR_selected_image_features, max_relevance_first_split
from os.path import join
import numpy as np


def custom_washington_dataset_cv_10fold(names):
    """
    :param names: An np array with the names of all the idxs
    :return: A generator that will give the idxs of the test/train data for 10 folds
    """
    # test_instance_for_all_folds are the instances that should be in the test set for every fold, as specified by the
    # authors of the Washington rgb-d dataset
    test_instances_for_all_folds = [
        ["apple_3", "ball_1", "banana_4", "bell_pepper_6", "binder_3", "bowl_5", "calculator_1", "camera_2", "cap_1", "cell_phone_2", "cereal_box_3", "coffee_mug_2", "comb_1", "dry_battery_3", "flashlight_3", "food_bag_1", "food_box_5", "food_can_11", "food_cup_2", "food_jar_4", "garlic_2", "glue_stick_5", "greens_2", "hand_towel_2", "instant_noodles_3", "keyboard_1", "kleenex_4", "lemon_4", "lightbulb_3", "lime_3", "marker_1", "mushroom_2", "notebook_3", "onion_4", "orange_4", "peach_3", "pear_8", "pitcher_3", "plate_4", "pliers_3", "potato_1", "rubber_eraser_3", "scissors_4", "shampoo_3", "soda_can_2", "sponge_5", "stapler_6", "tomato_8", "toothbrush_3", "toothpaste_5", "water_bottle_2"],
        ["apple_3", "ball_1", "banana_4", "bell_pepper_3", "binder_3", "bowl_6", "calculator_2", "camera_2", "cap_4", "cell_phone_2", "cereal_box_3", "coffee_mug_8", "comb_5", "dry_battery_4", "flashlight_5", "food_bag_2", "food_box_9", "food_can_7", "food_cup_3", "food_jar_4", "garlic_2", "glue_stick_1", "greens_1", "hand_towel_2", "instant_noodles_5", "keyboard_1", "kleenex_4", "lemon_3", "lightbulb_2", "lime_4", "marker_4", "mushroom_1", "notebook_1", "onion_2", "orange_1", "peach_2", "pear_1", "pitcher_2", "plate_6", "pliers_2", "potato_5", "rubber_eraser_4", "scissors_4", "shampoo_4", "soda_can_3", "sponge_11", "stapler_5", "tomato_5", "toothbrush_3", "toothpaste_4", "water_bottle_9"],
        ["apple_2", "ball_4", "banana_4", "bell_pepper_1", "binder_3", "bowl_1", "calculator_4", "camera_2", "cap_4", "cell_phone_3", "cereal_box_5", "coffee_mug_8", "comb_5", "dry_battery_5", "flashlight_1", "food_bag_2", "food_box_5", "food_can_11", "food_cup_5", "food_jar_5", "garlic_2", "glue_stick_4", "greens_2", "hand_towel_4", "instant_noodles_3", "keyboard_5", "kleenex_2", "lemon_2", "lightbulb_3", "lime_1", "marker_9", "mushroom_2", "notebook_2", "onion_4", "orange_4", "peach_2", "pear_2", "pitcher_1", "plate_2", "pliers_3", "potato_5", "rubber_eraser_1", "scissors_1", "shampoo_2", "soda_can_4", "sponge_4", "stapler_7", "tomato_8", "toothbrush_1", "toothpaste_1", "water_bottle_2"],
        ["apple_5", "ball_6", "banana_1", "bell_pepper_2", "binder_1", "bowl_6", "calculator_1", "camera_3", "cap_3", "cell_phone_5", "cereal_box_3", "coffee_mug_7", "comb_2", "dry_battery_6", "flashlight_4", "food_bag_7", "food_box_4", "food_can_8", "food_cup_1", "food_jar_2", "garlic_5", "glue_stick_4", "greens_3", "hand_towel_4", "instant_noodles_4", "keyboard_3", "kleenex_5", "lemon_3", "lightbulb_1", "lime_4", "marker_2", "mushroom_3", "notebook_1", "onion_6", "orange_1", "peach_1", "pear_2", "pitcher_1", "plate_7", "pliers_6", "potato_1", "rubber_eraser_1", "scissors_3", "shampoo_6", "soda_can_6", "sponge_10", "stapler_1", "tomato_5", "toothbrush_1", "toothpaste_1", "water_bottle_3"],
        ["apple_4", "ball_2", "banana_4", "bell_pepper_1", "binder_2", "bowl_5", "calculator_2", "camera_1", "cap_3", "cell_phone_5", "cereal_box_3", "coffee_mug_6", "comb_3", "dry_battery_6", "flashlight_5", "food_bag_6", "food_box_1", "food_can_12", "food_cup_3", "food_jar_4", "garlic_1", "glue_stick_1", "greens_2", "hand_towel_4", "instant_noodles_4", "keyboard_3", "kleenex_3", "lemon_6", "lightbulb_1", "lime_2", "marker_1", "mushroom_3", "notebook_2", "onion_5", "orange_3", "peach_3", "pear_2", "pitcher_1", "plate_1", "pliers_2", "potato_4", "rubber_eraser_2", "scissors_1", "shampoo_4", "soda_can_4", "sponge_7", "stapler_5", "tomato_1", "toothbrush_2", "toothpaste_3", "water_bottle_10"],
        ["apple_3", "ball_3", "banana_3", "bell_pepper_1", "binder_1", "bowl_6", "calculator_3", "camera_3", "cap_4", "cell_phone_1", "cereal_box_3", "coffee_mug_7", "comb_2", "dry_battery_4", "flashlight_3", "food_bag_6", "food_box_8", "food_can_4", "food_cup_1", "food_jar_1", "garlic_4", "glue_stick_2", "greens_2", "hand_towel_5", "instant_noodles_6", "keyboard_1", "kleenex_3", "lemon_3", "lightbulb_3", "lime_4", "marker_3", "mushroom_1", "notebook_2", "onion_3", "orange_2", "peach_2", "pear_3", "pitcher_2", "plate_1", "pliers_2", "potato_6", "rubber_eraser_3", "scissors_3", "shampoo_2", "soda_can_6", "sponge_3", "stapler_8", "tomato_1", "toothbrush_2", "toothpaste_3", "water_bottle_1"],
        ["apple_2", "ball_2", "banana_3", "bell_pepper_3", "binder_1", "bowl_4", "calculator_5", "camera_1", "cap_2", "cell_phone_4", "cereal_box_1", "coffee_mug_3", "comb_3", "dry_battery_5", "flashlight_2", "food_bag_3", "food_box_5", "food_can_9", "food_cup_2", "food_jar_1", "garlic_4", "glue_stick_3", "greens_3", "hand_towel_3", "instant_noodles_1", "keyboard_5", "kleenex_2", "lemon_6", "lightbulb_2", "lime_3", "marker_4", "mushroom_1", "notebook_3", "onion_4", "orange_2", "peach_2", "pear_4", "pitcher_2", "plate_3", "pliers_6", "potato_3", "rubber_eraser_4", "scissors_4", "shampoo_5", "soda_can_2", "sponge_1", "stapler_7", "tomato_2", "toothbrush_4", "toothpaste_5", "water_bottle_1"],
        ["apple_5", "ball_3", "banana_4", "bell_pepper_1", "binder_2", "bowl_2", "calculator_3", "camera_2", "cap_1", "cell_phone_3", "cereal_box_1", "coffee_mug_5", "comb_2", "dry_battery_3", "flashlight_4", "food_bag_8", "food_box_5", "food_can_13", "food_cup_1", "food_jar_4", "garlic_4", "glue_stick_1", "greens_1", "hand_towel_3", "instant_noodles_6", "keyboard_2", "kleenex_5", "lemon_2", "lightbulb_1", "lime_2", "marker_2", "mushroom_2", "notebook_1", "onion_3", "orange_3", "peach_3", "pear_8", "pitcher_3", "plate_7", "pliers_6", "potato_6", "rubber_eraser_3", "scissors_2", "shampoo_3", "soda_can_2", "sponge_5", "stapler_4", "tomato_2", "toothbrush_1", "toothpaste_5", "water_bottle_3"],
        ["apple_3", "ball_1", "banana_2", "bell_pepper_5", "binder_1", "bowl_6", "calculator_4", "camera_3", "cap_4", "cell_phone_3", "cereal_box_5", "coffee_mug_5", "comb_1", "dry_battery_6", "flashlight_3", "food_bag_4", "food_box_6", "food_can_3", "food_cup_3", "food_jar_2", "garlic_5", "glue_stick_1", "greens_1", "hand_towel_3", "instant_noodles_7", "keyboard_1", "kleenex_1", "lemon_2", "lightbulb_4", "lime_2", "marker_5", "mushroom_1", "notebook_4", "onion_6", "orange_3", "peach_2", "pear_4", "pitcher_1", "plate_7", "pliers_2", "potato_4", "rubber_eraser_4", "scissors_3", "shampoo_3", "soda_can_2", "sponge_11", "stapler_3", "tomato_5", "toothbrush_3", "toothpaste_2", "water_bottle_7"],
        ["apple_3", "ball_1", "banana_1", "bell_pepper_4", "binder_2", "bowl_5", "calculator_5", "camera_3", "cap_1", "cell_phone_1", "cereal_box_1", "coffee_mug_7", "comb_3", "dry_battery_4", "flashlight_4", "food_bag_3", "food_box_6", "food_can_4", "food_cup_3", "food_jar_4", "garlic_3", "glue_stick_1", "greens_4", "hand_towel_3", "instant_noodles_4", "keyboard_4", "kleenex_4", "lemon_4", "lightbulb_1", "lime_3", "marker_4", "mushroom_2", "notebook_3", "onion_4", "orange_2", "peach_2", "pear_2", "pitcher_2", "plate_2", "pliers_6", "potato_6", "rubber_eraser_1", "scissors_1", "shampoo_3", "soda_can_2", "sponge_11", "stapler_1", "tomato_4", "toothbrush_5", "toothpaste_2", "water_bottle_5"]
    ]
    print('In total there are: ' + str(len(test_instances_for_all_folds[1])) + 'classes')

    for test_instances_in_current_fold in test_instances_for_all_folds:
        train_idxs = [idx for idx, instance_name in enumerate(names) if instance_name not in test_instances_in_current_fold]
        test_idxs = [idx for idx, instance_name in enumerate(names) if instance_name in test_instances_in_current_fold]
        yield train_idxs, test_idxs


def load_vfh_data(folder_path=None):
    """
    :param folder_path: The path to the folder with .npy files
    :return: X(data), Y(labels), a 10 fold cross_validation generator
    """
    folder_path = folder_path or "./new_dataset"
    X = ()
    Y = []
    all_names = ()

    for label, cat in enumerate(cats):
        vfh_reps = np.load(join(folder_path, "{}_vfh_reps.npy".format(cat)))
        instance_names = np.load(join(folder_path, "{}_instance_names.npy".format(cat)))

        X += (vfh_reps,)
        Y.extend([label] * vfh_reps.shape[0])
        all_names += (instance_names,)

    return np.concatenate(X), np.array(Y), custom_washington_dataset_cv_10fold(np.concatenate(all_names))


def load_good5_data(folder_path=None):
    """
    :param folder_path: The path to the folder with .npy files
    :return: X(data), Y(labels), a 10 fold cross_validation generator
    """
    folder_path = folder_path or "./new_dataset"
    X = ()
    Y = []
    all_names = ()

    for label, cat in enumerate(cats):
        vfh_reps = np.load(join(folder_path, "{}_good5_reps.npy".format(cat)))
        instance_names = np.load(join(folder_path, "{}_instance_names.npy".format(cat)))

        X += (vfh_reps,)
        Y.extend([label] * vfh_reps.shape[0])
        all_names += (instance_names,)
	
    return np.concatenate(X), np.array(Y), custom_washington_dataset_cv_10fold(np.concatenate(all_names))


def load_good15_data(folder_path=None):
    """
    :param folder_path: The path to the folder with .npy files
    :return: X(data), Y(labels), a 10 fold cross_validation generator
    """
    folder_path = folder_path or "./new_dataset"
    X = ()
    Y = []
    all_names = ()

    for label, cat in enumerate(cats):
        vfh_reps = np.load(join(folder_path, "{}_good15_reps.npy".format(cat)))
        instance_names = np.load(join(folder_path, "{}_instance_names.npy".format(cat)))

        X += (vfh_reps,)
        Y.extend([label] * vfh_reps.shape[0])
        all_names += (instance_names,)

    return np.concatenate(X), np.array(Y), custom_washington_dataset_cv_10fold(np.concatenate(all_names))


def load_vfh_and_all_image_feature_data(folder_path=None, use_mRMR=False):
    folder_path = folder_path or "./new_dataset"
    X = ()
    Y = []
    all_names = ()

    for label, cat in enumerate(cats):
        vfh_reps = np.load(join(folder_path, "{}_vfh_reps.npy".format(cat)))
        image_features = np.load(join(folder_path, "{}_smoothed_image_features.npy".format(cat)))
        instance_names = np.load(join(folder_path, "{}_instance_names.npy".format(cat)))

        # When using mRMR choose 512 of the 4096 available image features
        if use_mRMR:
            image_features = image_features[:, max_relevance_first_split]
        X += (np.concatenate((vfh_reps, image_features), axis=1),)
        Y.extend([label] * vfh_reps.shape[0])
        all_names += (instance_names,)

    return np.concatenate(X), np.array(Y), custom_washington_dataset_cv_10fold(np.concatenate(all_names))


def load_all_image_feature_data(folder_path=None, use_mRMR=False):
    folder_path = folder_path or "./new_dataset"
    X = ()
    Y = []
    all_names = ()

    for label, cat in enumerate(cats):
        image_features = np.load(join(folder_path, "{}_smoothed_image_features.npy".format(cat)))
        instance_names = np.load(join(folder_path, "{}_instance_names.npy".format(cat)))

        # When using mRMR choose 512 of the 4096 available image features
        if use_mRMR:
            image_features = image_features[:, max_relevance_first_split]
        X += (image_features,)
        Y.extend([label] * image_features.shape[0])
        all_names += (instance_names,)

    return np.concatenate(X), np.array(Y), custom_washington_dataset_cv_10fold(np.concatenate(all_names))


def load_simple_vfh_dataset(folder_path=None):
    """
    :param folder_path: The path to the folder with .npy files
    :return: x_train, y_train, x_test, y_test
    """
    folder_path = folder_path or "./new_dataset"
    X_train = ()
    Y_train = []
    X_test = ()
    Y_test = []

    for label, cat in enumerate(cats):
        vfh_reps = np.load(join(folder_path, "{}_vfh_reps.npy".format(cat)))
        instance_names = np.load(join(folder_path, "{}_instance_names.npy".format(cat)))

        # Use all but the first instances as training data
        idxs_of_other_instances = np.where(instance_names != instance_names[0])
        train_vfh_reps = vfh_reps[idxs_of_other_instances]
        X_train += (train_vfh_reps,)
        Y_train.extend([label] * train_vfh_reps.shape[0])

        # Use first instance as test data
        idxs_of_first_instance = np.where(instance_names == instance_names[0])
        test_vfh_reps = vfh_reps[idxs_of_first_instance]
        X_test += (test_vfh_reps,)
        Y_test.extend([label] * test_vfh_reps.shape[0])

    return np.concatenate(X_train), np.array(Y_train), np.concatenate(X_test), np.array(Y_test)
