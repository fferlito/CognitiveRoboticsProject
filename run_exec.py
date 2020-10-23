import subprocess


def get_representation(file_path, descriptor_type):
    """
    This function calls a C++ executable to calculate a VHF representation for a given pointcloud filename and parses
    its output to a python list.
    :param file_path: File path to a .pcd file
    :param descriptor_type: integer that will be past to the executable VFH = 0, GOOD_5 = 1, GOOD_15 = 2
    :return: The 308 numbers long list representing the VHF histogram of the pointcloud
    """
    args = ("./CPP/build/project", file_path, str(descriptor_type))
    popen = subprocess.Popen(args, stdout=subprocess.PIPE)
    popen.wait()
    output = popen.stdout.read().decode("utf-8")
    try:
        return eval(output)
    except SyntaxError:
        return None


def get_vhf_representation(file_path):
    return get_representation(file_path, 0)


def get_good5_representation(file_path):
    return get_representation(file_path, 1)


def get_good15_representation(file_path):
    return get_representation(file_path, 2)
