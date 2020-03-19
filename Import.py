import os
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import signal, ndimage
import sklearn as sklearn
import cv2
import glob #for pathname finding


class files_class:
    def __init__(self, directory, folder_dir, class_dir, reg_expression):
        self.files_test_normal = glob.glob(os.path.join(directory, folder_dir[1], class_dir[0], reg_expression))
        self.files_test_pneumonia = glob.glob(os.path.join(directory, folder_dir[1], class_dir[1], reg_expression))

        self.files_train_normal = glob.glob(os.path.join(directory, folder_dir[2], class_dir[0], reg_expression))
        self.files_train_pneumonia = glob.glob(os.path.join(directory, folder_dir[2], class_dir[1], reg_expression))

        self.files_val_normal = glob.glob(os.path.join(directory, folder_dir[3], class_dir[0], reg_expression))
        self.files_val_pneumonia = glob.glob(os.path.join(directory, folder_dir[3], class_dir[1], reg_expression))


def find_files(directory, reg_expression):
    '''given directory and regular expression, find all files that match'''
    folder_dir = ['chest_xray', 'test', 'train', 'val', '__MACOSX']
    class_dir = ['NORMAL', 'PNEUMONIA']

    if not os.path.isdir(directory):
        os.sys.exit('Cannot Find Directory, type it correctly')
    if not os.listdir(directory) == folder_dir:
        os.sys.exit('The internal folders do not match, {}. Exiting...'.format(os.listdir(directory)))

    for i in np.arange(1,4):
        if not os.listdir(os.path.join(directory, folder_dir[i])) == class_dir:
            os.sys.exit('You are missing NORMAL or PNEUMONIA dataset, {}. Exiting...'.format(os.listdir(os.path.join(directory, folder_dir[i]))))

    files = files_class(directory, folder_dir, class_dir, reg_expression)
    return files



#specify your file path here
data_directory = 'D:\\University of Calgary Backups\\Fourth Year\\BMEN 509\\Data\\chest_xray'
data_directory_list = os.listdir(data_directory)
data_regular_expression = '*.jpeg' #placeholder

# test = glob.glob('D:/University of Calgary Backups/Fourth Year/BMEN 509/Data/chest_xray/test/NORMAL/*.jpeg')

files = find_files(data_directory, data_regular_expression)



