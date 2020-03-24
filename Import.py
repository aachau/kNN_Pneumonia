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

    def test(self):
        print(test)


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

    print('FINDING DONE')
    return files

class read_files:
    def __init__(self, files, file_num, read_all):

        print('READING: "test_normal"')
        if read_all == True:
            self.test_normal = [cv2.imread(file_path) for file_path in files.files_test_normal]
        else:
            self.test_normal = [cv2.imread(file_path) for file_path in files.files_test_normal[0:file_num]]
        print('"test_normal" READ SUCCESSFULLY')

        print('READING "test_pneumonia"')
        if read_all == True:
            self.test_pneumonia = [cv2.imread(file_path) for file_path in files.files_test_pneumonia]
        else:
            self.test_pneumonia = [cv2.imread(file_path) for file_path in files.files_test_pneumonia[0:file_num]]
        print('"test_pneumonia" READ SUCCESSFULLY')

        print('READING "train_normal"')
        if read_all == True:
            self.train_normal = [cv2.imread(file_path) for file_path in files.files_train_normal]
        else:
            self.train_normal = [cv2.imread(file_path) for file_path in files.files_train_normal[0:file_num]]
        print('"train_normal" READ SUCCESSFULLY')

        print('READING "train_pneumonia')
        if read_all == True:
            self.train_pneumonia = [cv2.imread(file_path) for file_path in files.files_train_pneumonia]
        else:
            self.train_pneumonia = [cv2.imread(file_path) for file_path in files.files_train_pneumonia[0:file_num]]
        print('"train_pneumonia" READ SUCCESSFULLY')

        print('READING "val_normal')
        if read_all == True:
            self.val_normal = [cv2.imread(file_path) for file_path in files.files_val_normal]
        else:
            self.val_normal = [cv2.imread(file_path) for file_path in files.files_val_normal[0:file_num]]
        print('"val_normal" READ SUCCESSFULLY')

        print('READING "val_pneumonia')
        if read_all == True:
            self.val_pneumonia = [cv2.imread(file_path) for file_path in files.files_val_pneumonia]
        else:
            self.val_pneumonia = [cv2.imread(file_path) for file_path in files.files_val_pneumonia[0:file_num]]
        print('"val_pneumonia" READ SUCCESSFULLY')

        print('READING DONE')

def vectorize(data):
    new_data = data
    for i in np.arange(0, len(new_data.test_normal)):
        new_data.test_normal[i] = data.test_normal[i][:,:,0]
    for i in np.arange(0, len(new_data.test_pneumonia)):
        new_data.test_pneumonia[i] = data.test_pneumonia[i][:,:,0]
    for i in np.arange(0, len(new_data.train_normal)):
        new_data.train_normal[i] = data.train_normal[i][:,:,0]
    for i in np.arange(0, len(new_data.train_pneumonia)):
        new_data.train_pneumonia[i] = data.train_pneumonia[i][:,:,0]
    for i in np.arange(0, len(new_data.val_normal)):
        new_data.val_normal[i] = data.val_normal[i][:,:,0]
    for i in np.arange(0, len(new_data.val_pneumonia)):
        new_data.val_pneumonia[i] = data.val_pneumonia[i][:,:,0]
    return(new_data)

def find_max_shape(data):
    """finds the maximum shape/dimensions of an image in our dataset, then specifies border dimensions to be the next multiple of 1000 of that original shape
        ex. if the max image size is found to be: (2200, 3300) we create a final image size of (3000,4000) on each image after borders are added """
    max_shape = np.empty(2)
    for i in np.arange(0, len(data.test_normal)):
        if max_shape[0] <= data.test_normal[i].shape[0] and max_shape[1] <= data.test_normal[i].shape[1]:
            max_shape = data.test_normal[i].shape
    for i in np.arange(0, len(data.test_pneumonia)):
        if max_shape[0] <= data.test_pneumonia[i].shape[0] and max_shape[1] <= data.test_pneumonia[i].shape[1]:
            max_shape = data.test_pneumonia[i].shape
    for i in np.arange(0, len(data.train_normal)):
        if max_shape[0] <= data.train_normal[i].shape[0] and max_shape[1] <= data.train_normal[i].shape[1]:
            max_shape = data.train_normal[i].shape
    for i in np.arange(0, len(data.train_pneumonia)):
        if max_shape[0] <= data.train_pneumonia[i].shape[0] and max_shape[1] <= data.train_pneumonia[i].shape[1]:
            max_shape = data.train_pneumonia[i].shape
    for i in np.arange(0, len(data.val_normal)):
        if max_shape[0] <= data.val_normal[i].shape[0] and max_shape[1] <= data.val_normal[i].shape[1]:
            max_shape = data.val_normal[i].shape
    for i in np.arange(0, len(data.val_pneumonia)):
        if max_shape[0] <= data.val_pneumonia[i].shape[0] and max_shape[1] <= data.val_pneumonia[i].shape[1]:
            max_shape = data.val_pneumonia[i].shape
    max_shape = np.asarray(max_shape)
    extend_shape = (np.ceil(max_shape/1000)*1000).astype(np.int32)
    return extend_shape



def add_border(data, image_shape):
    "Adds border to image using cv2.copyMakeBorder, and the consistent image dimensions specified by image_shape"
    bordered_data = data
    border_colour = [0, 0, 0]
    for i in np.arange(0, len(data.test_normal)):
        vert_border = np.round((image_shape[0] - data.test_normal[i].shape[0]) / 2).astype(np.int32)
        horz_border = np.round((image_shape[1] - data.test_normal[i].shape[1]) / 2).astype(np.int32)
        bordered_data.test_normal[i] = cv2.copyMakeBorder(data.test_normal[i],
                                       top =  vert_border,
                                       left = horz_border,
                                       right = horz_border,
                                       bottom = vert_border,
                                       borderType = cv2.BORDER_CONSTANT,
                                       value = border_colour)
    for i in np.arange(0, len(data.test_pneumonia)):
        vert_border = np.round((image_shape[0] - data.test_pneumonia[i].shape[0]) / 2).astype(np.int32)
        horz_border = np.round((image_shape[1] - data.test_pneumonia[i].shape[1]) / 2).astype(np.int32)
        bordered_data.test_pneumonia[i] = cv2.copyMakeBorder(data.test_pneumonia[i],
                                       top =  vert_border,
                                       left = horz_border,
                                       right = horz_border,
                                       bottom = vert_border,
                                       borderType = cv2.BORDER_CONSTANT,
                                       value = border_colour)
    for i in np.arange(0, len(data.train_normal)):
        vert_border = np.round((image_shape[0] - data.train_normal[i].shape[0]) / 2).astype(np.int32)
        horz_border = np.round((image_shape[1] - data.train_normal[i].shape[1]) / 2).astype(np.int32)
        bordered_data.train_normal[i] = cv2.copyMakeBorder(data.train_normal[i],
                                       top =  vert_border,
                                       left = horz_border,
                                       right = horz_border,
                                       bottom = vert_border,
                                       borderType = cv2.BORDER_CONSTANT,
                                       value = border_colour)
    for i in np.arange(0, len(data.train_pneumonia)):
        vert_border = np.round((image_shape[0] - data.train_pneumonia[i].shape[0]) / 2).astype(np.int32)
        horz_border = np.round((image_shape[1] - data.train_pneumonia[i].shape[1]) / 2).astype(np.int32)
        bordered_data.train_pneumonia[i] = cv2.copyMakeBorder(data.train_pneumonia[i],
                                       top =  vert_border,
                                       left = horz_border,
                                       right = horz_border,
                                       bottom = vert_border,
                                       borderType = cv2.BORDER_CONSTANT,
                                       value = border_colour)
    for i in np.arange(0, len(data.val_normal)):
        vert_border = np.round((image_shape[0] - data.val_normal[i].shape[0]) / 2).astype(np.int32)
        horz_border = np.round((image_shape[1] - data.val_normal[i].shape[1]) / 2).astype(np.int32)
        bordered_data.val_normal[i] = cv2.copyMakeBorder(data.val_normal[i],
                                       top =  vert_border,
                                       left = horz_border,
                                       right = horz_border,
                                       bottom = vert_border,
                                       borderType = cv2.BORDER_CONSTANT,
                                       value = border_colour)
    for i in np.arange(0, len(data.val_pneumonia)):
        vert_border = np.round((image_shape[0] - data.val_pneumonia[i].shape[0]) / 2).astype(np.int32)
        horz_border = np.round((image_shape[1] - data.val_pneumonia[i].shape[1]) / 2).astype(np.int32)
        bordered_data.val_pneumonia[i] = cv2.copyMakeBorder(data.val_pneumonia[i],
                                       top =  vert_border,
                                       left = horz_border,
                                       right = horz_border,
                                       bottom = vert_border,
                                       borderType = cv2.BORDER_CONSTANT,
                                       value = border_colour)
    return bordered_data
"""======================================================================================================================"""


#specify your file path here
data_directory = 'D:\\University of Calgary Backups\\Fourth Year\\BMEN 509\\Data\\chest_xray'
data_directory_list = os.listdir(data_directory)
data_regular_expression = '*.jpeg'


"""please note, the data structure of data is as follows
indicate the object: data, and then put a: "." after to access the objects lists inside data.
The lists are organized into the image classifications of: test_normal, test_pneumonia... etc.
Index into an image of the list by: data.test_normal[i], and this will give the information for a simple 2D image."""
files_object = find_files(data_directory, data_regular_expression) #use this value to change how many images you read of each category for rapid evaluation
data = read_files(files=files_object, file_num=10, read_all=False)
vectorize_data = vectorize(data)
image_shape = find_max_shape(vectorize_data)
final_data = add_border(vectorize_data, image_shape)

print('placeholder')