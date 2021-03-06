import os
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import signal, ndimage
import sklearn as sklearn
import cv2
import glob #for pathname finding
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn import metrics


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

def define_border(border_val, data_set, image_shape, i):
    vert_border = (image_shape[0] - data_set[i].shape[0]) / 2
    horz_border = (image_shape[1] - data_set[i].shape[1]) / 2
    if vert_border % 2 == 0:
        border_val[0] = vert_border
        border_val[3] = vert_border
    else:
        border_val[0] = np.ceil(vert_border)
        border_val[3] = np.floor(vert_border)
    if horz_border % 2 == 0:
        border_val[1] = horz_border
        border_val[2] = horz_border
    else:
        border_val[1] = np.ceil(horz_border)
        border_val[2] = np.floor(horz_border)
    border_val = border_val.astype(np.int32)
    return border_val

def add_border(data, image_shape):
    "Adds border to image using cv2.copyMakeBorder, and the consistent image dimensions specified by image_shape"
    bordered_data = data
    border_val = np.empty(4) #top, left, right, bottom
    border_colour = [0, 0, 0]
    for i in np.arange(0, len(data.test_normal)):
        border_val = define_border(border_val, data.test_normal, image_shape, i)
        bordered_data.test_normal[i] = cv2.copyMakeBorder(data.test_normal[i],
                                       top =  border_val[0],
                                       left = border_val[1],
                                       right = border_val[2],
                                       bottom = border_val[3],
                                       borderType = cv2.BORDER_CONSTANT,
                                       value = border_colour)
    for i in np.arange(0, len(data.test_pneumonia)):
        border_val = define_border(border_val, data.test_pneumonia, image_shape, i)
        bordered_data.test_pneumonia[i] = cv2.copyMakeBorder(data.test_pneumonia[i],
                                       top =  border_val[0],
                                       left = border_val[1],
                                       right = border_val[2],
                                       bottom = border_val[3],
                                       borderType = cv2.BORDER_CONSTANT,
                                       value = border_colour)
    for i in np.arange(0, len(data.train_normal)):
        border_val = define_border(border_val, data.train_normal, image_shape, i)
        bordered_data.train_normal[i] = cv2.copyMakeBorder(data.train_normal[i],
                                       top =  border_val[0],
                                       left = border_val[1],
                                       right = border_val[2],
                                       bottom = border_val[3],
                                       borderType = cv2.BORDER_CONSTANT,
                                       value = border_colour)
    for i in np.arange(0, len(data.train_pneumonia)):
        border_val = define_border(border_val, data.train_pneumonia, image_shape, i)
        bordered_data.train_pneumonia[i] = cv2.copyMakeBorder(data.train_pneumonia[i],
                                       top =  border_val[0],
                                       left = border_val[1],
                                       right = border_val[2],
                                       bottom = border_val[3],
                                       borderType = cv2.BORDER_CONSTANT,
                                       value = border_colour)
    for i in np.arange(0, len(data.val_normal)):
        border_val = define_border(border_val, data.val_normal, image_shape, i)
        bordered_data.val_normal[i] = cv2.copyMakeBorder(data.val_normal[i],
                                       top =  border_val[0],
                                       left = border_val[1],
                                       right = border_val[2],
                                       bottom = border_val[3],
                                       borderType = cv2.BORDER_CONSTANT,
                                       value = border_colour)
    for i in np.arange(0, len(data.val_pneumonia)):
        border_val = define_border(border_val, data.val_pneumonia, image_shape, i)
        bordered_data.val_pneumonia[i] = cv2.copyMakeBorder(data.val_pneumonia[i],
                                       top =  border_val[0],
                                       left = border_val[1],
                                       right = border_val[2],
                                       bottom = border_val[3],
                                       borderType = cv2.BORDER_CONSTANT,
                                       value = border_colour)
    return bordered_data


def vector2(data):
    """from jupyter"""
    data1 = data
    new_data = data1
    for i in np.arange(0, len(new_data.test_normal)):
        rows, cols = data.test_normal[i].shape
        new_data.test_normal[i] = data.test_normal[i].reshape(rows * cols)

    for i in np.arange(0, len(new_data.test_pneumonia)):
        rows, cols = data.test_pneumonia[i].shape
        new_data.test_pneumonia[i] = data.test_pneumonia[i].reshape(rows * cols)

    for i in np.arange(0, len(new_data.train_normal)):
        rows, cols = data.train_normal[i].shape
        new_data.train_normal[i] = data.train_normal[i].reshape(rows * cols)

    for i in np.arange(0, len(new_data.train_pneumonia)):
        rows, cols = data.train_pneumonia[i].shape
        new_data.train_pneumonia[i] = data.train_pneumonia[i].reshape(rows * cols)

    for i in np.arange(0, len(new_data.val_normal)):
        rows, cols = data.val_normal[i].shape
        new_data.val_normal[i] = data.val_normal[i].reshape(rows * cols)

    for i in np.arange(0, len(new_data.val_pneumonia)):
        rows, cols = data.val_pneumonia[i].shape
        new_data.val_pneumonia[i] = data.val_pneumonia[i].reshape(rows * cols)

    return (new_data)

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
data = read_files(files=files_object, file_num=10, read_all=False) #use file_num to indicate the number of images in each class to load, or use "read_all = True" to read everything)
vectorize_data = vectorize(data)
image_shape = find_max_shape(vectorize_data)
final_data = add_border(vectorize_data, image_shape)


x_data=vector2(final_data)

test_normal=np.asarray(x_data.test_normal, dtype=np.uint8)
test_virus=np.asarray(x_data.test_pneumonia, dtype=np.uint8)

train_normal=np.asarray(x_data.train_normal, dtype=np.uint8)
train_virus=np.asarray(x_data.train_pneumonia, dtype=np.uint8)

train_data=np.concatenate((train_normal, train_virus), axis=0)

print('PCA BEGINNING...')
PCA_data = PCA(n_components=2).fit_transform(train_data)
print('PCA FINISHED')
k=2
clust=KMeans(n_clusters=k, n_init=10, tol=0.0001)
clust.fit(PCA_data)
centroids = clust.cluster_centers_
c=clust.labels_.astype(float)


labels = np.empty([200])
labels[0:100] = 1
labels[101:200] = 0
label_binary = np.array(('normal','viral'))

plt.figure(figsize=(9,5))
x=PCA_data[:,0]
y=PCA_data[:,1]
plt.scatter(x,y, c=clust.labels_, cmap=plt.cm.get_cmap('Spectral', 10))
plt.colorbar()
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.title('Pneumonia Classification Dataset KMeans Clustering')
plt.scatter(centroids[:,0],centroids[:,1],marker='x')
for i in range (centroids.shape[0]):
    xy=(centroids[i, 0],centroids[i, 1])
    plt.annotate(label_binary[i],xy, horizontalalignment='right', verticalalignment='top')
plt.show()

H=metrics.homogeneity_score(labels,clust.labels_.astype(float))  #Homogeneity. It compares your true output classes to how those outputs are arranged throughout the clusters. So clusters = 10, but only 3 true classes.
C=metrics.completeness_score(labels,clust.labels_.astype(float))
F=metrics.fowlkes_mallows_score(labels,clust.labels_.astype(float))

print('Homogeneity =',H)
print('Completeness =',C)
print('FM Score =',F)
print('Inertia=',inertia)

print('placeholder')