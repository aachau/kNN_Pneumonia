import os
import pandas as pd
import numpy as np
from numpy import mean
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
from skimage import io
#Test with all train data from original folders
image_directory = r'C:\Users\HP\Documents\University\BMEN 509\Final Project\chest-xray-pneumonia\chest_xray\train'
xray_types = os.listdir(image_directory) #'BACTERIAL, 'NORMAL','PNEUMONIA' in that order

bacterial=[]
normal=[]
viral=[]
type_set=[bacterial, normal ,viral]
N_images=10 #Number of images from each case. Set equal to len(image_list) for all

for i in range (0,len(xray_types)): #outer loop to go through each folder differentiating xray types
    type_directory = os.path.join(image_directory, xray_types[i]) #joins directory path
    image_list = os.listdir(type_directory) #list of images in each folder
    if 'desktop.ini' in image_list: image_list.remove('desktop.ini') #Removes this hidden file
    os.chdir(type_directory) #enter the directory for a type of xray
    image_set=[]
#Start image reading loop
    for j in range (0,N_images):
        image = io.imread(image_list[j]) #Reading files in the folder. Make sure files are not open.
        type_set[i].append(image)

#Calculating std of histograms
bact_hist_std=[]
norm_hist_std=[]
viral_hist_std=[]
hist_std_set=[bact_hist_std, norm_hist_std ,viral_hist_std]

#Loops to get fractions
for i in range (0, len(hist_std_set)): #i is the class we are in
    for j in range (0, N_images): #j is the image within the class set
        image = type_set[i][j]          #Accessing the array (j) in the class list (i) i.e normal) within the overall list (type_set)
        hist,bins = np.histogram(image.flatten(),256,[0,256]) #Histrogram is a 256x1 array with value of each index as number of pixels with that intensity
        # hist=hist[0:256] #Optional to cut out bins
        std=round(np.std(hist),2)
        hist_std_set[i].append(std)


# Calculating std of images
B_std=[]
N_std=[]
V_std=[]
for j in range(0, len(bacterial)):
    std = np.std(bacterial[j])
    B_std.append(std)

for j in range(0, len(normal)):
    std = np.std(normal[j])
    N_std.append(std)

for j in range(0, len(viral)):
    std = np.std(viral[j])
    V_std.append(std)

all_hist_std = bact_hist_std + norm_hist_std# + viral_hist_std
all_std = B_std + N_std# + V_std

#Start Kmeans
clust = KMeans(init='k-means++', n_clusters=2, n_init=10, tol=0.0001 ) #n_init = number of times algorithm is run and the best inertia is the one chosen
stacked_data = np.column_stack((all_hist_std, all_std)) #Turning lists into ordered pairs
clust.fit(stacked_data) #Fitting Kmeans with our data
centroids = clust.cluster_centers_

labels = np.empty([2*N_images])     #2*N_images if only 2 classes, 3*N_images if 3
labels[0:N_images] = 1              #Write in terms of N_images
labels[N_images:2*N_images] = 0       #Write in terms of N_images
#labels[120:180] = 2
label_trifecta = np.array(('bacterial', 'normal'))

#Plotting predicted
plt.figure(figsize=(8,8))

plt.subplot(211)
plt.scatter(stacked_data[:, 0], stacked_data[:, 1], c=clust.labels_,cmap=plt.cm.get_cmap('Spectral', 2), alpha=0.7)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x')

for i in range (centroids.shape[0]):
    xy=(centroids[i, 0],centroids[i, 1])
    plt.annotate(label_trifecta[i],xy, horizontalalignment='right', verticalalignment='top')

plt.title('K-means clustering\n'
          'Centroids are marked as x (k=2)\n'
          '%i points per class' % N_images)
plt.xlabel('histogram std')
plt.ylabel('std')
#Plotting Actual
plt.subplot(212)
plt.scatter(stacked_data[:, 0], stacked_data[:, 1], c=labels, cmap=plt.cm.get_cmap('Spectral', 2), alpha=0.7)

plt.title('Actual Data\n' '%i points per class' % N_images)
plt.xlabel('histogram std')
plt.ylabel('std')
plt.show()

H=metrics.homogeneity_score(labels,clust.labels_.astype(float)) #The actual numbers for the labels do not matter, they just represent a class
C=metrics.completeness_score(labels,clust.labels_.astype(float))
F=metrics.fowlkes_mallows_score(labels,clust.labels_.astype(float))

print('Homogeneity =',H)
print('Completeness =',C)
print('FM Score =',F)

print('DONE')