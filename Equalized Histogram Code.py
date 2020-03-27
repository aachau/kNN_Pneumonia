#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Orignal Histograms
histograms=[] #list of original histograms
for i in range (0,len(train_data)):
    Image=train_data[i]
    hist,bins = np.histogram(Image.flatten(),np.max(Image),[0,np.max(Image)])
    histograms.append(hist) #Need to append to an empty list

#Manipulated Histograms
print('Start of Equalized Histograms')
mani_histograms=[] #Equalized histograms
mani_images=[] #Equalized Images
    
for i in range (0,len(train_data)):
    Im=train_data[i]
    hist,bins = np.histogram(Im.flatten(),np.max(Im)+1,[0,np.max(Im)])
    cdf = hist.cumsum()   
    cdf = (cdf - cdf.min())*np.max(Im)/(cdf.max()-cdf.min())         
    cdf = np.ma.filled(cdf,0).astype('uint16')                
    Im2 = cdf[Im]
    mani_images.append(Im2) #List of new images
    mani_hist,mani_bins = np.histogram(Im2.flatten(),np.max(Im2),[0,np.max(Im2)])
    mani_histograms.append(mani_hist)
    

