# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 17:26:09 2019

@author: Stephan
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib

from tqdm import tqdm_notebook
from PIL import Image

from keras.preprocessing.image import ImageDataGenerator
from imgaug import augmenters as iaa

aug = iaa.SomeOf((1,1), [
    iaa.GaussianBlur(sigma = (0, 2.0)),
    iaa.Noop(),
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    iaa.Multiply((0.5, 1.5), per_channel=0.5)
])

def augmentor(image, *args, **kwargs):
    image = aug.augment_image(image)
    return image



def show_augmentations(datagen, x_train, y_train):
    batches = datagen.flow(x_train[0:1], y_train[0:1], batch_size=10)
    super_x_batch = []
    for i in range(10):
        x_batch, y_batch = next(batches)
        super_x_batch.append(x_batch[0])
    
    # plot the results
    matplotlib.rcParams['figure.figsize'] = (15,7)
    fig, axes = plt.subplots(2, 5)
    for i in range(2):
        for j in range(5):
            axes[i, j].imshow(super_x_batch[(i*5)+j], aspect='auto')
    plt.show()



if __name__ == '__main__':
    '''
    Shows an example of the augmentations
    
    '''
    
    #load training data ids
    df_data = pd.read_csv('../data/train_labels.csv')
    df_data['id'] = df_data['id'].astype(str) + '.tif'
            
    # Load data into memory, take just 1 image so the operations are clear
    df_data_copy = df_data
    df_data_copy = df_data_copy[3:4]
    
    
    # gather training data in np array
    x_train = []
    y_train = np.array(df_data_copy['label'])
    for file_path in tqdm_notebook(df_data_copy['id']):
        x_train.append(np.array(Image.open('../data/train/{}'.format(file_path))))
    x_train = np.array(x_train)  
    
    # define a generator with the augmentor
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            preprocessing_function = augmentor
    )
    
    show_augmentations(train_datagen, x_train, y_train)
    
    