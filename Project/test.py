# -*- coding: utf-8 -*-
"""
Created on Mon May  6 17:14:07 2019

@author: Stephan
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from imgaug import augmenters as iaa

data_dir = 'data/0000ec92553fda4ce39889f9226ace43cae3364e.tif'

im = Image.open(data_dir)
plt.imshow(im)

img = np.array(im)

# Basic image augmentations

''' 
To think about:
    1. In what way does it extend one augmentation extend another?
    2. Do they (randomly) apply augmentation on top of augmentation? or something else?
'''


# BASIC DONE
'''
basic: 
    Operations
    ---
        1. 90 degree rotations
        2. horizontal/vertical image mirroring
'''

basic = iaa.Sequential( [iaa.Affine(rotate=90) ,iaa.Fliplr(1), iaa.Flipud(1)])

basic_img = basic.augment_image(img)



'''
Morphology
    Operations
    ---
        1. scaling
        2. elastic deformation
        3. additive Gaussian noise (perturbing the signal-to-noise ratio), 
        4. Gaussian blurring (simulating out-of-focus artifacts).
'''

morphology = iaa.Sequential([iaa.GaussianBlur(sigma=0.35), 
                             iaa.AdditiveGaussianNoise(scale=0.05*255),
                             iaa.ElasticTransformation(alpha=0.35, sigma=0.5),
                             iaa.Affine(scale=(1, 1.5))])

morph_img = morphology.augment_image(img)
plt.imshow(img)
plt.imshow(morph_img )

'''
Brightness & Contract (BC)
    Operations
    ---
        1. Random brightness image perturbations
        2. Random contrast image perturbations
        3. Haeberli and Voorhies (1994)

'''

bc = iaa.Sequential([iaa.ContrastNormalization((0.75, 1.25)), 
                     iaa.Multiply((0.5, 1.5))])
bc_img = bc.augment_image(img)

plt.imshow(img)
plt.imshow(bc_img)

'''
Hue Saturation Value (HSV)
    Operations
    ---
        1. Randomly shifting hue and saturations channels in the HSV color space
        2. Color variation strength: light and strong
        3. Van der Walt et al. 2014
        

'''

#Values are kind of extreme

hsv = iaa.WithColorspace(
    to_colorspace="HSV",
    from_colorspace="RGB",
    children=iaa.Sequential([iaa.WithChannels((1), iaa.Add((-100, 150))),
                         iaa.WithChannels((0), iaa.Add((-100, 150)))])
)


hsv_image = hsv.augment_image(img)

plt.imshow(hsv_image)
plt.imshow(img)

