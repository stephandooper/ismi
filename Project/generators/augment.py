# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 17:26:09 2019

@author: Stephan
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from imgaug import augmenters as iaa


'''Define the augmentation space and topology '''
basic = iaa.SomeOf((0,None), [iaa.Affine(rotate=90) ,iaa.Fliplr(1), iaa.Flipud(1)])

morphology = iaa.SomeOf((0,None),[iaa.GaussianBlur(sigma=(0.0, 0.35)),
                             iaa.AdditiveGaussianNoise(scale=(0, 0.07*255)),
                             iaa.ElasticTransformation(alpha=(0, 0.55), sigma=0.25),
                             iaa.Affine(scale=(1, 1.25))])

bc = iaa.SomeOf((0,None), [iaa.ContrastNormalization((0.65, 1.35)),
                     iaa.Multiply((0.9, 1.1))])

hsv = iaa.SomeOf((0,None),
                 [
                    iaa.WithColorspace(
                    to_colorspace="HSV",
                    from_colorspace="RGB",
                    children=iaa.Sequential([iaa.WithChannels((1), iaa.Add((-20, 20))),
                             iaa.WithChannels((0), iaa.Add((-20, 20)))])
                    )
                 ]
                )


def define_augmentation(p_basic=0.2, p_morph=0.2, p_bc=0.2, p_hsv=0.2):
    '''
    Define the image augmentation pipeline
    
        Parameters
        ---
            p_basic: probability that an augmentation of type basic will occur
            p_morph: probability that an augmentation of type morphology will occur
            p_bc: probability that an augmentation of type bc will occur
            p_hsv: probability that an augmentation of type hsv will occur
    '''

    aug = iaa.Sequential(
        [
            iaa.Sometimes(p_basic, basic),
            iaa.Sometimes(p_morph, morphology),
            iaa.Sometimes(p_bc, bc),
            iaa.Sometimes(p_hsv, hsv)
        ]
    )
    return aug

def augmentor(image, *args, **kwargs):
    '''
    Wrapper function for the augmentation
    This function is fed into the Keras generator as an executor.
    
    Parameters
    ---
        image: the image to be (possibly) augmented
    '''
    #print(image.shape)
    #print(image.dtype)
    image = image.astype('uint8')
    image = np.array(image)
    
    aug = define_augmentation()
    image = aug.augment_image(image)
    image = image.astype('float32')
    return image



def show_augmentations(img):
    '''
    Shows several augmentations for a single example image
    
    Parameters
    ---
        img: the example image
            
    '''
    images = np.array(
        [img for _ in range(32)],
        dtype=np.uint8
    )
    

    w=4
    h=8
    fig=plt.figure(figsize=(8, 8))
    columns = 4
    rows = 5
    for i in range(1, columns*rows +1):
        fig.add_subplot(rows, columns, i)
        plt.imshow(augmentor(images[i]))
    plt.show()



if __name__ == '__main__':
    '''
    Shows an example of the augmentations
    
    '''
    (x_train, y_train, meta_train), (x_valid, y_valid, meta_valid), (x_test, y_test, meta_test) = load_data()
    show_augmentations(x_train[0])
    
