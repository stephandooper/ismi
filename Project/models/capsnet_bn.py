import numpy as np
from keras import layers, models, optimizers
from keras import backend as K
from keras.layers import BatchNormalization, LeakyReLU
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from PIL import Image
from models.capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask


def build_capsnet_bn(**kwargs):
    n_class=1
    routings = kwargs.get('routings', 1)
    dimensions = 16
    input_shape = (*kwargs.get('target_size', (96, 96)),3,)
    """
    A Capsule Network on MNIST.
    :param n_class: number of classes
    :param routings: number of routing iterations
    :return: Two Keras Models, the first one used for training, and the second one for evaluation.
            `eval_model` can also be used for training.
    """
    x = layers.Input(shape=input_shape)

    # Layer 1: Just a conventional Conv2D layer
    conv1 = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='valid', activation='relu', name='conv1')(x)
    conv1 = BatchNormalization()(conv1)
    conv1 = LeakyReLU()(conv1)
    
    conv2 = layers.Conv2D(filters=64, kernel_size=3, strides=2, padding='valid', activation='relu', name='conv2')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = LeakyReLU()(conv2)
    
    conv3 = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='valid', activation='relu', name='conv3')(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = LeakyReLU()(conv3)
    
    conv4 = layers.Conv2D(filters=128, kernel_size=3, strides=2, padding='valid', activation='relu', name='conv4')(conv3)
    conv4 = BatchNormalization()(conv4)
    conv4 = LeakyReLU()(conv4)
    
    conv5 = layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='valid', activation='relu', name='conv5')(conv4)
    conv5 = BatchNormalization()(conv5)
    conv5 = LeakyReLU()(conv5)
    
    conv6 = layers.Conv2D(filters=256, kernel_size=3, strides=2, padding='valid', activation='relu', name='conv6')(conv5)
    conv6 = BatchNormalization()(conv6)
    conv6 = LeakyReLU()(conv6)
   
    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
    primarycaps = PrimaryCap(conv6, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='valid')

    # Layer 3: Capsule layer. Routing algorithm works here.
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=dimensions, routings=routings,
                             name='digitcaps')(primarycaps)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary. :)
    out_caps = Length(name='capsnet')(digitcaps)

    # Decoder network.
    y = layers.Input(shape=(n_class,))
    masked_by_y = Mask()([digitcaps, y])  # The true label is used to mask the output of capsule layer. For training
    masked = Mask()(digitcaps)  # Mask using the capsule with maximal length. For prediction

    # Shared Decoder model in training and prediction
    decoder = models.Sequential(name='decoder')
    decoder.add(layers.Dense(256, activation='relu', input_dim=dimensions*n_class))
    decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
    decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))

    # Models for training and evaluation (prediction)
    train_model = models.Model([x, y], [out_caps, decoder(masked_by_y)])
    
    train_model.summary()

    return train_model
