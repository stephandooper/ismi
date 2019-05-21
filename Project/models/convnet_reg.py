from keras.layers import Input, Dense, Flatten, GlobalAveragePooling2D, Dropout, LeakyReLU, BatchNormalization
from keras.models import Model
from keras.layers.convolutional import Conv2D
from keras.regularizers import l1

"""
The architecture of such CNN classifiers consisted of 9layers of strided convolutions with 32, 64, 64, 128, 128,256, 256, 512 and 512 3x3 filters, stride of 2 in the evenlayers, BN and LRA; followed by global average pooling;50% dropout; a dense layer with 512 units, BN and LRA;and a linear dense layer with either 2 or 9 units depend-ing on the classification task, followed by a softmax.  Weapplied L2 regularization with a factor of 1×10−6.We minimized the cross-entropy loss using stochasticgradient descent with Adam optimization and 64-sampleclass-balanced mini-batch, decreasing the learning rate bya factor of 10 starting from 1×10−2every time the valida-tion loss stopped improving for 4 consecutive epochs until1×10−5.  Finally, we selected the weights correspondingto the model with the lowest validation loss during train-ing
"""

def build_convnet_reg(**kwargs):
    """
    target_size: The size of the target images.
    
    """
    inputs = Input(shape=(*kwargs.get('target_size', (96, 96)),3,))
    
    # Block 1
    x = Conv2D(32, kernel_size=(3, 3), kernel_regularizer=l1(0.001))(inputs)
    x = Conv2D(64, kernel_size=(3, 3), strides=(2, 2), kernel_regularizer=l1(0.001))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    
    # Block 2
    x = Conv2D(64, kernel_size=(3, 3), kernel_regularizer=l1(0.001))(x)
    x = Conv2D(128, kernel_size=(3, 3), strides=(2, 2), kernel_regularizer=l1(0.001))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    
    # Block 3
    x = Conv2D(128, kernel_size=(3, 3), kernel_regularizer=l1(0.001))(x)
    x = Conv2D(256, kernel_size=(3, 3), strides=(2, 2), kernel_regularizer=l1(0.001))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    
    # Block 4
    x = Conv2D(256, kernel_size=(3, 3), kernel_regularizer=l1(0.001))(x)
    x = Conv2D(512, kernel_size=(3, 3), strides=(2, 2), kernel_regularizer=l1(0.001))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2D(512, kernel_size=(3, 3), kernel_regularizer=l1(0.001))(x)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu', kernel_regularizer=l1(0.001))(x)
    predictions = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=predictions)
    
    model.summary()
    
    return model