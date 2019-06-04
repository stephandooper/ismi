from keras.layers import Input, Dense, Flatten, GlobalAveragePooling2D, Dropout, LeakyReLU, BatchNormalization, MaxPooling2D
from keras.models import Model
from keras.layers.convolutional import Conv2D

"""
"""

def build_custom_convnet(**kwargs):
    """
    target_size: The size of the target images.
    
    """
    inputs = Input(shape=(*kwargs.get('target_size', (96, 96)),3,))
    
    # Block 1
    x = Conv2D(32, kernel_size=(5, 5))(inputs)   
    x = MaxPooling2D()(x)
    
    # Block 2
    x = Conv2D(64, kernel_size=(3, 3))(x)
    x = MaxPooling2D()(x)
    
    # Block 3
    x = Conv2D(128, kernel_size=(3, 3))(x)
    x = MaxPooling2D()(x)

    # Block 4
    x = Conv2D(256, kernel_size=(3, 3))(x)
    x = MaxPooling2D()(x)
    
    
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=predictions)
    
    model.summary()
    
    return model