from keras.layers import Input, Dense, Flatten, GlobalAveragePooling2D, Dropout, LeakyReLU, BatchNormalization, GlobalMaxPooling2D
from keras.models import Model
from keras.layers.convolutional import Conv2D

"""
"""

def build_convnet(**kwargs):
    """
    target_size: The size of the target images.
    
    """
    inputs = Input(shape=(*kwargs.get('target_size', (96, 96)),3,))
    
    # Block 1
    x = Conv2D(64, kernel_size=(5, 5), activation='relu')(inputs)
    x = BatchNormalization()(x)    
    x = GlobalMaxPooling2D()(x)
    x = LeakyReLU()(x)
    
    # Block 2
    x = Conv2D(128, kernel_size=(3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = GlobalMaxPooling2D()(x)
    x = LeakyReLU()(x)
    
    # Block 3
    x = Conv2D(256, kernel_size=(3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = GlobalMaxPooling2D()(x)
    x = LeakyReLU()(x)

    x = Conv2D(512, kernel_size=(3, 3), activation='relu')(x)
    x = GlobalMaxPooling2D()(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=predictions)
    
    model.summary()
    
    return model