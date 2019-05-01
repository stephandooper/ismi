from keras.layers import Input, Dense, Flatten
from keras.models import Model

def build_dense(**kwargs):
    """
    target_size: The size of the target images.
    
    """
    target_size = kwargs.get('target_size')
    inputs = Input(shape=(*target_size,3,))

    x = Dense(64, activation='relu')(inputs)
    x = Dense(64, activation='relu')(x)
    x = Flatten()(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=predictions)
    
    return model