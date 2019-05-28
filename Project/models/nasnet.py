from keras.layers import Input, Dense, Flatten
from keras.models import Model

def build_nasnet(**kwargs):
    """
    weights ('imagenet'): Pre-trained weights, specify None to have no pre-training.
    
    """
    from keras.applications.nasnet import NASNetLarge
    from keras.layers import Dense, GlobalAveragePooling2D
    
    weights = kwargs.get('weights', 'imagenet')
    
    base_model = NASNetLarge(input_shape=(96, 96, 3), weights=weights, include_top=False)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    x = Dense(512, activation='relu')(x)

    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    
    return model