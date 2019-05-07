from keras.layers import Input, Dense, Flatten
from keras.models import Model

def build_resnet(**kwargs):
    """
    weights ('imagenet'): Pre-trained weights, specify None to have no pre-training.
    
    """
    from keras.applications.resnet50 import ResNet50
    from keras.layers import Dense, GlobalAveragePooling2D
    
    weights = kwargs.get('weights', 'imagenet')
    
    base_model = ResNet50(weights=weights, include_top=False, )

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    
    return model