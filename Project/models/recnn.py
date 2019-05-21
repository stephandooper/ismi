from keras_gcnn.applications.densenetnew import GDenseNet
from keras import backend as K

## TODO parameterize
# Parameters for the DenseNet model builder
depth = 22
nb_dense_block = 1
growth_rate = 3  # number of z2 maps equals growth_rate * group_size, so keep this small.
nb_filter = 8
dropout_rate = 0.0  # 0.0 for data augmentation
conv_group = 'D4'  # C4 includes 90 degree rotations, D4 additionally includes reflections in x and y axis.
use_gcnn = True

def build_recnn(**kwargs):
    """
    target_size: The size of the target images.
    
    """
    depth = kwargs.get('depth', 22)
    nb_dense_block = kwargs.get(nb_dense_block, 1)
    growth_rate = kwargs.get(growth_rate, 3)  # number of z2 maps equals growth_rate * group_size, so keep this small.
    nb_filter = kwargs.get(nb_filter, 8)
    dropout_rate = kwargs.get(dropout_rate, 0.0)  # 0.0 for data augmentation
    conv_group = kwargs.get(conv_group, 'D4')  # C4 includes 90 degree rotations, D4 additionally includes reflections in x and y axis.
    use_gcnn = kwargs.get(use_gcnn, True)
    target_size = kwargs.get('target_size', (96,96))
    input_size = (target_size[0], target_size[1], 3)

    model = GDenseNet(mc_dropout=False, padding='same', nb_dense_block=nb_dense_block, growth_rate=growth_rate, nb_filter=nb_filter, dropout_rate=dropout_rate, weights=None, input_shape=input_size, depth=depth, classes = 1, use_gcnn=use_gcnn, conv_group=conv_group,activation='sigmoid')
    
    return model