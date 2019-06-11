from keras_gcnn.applications.densenetnew import GDenseNet, GDenseNetFCN
from keras import backend as K

def build_recnn(**kwargs):
    """
    target_size: The size of the target images.
    
    """
    depth = kwargs.get('depth', None)
    
    nb_dense_block = kwargs.get('nb_dense_block', 5)
    nb_layers_per_block = kwargs.get('nb_layers_per_block', 1)
    # default for D4 here now
    growth_rate = kwargs.get('growth_rate', 8)  # number of z2 maps equals growth_rate * group_size, so keep this small.
    nb_filter = kwargs.get('nb_filter', 8)
    
    bottleneck = kwargs.get('bottleneck', True)
    reduction = kwargs.get('reduction', 0.33)
    include_top = kwargs.get('include_top',True)
  #  pooling = kwargs.get('pooling','avg')
    
    subsample_initial_block = kwargs.get('subsample_initial_block',False)
    
    
    dropout_rate = kwargs.get('dropout_rate', 0.0)  # 0.0 for data augmentation
    # C4 includes 90 degree rotations, D4 additionally includes reflections in x and y axis.
    conv_group = kwargs.get('conv_group', 'D4')
    use_gcnn = kwargs.get('use_gcnn', True)
    target_size = kwargs.get('target_size', (96,96))
    input_size = (target_size[0], target_size[1], 3)
    
    model = GDenseNet(mc_dropout=False, padding='same', nb_dense_block=nb_dense_block, nb_layers_per_block = nb_layers_per_block,
                      growth_rate=growth_rate, nb_filter=nb_filter, 
                      dropout_rate=dropout_rate, weights=None, input_shape=input_size, depth=depth, classes = 1, use_gcnn=use_gcnn, 
                      conv_group=conv_group,activation='sigmoid', bottleneck=bottleneck, reduction=reduction, include_top=include_top,
                      subsample_initial_block=subsample_initial_block)
    
    return model
x=build_recnn()
x=x.summary()
