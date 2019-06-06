from experiment import run_experiment
import gc

# ConvNet custom
config = {'model': 'custom_convnet',
          'use_augment': True,
          'epochs': 15,
          'batch_size': 32,
          'target_size':(96,96),
          'reduce_lr_on_plateau': True,
          'only_use_subset':False}

run_experiment(config, predict_test="custom_convnet_augmentation", predict_val="custom_convnet_augmentation")
gc.collect()

# Capsnet without Batch Normalization
config = {'model': 'capsnet',
               'use_augment': True,
               'model_params': {'weights': None},
               'epochs': 15,
               'use_capsnet': True,
               'batch_size': 32,
               'target_size':(96,96),
               'reduce_lr_on_plateau': True,
               'only_use_subset':False}

run_experiment(config, predict_test="capsnet_15epochs", predict_val="capsnet_15epochs")
gc.collect()

# Capsnet with batch normalization
config = {'model': 'capsnet_bn',
               'use_augment': True,
               'model_params': {'weights': None},
               'epochs': 15,
               'use_capsnet': True,
               'batch_size': 32,
               'target_size':(96,96),
               'reduce_lr_on_plateau': True,
               'only_use_subset':False}

run_experiment(config, predict_test="capsnet_bn_15epochs", predict_val="capsnet_bn_15epochs")
gc.collect()

# NasNet
config = {'model': 'nasnet',
               'use_augment': True,
               'model_params': {'weights': None},
               'epochs': 30,
               'batch_size': 32,
               'target_size':(96,96),
               'reduce_lr_on_plateau': True,
               'only_use_subset':False}

run_experiment(config, predict_test="nasnet_30epochs", predict_val="nasnet_30epochs")
