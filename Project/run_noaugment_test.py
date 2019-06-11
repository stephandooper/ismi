from experiment import run_experiment
import gc


# Capsnet without Batch Normalization
config = {'model': 'capsnet',
               'use_augment': False,
               'model_params': {'weights': None},
               'epochs': 5,
               'use_capsnet': True,
               'batch_size': 32,
               'target_size':(96,96),
               'reduce_lr_on_plateau': True,
               'only_use_subset':False}

run_experiment(config, predict_test="capsnet_5epochs_augment", predict_val="capsnet_5epochs_augment")
gc.collect()

# Capsnet with batch normalization
config = {'model': 'capsnet_bn',
               'use_augment': False,
               'model_params': {'weights': None},
               'epochs': 5,
               'use_capsnet': True,
               'batch_size': 32,
               'target_size':(96,96),
               'reduce_lr_on_plateau': True,
               'only_use_subset':False}

run_experiment(config, predict_test="capsnet_bn_5epochs_augment", predict_val="capsnet_bn_5epochs_augment")
gc.collect()


# NasNet
config = {'model': 'nasnet',
               'use_augment': False,
               'model_params': {'weights': None},
               'epochs': 5,
               'batch_size': 32,
               'target_size':(96,96),
               'reduce_lr_on_plateau': True,
               'only_use_subset':False}


run_experiment(config, predict_test="nasnet_5epochs_augment", predict_val="nasnet_5epochs_augment")
gc.collect()
