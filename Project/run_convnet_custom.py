from experiment import run_experiment

config = {'model': 'custom_convnet',
               'use_augment': True,           
               'epochs': 15,
               'batch_size': 32,
               'target_size':(96,96),
               'reduce_lr_on_plateau': True,
               'only_use_subset':False}

run_experiment(config, predict_test="convnet_custom", predict_val="convnet_custom")