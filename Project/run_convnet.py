from experiment import run_experiment

config = {'model': 'convnet',
               'use_augment': True,
               'epochs': 10,
               'batch_size': 32,
               'target_size':(96,96),
               'reduce_lr_on_plateau': True,
               'only_use_subset':False}

run_experiment(config, predict_test="convnet_augmentation", predict_val="convnet_augmentation")

config = {'model': 'convnet',
               'use_augment': False,
               'epochs': 10,
               'batch_size': 32,
               'target_size':(96,96),
               'reduce_lr_on_plateau': True,
               'only_use_subset':False}

run_experiment(config, predict_test="convnet_no_augmentation", predict_val="convnet_no_augmentation")
