from experiment import run_experiment

config = {'model': 'nasnet',
               'use_augment': True,
               'model_params': {'weights': None},
               'epochs': 5,
               'batch_size': 64,
               'target_size':(96,96),
               'reduce_lr_on_plateau': True,
               'only_use_subset':False}

#run_experiment(config, predict_test=True)

config = {'model': 'convnet',
               'use_augment': True,
               'model_params': {'weights': None},
               'epochs': 5,
               'batch_size': 64,
               'target_size':(96,96),
               'reduce_lr_on_plateau': True,
               'only_use_subset':False}

run_experiment(config, predict_test=True)


config = {'model': 'convnet_reg',
               'use_augment': True,
               'model_params': {'weights': None},
               'epochs': 20,
               'batch_size': 64,
               'target_size':(96,96),
               'reduce_lr_on_plateau': True,
               'only_use_subset':False}

run_experiment(config, predict_test=True)
