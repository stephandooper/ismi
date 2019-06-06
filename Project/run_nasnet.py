from experiment import run_experiment

config = {'model': 'nasnet',
               'use_augment': True,
               'model_params': {'weights': None},
               'epochs': 5,
               'batch_size': 64,
               'target_size':(96,96),
               'reduce_lr_on_plateau': True,
               'only_use_subset':False}

run_experiment(config, predict_test="Nasnet_5epoch_aug", predict_val="Nasnet_5epoch_aug")

config = {'model': 'nasnet',
               'use_augment': False,
               'model_params': {'weights': None},
               'epochs': 5,
               'batch_size': 64,
               'target_size':(96,96),
               'reduce_lr_on_plateau': True,
               'only_use_subset':False}

run_experiment(config, predict_test="Nasnet_5epoch_nonaug", predict_val="Nasnet_5epoch_nonaug")

