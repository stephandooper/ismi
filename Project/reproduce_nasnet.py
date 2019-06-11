from experiment import run_experiment

config = {'model': 'nasnet',
               'use_augment': True,
               'model_params': {'weights': None},
               'epochs': 0,
               'batch_size': 64,
               'target_size':(96,96),
               'reduce_lr_on_plateau': True,
               'only_use_subset':False}

run_experiment(config, reproduce_result='1559109999.9353304', predict_test="Nasnet_15epoch_aug", predict_val="Nasnet_15epoch_aug")

