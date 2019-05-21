from experiment import run_experiment

config = {'model': 'capsnet',
               'use_augment': True,
               'model_params': {'weights': None},
               'epochs': 5,
               'use_capsnet': True,
               'batch_size': 32,
               'target_size':(96,96),
               'only_use_subset':False}

run_experiment(config, predict_test=False)