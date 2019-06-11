from experiment import run_experiment

config = {'model': 'capsnet',
               'use_augment': True,
               'model_params': {'weights': None},
               'epochs': 0,
               'use_capsnet': True,
               'batch_size': 32,
               'target_size':(96,96),
               'only_use_subset':False}
run_experiment(config, reproduce_result='1559578427.8389907', predict_test="capsnet_15epoch_aug", predict_val="capsnet_15epoch_aug")
