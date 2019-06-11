from experiment import run_experiment

config = {'model': 'convnet',
               'use_augment': True,
               'epochs': 0,
               'batch_size': 32,
               'target_size':(96,96),
               'reduce_lr_on_plateau': True,
               'only_use_subset':False}

run_experiment(config, reproduce_result='1559041668.5452788', predict_test="convnet_15epoch", predict_val="convnet_15epoch")
