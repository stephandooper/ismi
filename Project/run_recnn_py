from experiment import run_experiment
config = {'model': 'recnn',
               'use_augment': True,           
               'epochs': 15,
               'batch_size': 32,
               'reduce_lr_on_plateau': True,
               'target_size':(96,96),
               'only_use_subset':False}

run_experiment(config, predict_test="recnn_augmentation_15", predict_val="recnn_augmentation_15")
