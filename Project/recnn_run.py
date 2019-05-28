from experiment import run_experiment
config = {'model': 'recnn',
               'use_augment': False,           
               'epochs': 10,
               'batch_size': 32,
               'target_size':(96,96),
               'only_use_subset':False}

run_experiment(config, predict_test="recnn_no_augmentation", predict_val="recnn_no_augmentation")
