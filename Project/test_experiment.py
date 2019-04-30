from experiment import run_experiment 


run_experiment(config={'model': 'dense',
               'model_params': {'weights': None},
               'epochs': 2,
               'batch_size': 32,
               'target_size':(96,96),
               'only_use_subset':True}, fish='This sucks ass.')
 

#ex.add_config({'model': 'dense',
#               'model_params': {'weights': None},
#               'epochs': 2,
#               'batch_size': 32,
#               'target_size':(96,96),
#               'only_use_subset':True})
#ex.run()
