Add configurations that you want to run to this list. I'll pull the repository before every run and update this book with results and `run_id`.

# NasNet

# Capsnet

```python
config = {'model': 'capsnet',
               'use_augment': True,
               'model_params': {'weights': None},
               'epochs': 10,
               'use_capsnet': True,
               'batch_size': 32,
               'target_size':(96,96),
               'only_use_subset':False}

run_experiment(config, predict_test="capsnet_10epochs")
```

# ReCNN
## Augmentation
```python
config = {'model': 'recnn',
               'use_augment': True,           
               'epochs': 10,
               'batch_size': 32,
               'target_size':(96,96),
               'only_use_subset':False}

run_experiment(config, predict_test="recnn_augmentation")
```
## Without Augmentation [RUNNING]
```python
config = {'model': 'recnn',
               'use_augment': False,           
               'epochs': 10,
               'batch_size': 32,
               'target_size':(96,96),
               'only_use_subset':False}

run_experiment(config, predict_test="recnn_no_augmentation")
```

```console
INFO - ISMI - Started run with ID "225"
```

# 
