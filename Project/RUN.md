Add configurations that you want to run to this list. I'll pull the repository before every run and update this book with results and `run_id`.
# ResNet

# NasNet
## Pre-trained [Failed]
```python
config = {'model': 'nasnet',
               'use_augment': True,
               'model_params': {'weights': 'imagenet'},
               'epochs': 10,
               'batch_size': 32,
               'target_size':(96,96),
               'reduce_lr_on_plateau': True,
               'only_use_subset':False}

run_experiment(config, predict_test="nasnet_pretrained", predict_val="nasnet_pretrained")
```

## Not pre-trained [Done]
```python
config = {'model': 'nasnet',
               'use_augment': True,
               'model_params': {'weights': None},
               'epochs': 10,
               'batch_size': 32,
               'target_size':(96,96),
               'reduce_lr_on_plateau': True,
               'only_use_subset':False}

run_experiment(config, predict_test="nasnet_non_pretrained", predict_val="nasnet_non_pretrained")
```


```
INFO - ISMI - Started run with ID "236"
```

# ConvNet [Done]
## Augmentation (70 minutes) 
```python
config = {'model': 'convnet',
               'use_augment': True,
               'epochs': 10,
               'batch_size': 32,
               'target_size':(96,96),
               'reduce_lr_on_plateau': True,
               'only_use_subset':False}

run_experiment(config, predict_test="convnet_augmentation", predict_val="convnet_augmentation")
```

```console
INFO - ISMI - Started run with ID "231"
Tensorboard run 1559041663.5635588, Modelcheckpoint 1559041668.5452788. Fitting on more epochs would probably increase the accuracy even more!
```
## No augumentation (70 minutes) [Done]
```python
config = {'model': 'convnet',
               'use_augment': False,
               'epochs': 10,
               'batch_size': 32,
               'target_size':(96,96),
               'reduce_lr_on_plateau': True,
               'only_use_subset':False}

run_experiment(config, predict_test="convnet_no_augmentation", predict_val="convnet_no_augmentation")
```

```
INFO - ISMI - Started run with ID "232"
```

# Capsnet (10 hours)

```python
config = {'model': 'capsnet',
               'use_augment': True,
               'model_params': {'weights': None},
               'epochs': 15,
               'use_capsnet': True,
               'batch_size': 32,
               'target_size':(96,96),
               'reduce_lr_on_plateau': True,
               'only_use_subset':False}

run_experiment(config, predict_test="capsnet_15epochs", predict_val="capsnet_15epochs")
```

```console
RUN 238
```

# ReCNN
## Augmentation (7 hours) [DONE]
```python
config = {'model': 'recnn',
               'use_augment': True,           
               'epochs': 10,
               'batch_size': 32,
               'target_size':(96,96),
               'reduce_lr_on_plateau': True,
               'only_use_subset':False}

run_experiment(config, predict_test="recnn_augmentation", predict_val="recnn_augmentation")
```

```
INFO - ISMI - Started run with ID "232"
```
## Without Augmentation [Running]
Stopped because Klaus is still updating the architecture.
```python
config = {'model': 'recnn',
               'use_augment': False,           
               'epochs': 10,
               'batch_size': 32,
               'target_size':(96,96),
               'reduce_lr_on_plateau': True,
               'only_use_subset':False}

run_experiment(config, predict_test="recnn_no_augmentation", predict_val="recnn_no_augmentation")
```

```console
INFO - ISMI - Started run with ID "237"
```

# Results
Network | Augmented | Epochs | Validation accuracy | Test AUC 
--------|-----------|--------|---------------------|--------------
NasNet  | Yes       |15      | 0.8945              | 0.9623 	
ReCNN   | Yes       | 15     | 0.8560              | -
ReCNN   | No        | 15     | 0.8572              | - 
CapsNet | Yes       | 15     | 0.8580              | 0.8939
ConvNet | Yes       | 15     |                     | 0.9489 	
