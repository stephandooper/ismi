Add configurations that you want to run to this list. I'll pull the repository before every run and update this book with results and `run_id`.
# ResNet

# NasNet
## Pre-trained
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

## Not pre-trained
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
# ConvNet [RUNNING]
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

```
## No augumentation (70 minutes)
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

# Capsnet (10 hours)

```python
config = {'model': 'capsnet',
               'use_augment': True,
               'model_params': {'weights': None},
               'epochs': 10,
               'use_capsnet': True,
               'batch_size': 32,
               'target_size':(96,96),
               'reduce_lr_on_plateau': True,
               'only_use_subset':False}

run_experiment(config, predict_test="capsnet_10epochs", predict_val="capsnet_10epochs")
```

# ReCNN
## Augmentation (7 hours)
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
## Without Augmentation [STOPPED] (7 hours)
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
INFO - ISMI - Started run with ID "230"
```
