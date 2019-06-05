Add configurations that you want to run to this list. I'll pull the repository before every run and update this book with results and `run_id`.
# ResNet

# NasNet
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

# ReCNN (old params)
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
## Without Augmentation [DONE]
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

# ReCNN (new params)
## No Augmentation
Run ID: 267

## Augmentation
Run ID: 286

# Runs June 4th
Network | Settings | Machine | Time
--------|----------|---------|----
NasNet | 5 epochs, with augmentation | Thunderlane | 6 hours
NasNet | 5 epochs, without augmentation | Thunderlane | 6 hours
NasNet | 30 epochs | Thunderlane | 32 hours
ConvNet Custom | 15 epochs, k-max pooling | Thunderlane | 3 hours
CapsNet | Batch Normalization | Thunderlane | 3 hours
ReCNN  | Different parameters, 15 epochs, augmentation | Twist

Move run results to "results" and remove from table above when done.

# Ensemble
Set `shuffle=False`
Generate using trained weights using:
* Convnet augmented
* Capsnet 15
* Nasnet 15


# Results
Network | Augmented | Epochs | Validation accuracy | Test AUC 
--------|-----------|--------|---------------------|--------------
NasNet  | Yes       | 15     | 0.8945              | 0.9623 	
ReCNN (old)  | Yes       | 15     | 0.8560              | -
ReCNN  (old) | No        | 15     | 0.8572              | - 
CapsNet | Yes       | 15     | 0.8580              | 0.8939
ConvNet | Yes       | 15     | 0.8682              | 0.9489 	
ConvNet | No        | 15     | 0.8345              | 0.9277
ReCNN   | No        | 5      | 0.8269              | 0.9143
ReCNN   | Yes       | 5      | 0.8915              |  ?

