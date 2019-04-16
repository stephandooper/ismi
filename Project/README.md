# ISMI Project

## Dataset
The dataset can be acquired [here](https://www.kaggle.com/c/histopathologic-cancer-detection/data). You can unzip the file to [./data/](data/).

## Keras API

### Tensorboard
For hyperparameter tuning, we can use Tensorboard. This can be installed by running `pip install tensorboard`. You can run Tensorboard by running `tensorboard --logdir logs/` in Shell. Tensorboard will then start running on [http://localhost:6006/](http://localhost:6006/). You can add Tensorboard as callback in Keras:

```python
tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
model.fit_generator(..., callbacks=[tensorboard])
```

### EarlyStopping
It can be usefull to stop training when the validation loss does not increase anymore. With `patience = 1`, we wait one epoch before stopping early.

```python
earlystopping = EarlyStopping(monitor='val_loss', patience=1)
model.fit_generator(..., callbacks=[earlystopping])
```

### ModelCheckpoint
Can be used to save models between runs.

```python
modelcheckpoint = ModelCheckpoint(modelcheckpoint_name, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)
model.fit_generator(..., callbacks=[modelcheckpoint])
```

