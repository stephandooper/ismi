# ISMI Project

## Installing dependencies and running Jupyter Notebook
1. Install [Conda](https://www.anaconda.com/distribution/#download-section).
2. Clone the repository and install the dependencies using Conda.
```bash
conda env create -f environment.yml
conda activate ismi19
```
3. Launch Jupyter Notebook.
```bash
nice -n 19 jupyter notebook --port 8888
```
Sometimes port 8888 can already be in use. In that case, another port will be automatically assigned, and port 8888 should be changed with the newly assignment port.

In case the jupyter notebook connection is refused by the client, try the following command:

```
nice -n 19 jupyter notebook --no-browser --ip=0.0.0.0
```

## Configuring on Ponyland (Linux only!)
Follow step 1-3 from the ordinary install process from [here](https://www.digitalocean.com/community/tutorials/how-to-install-anaconda-on-ubuntu-18-04-quickstart). If the .bashrc script does not exist, then manually add it with the following line
```
export PATH="/home/<USER>/anaconda3/bin:$PATH"
```
where ```<USER>``` is replaced by the username. Then, activate the bashrc with ``` source ./bashrc```. From this point onwards, the conda instance should be running and you can create your own environment. 


4. Tunnel internet traffic trough SSH, run from a terminal on your own machine.
```bash
ssh -L 8888:thunderlane:8888 user@applejack.science.ru.nl
ssh -L 6006:thunderlane:6006 user@applejack.science.ru.nl
```
The first tunnel is for the jupyter notebook itself. The second tunnel will be needed if you want to import tensorboard


5. Access Jupyter Notebook on your [own system](http://localhost:8888).

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

