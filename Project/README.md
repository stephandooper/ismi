# ISMI Project (LINUX ONLY!)

## Configuring Ponyland access
In order to get access to the Ponyland cluster from Radboud University one needs:
* A science account from Radboud
* Access to the University network (also possible via a VPN connection)

General information on how to get a science account, and set up a VPN connection, as well as how to connect to the ponies can be found on the [Ponyland wikiw](https://ponyland.science.ru.nl/doku.php?id=start)

## Installing Anaconda on Ponyland 
We assume that the user has access to the ponies. We first set up an anaconda environment to manage our packages. In order to avoid problems with disk space in the home folder of the user, we use the ``tensusers`` volume to install anaconda. We mostly follow steps 1-3 from the ordinary install process from [here](https://www.digitalocean.com/community/tutorials/how-to-install-anaconda-on-ubuntu-18-04-quickstart).

1. From the home folder, cd to the tensusers volume and replace <YOUR_USERNAME> ``` cd /vol/tensusers/<YOUR_USERNAME>/ ```.
2. Get the install link from the [anaconda webpage](https://www.anaconda.com/distribution/#download-section) and download it using ```wget```.
3. Once the download is complete, create a new bashrc script by using ``` nano bashrc```
4. In the bash script, put the following code and replace <USER> with your username ``` export PATH="/home/<USER>/anaconda3/bin:$PATH" ```
5. Activate the bash script using ``` source ~/.bashrc```. This will activate the conda environment.
6. 

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

