## Overleaf link

https://www.overleaf.com/9492892348gbrjxhvvyrcw

# ISMI Project (LINUX ONLY!)
Below is the tutorial to set up the project on the Ponyland cluster. In case you want to work locally, you can ignore this section. It might still be helpfull to read the sections on cloning the anaconda environment.

## Configuring Ponyland access
In order to get access to the Ponyland cluster from Radboud University one needs:
* A science account from Radboud
* Access to the University network (also possible via a VPN connection)

General information on how to get a science account, and set up a VPN connection, as well as how to connect to the ponies can be found on the [Ponyland wikiw](https://ponyland.science.ru.nl/doku.php?id=start)

## Installing Anaconda on Ponyland 
We assume that the user has access to the ponies. We first set up an anaconda environment to manage our packages. In order to avoid problems with disk space in the home folder of the user, we use the ``tensusers`` volume to install anaconda. We mostly follow steps 1-3 from the ordinary install process from [here](https://www.digitalocean.com/community/tutorials/how-to-install-anaconda-on-ubuntu-18-04-quickstart).

1. From the home folder, cd to the tensusers volume and replace <YOUR_USERNAME> ``` cd /vol/tensusers/<YOUR_USERNAME>/ ```.
2. Get the install link from the [anaconda webpage](https://www.anaconda.com/distribution/#download-section) and download it using ```wget```.
3. Once the download is complete, create a new bashrc script by using 
```bash
nano bashrc
```
4. In the bash script, put the following code and replace <USER> with your username ``` export PATH="/home/<USER>/anaconda3/bin:$PATH" ```
5. Activate the bash script using ``` source ~/.bashrc```. This will activate the conda environment.
  
### Cloning the anaconda environment

The git repository includes a premade environment (called ismi.yml). With this, you can clone the complete environment using 
```bash
conda env create -f ismi.yml
conda activate ismi19
```
Now, the complete environment will be copied.

## Running a Jupyter Notebook on Ponyland
In order to run a jupyter notebook on the server, we need to tunnel the network traffic through applejack. 

1. Activate the anaconda environment (see previous section).
2. Launch the jupyter notebook using the following command
```
nice -n 19 jupyter notebook --port 8888 --no-browser --ip=0.0.0.0
```
This will create a job with niceness value 19. The no-browser command will prevent the notebook from running within the terminal, and the ip command opens the ports (will fix any connection refused errors). 
If port 8888 is closed or already taken by any chance, then you will be automatically assigned a new port (remember this), or you will have to assign a new port yourself.

### Configuring on Ponyland (Linux only!)
In order to run a jupyter notebook in a local browser, we need to tunnel internet traffic through ssh. First, open a terminal on your local machine and type in the following commands (assuming you assigned port 8888 earlier):
```bash
ssh -L 8888:thunderlane:8888 user@applejack.science.ru.nl
ssh -L 6006:thunderlane:6006 user@applejack.science.ru.nl
```
The first tunnel is for the jupyter notebook itself. The second tunnel will be needed if you want to import tensorboard
You should now be able to access the jupyter on your [own system](http://localhost:8888) (via http://localhost:/8888).

## Dataset
There are several different datasets that could be used for the challenge. The first one is a Kaggle competition (not supported)
The dataset can be acquired [here](https://www.kaggle.com/c/histopathologic-cancer-detection/data). You can unzip the file to [./data/](data/).

The second one is from the [Grand Challenge website](https://patchcamelyon.grand-challenge.org/) . This one is basically the same as the Kaggle dataset, but contains duplicates and is already split into a training/validation/test set.



## Sacred with MongoDB (KLAUS!!!!)


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
