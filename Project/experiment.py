import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sacred
from PIL import Image
import os
from time import time
from tqdm import tqdm
from IPython.core.display import display, HTML            

# Suppress GPU if needed
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

from keras.layers import Input, Dense, Flatten
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, Callback

from skimage import exposure

from sacred import Experiment
from sacred.observers import MongoObserver
import pymongo

from constants import MONGO_URI, RANDOM_SEED
from data.data import load_data
from models.resnet import build_resnet
from models.densenet import build_dense
from generators.augment import augmentor

# TODO add more parameters    
def build_generators(batch_size=32, target_size= (96,96), only_use_subset=False, use_augment=False):
    (x_train, y_train, meta_train), (x_valid, y_valid, meta_valid), (x_test, y_test, meta_test) = load_data()
    
    if only_use_subset:
        x_train = x_train[0:10000]
        y_train = y_train[0:10000]
        x_valid = x_valid[0:10000]
        y_valid = y_valid[0:10000]
        
        x_test = x_test[0:100]
        y_test = y_test[0:100]
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        # Specify other augmentations here.
    )
    
    if use_augment:
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            preprocessing_function = augmentor
        )
    
    print("[!] Creating validation_generator")
    validation_generator = train_datagen.flow(
        x=x_valid,
        y=np.ravel(y_valid),
        batch_size=batch_size)

    print("[!] Creating train_generator")
    train_generator = train_datagen.flow(
        x=x_train,
        y=np.ravel(y_train),
        batch_size=batch_size)

    test_datagen = ImageDataGenerator(
        rescale=1./255,
        # Specify other augmentations here.
    )

    print("[!] Creating test_generator")
    test_generator = test_datagen.flow(
        x=x_test,
        y=np.ravel(y_test),
        batch_size=32,
        seed=0,
        shuffle=False
    )

    return train_generator, validation_generator, test_generator

def run_experiment(config, predict_test = True):

    ex = Experiment('ISMI', interactive=True)
    # log to the mongoDB instance living in the cloud
    client = pymongo.MongoClient(MONGO_URI)
    ex.observers.append(MongoObserver.create(client=client))
    
    ex.add_config(config)
    ex.add_config({'seed': RANDOM_SEED})

    @ex.capture
    def my_metrics(_run, logs):
        _run.log_scalar("loss", float(logs.get('loss')))
        _run.log_scalar("acc", float(logs.get('acc')))
        _run.log_scalar("val_loss", float(logs.get('val_loss')))
        _run.log_scalar("val_acc", float(logs.get('val_acc')))
        _run.result = float(logs.get('val_acc'))


    # a callback to log to Sacred, found here: https://www.hhllcks.de/blog/2018    /5/4/version-your-machine-learning-models-with-sacred
    class LogMetrics(Callback):
        def on_epoch_end(self, _, logs={}):
            my_metrics(logs=logs)

    # the entry point of the experiment
    @ex.main
    def run(_run):

        config = _run.config
        
        print('[!] Loading data')
        
        batch_size = config.get('batch_size')
        target_size = config.get('target_size')
        only_use_subset = config.get('only_use_subset')
        
        use_augment = config.get('use_augment')
        
        train_generator, validation_generator, test_generator =            build_generators(batch_size=batch_size,target_size=target_size,only_use_subset=only_use_subset,use_augment=use_augment)
        
        print('[!] Building model')
        
        # The function for building the model
        model_func = model_dict[config.get('model')]
        # Actually invoke the function
        model_params = config.get('model_params',{})

        if config.get('model') == 'dense' and 'target_size' not in model_params:
            model_params['target_size'] = target_size

        # TODO add kwargs
        model = model_func(**model_params)

        # TODO: parametrize optimizer and learning rate here
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

        epochs = config.get('epochs')
        
        print('[!] Training model')

        model.fit_generator(train_generator, 
                            steps_per_epoch=len(train_generator),
                            epochs=epochs, 
                            validation_data=validation_generator, 
                            validation_steps=len(validation_generator),
                            callbacks=[tensorboard, LogMetrics()])
        
        if predict_test:
            print('[!] Predicting test set')
        
            prediction = model.predict_generator(test_generator, steps=len(test_generator), verbose=1)
            
            data = {'case': np.arange(len(prediction)), 'prediction': np.ravel(prediction)}
            submission = pd.DataFrame(data=data)
            submission.to_csv('./data/submission.csv', index=False)
            
            display(HTML('Prediction saved to file: <a target="_blank" href="./data/submission.csv">data/submission.csv</a>'))
        
            # add the submissions as an artifact
            _run.add_artifact('./data/submission.csv')
            
    run = ex.run()
    
    return run

model_dict = {
    'dense' : build_dense,
    'resnet': build_resnet 
}
    

