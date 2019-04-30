import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sacred
from PIL import Image
import os
from time import time
from tqdm import tqdm

# Suppress GPU if needed
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from keras.layers import Input, Dense, Flatten
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, Callback

from skimage import exposure

from sacred import Experiment
from sacred.observers import MongoObserver
import pymongo

from constants import MONGO_URI, RANDOM_SEED

ex = Experiment('ISMI', interactive=True)
# log to the mongoDB instance living in the cloud
client = pymongo.MongoClient(MONGO_URI)
ex.observers.append(MongoObserver.create(client=client))

ex.add_config({'seed': RANDOM_SEED})


@ex.capture
def my_metrics(_run, logs):
    _run.log_scalar("loss", float(logs.get('loss')))
    _run.log_scalar("acc", float(logs.get('acc')))
    _run.log_scalar("val_loss", float(logs.get('val_loss')))
    _run.log_scalar("val_acc", float(logs.get('val_acc')))
    _run.result = float(logs.get('val_acc'))
    
    
# a callback to log to Sacred, found here: https://www.hhllcks.de/blog/2018/5/4/version-your-machine-learning-models-with-sacred
class LogMetrics(Callback):
    def on_epoch_end(self, _, logs={}):
        my_metrics(logs=logs)

# the entry point of the experiment
@ex.main
def run(_run):
    
    config = _run.config
    
    print('Running experiment!')
    
    batch_size = config.get('batch_size')
    target_size = config.get('target_size')
    only_use_subset = config.get('only_use_subset')
    train_generator, validation_generator, test_generator = load_data(batch_size = batch_size, target_size=target_size, 
                                                                      only_use_subset=only_use_subset)
    
    
    model = build_model()
    
    tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
    
    epochs = config.get('epochs')

    model.fit_generator(train_generator, 
                        steps_per_epoch=len(train_generator),
                        epochs=epochs, 
                        validation_data=validation_generator, 
                        validation_steps=len(validation_generator),
                        callbacks=[tensorboard, LogMetrics()])
    
    prediction = model.predict_generator(test_generator, steps = len(test_generator), verbose=1)
    submission = pd.read_csv('./data/sample_submission.csv')
    submission['label'] = prediction
    submission.to_csv('./data/submission.csv', index=False)
    
    # add the submissions as an artifact
    _run.add_artifact('./data/submission.csv')
    
    
# TODO parametrize
def build_model():
    from keras.applications.resnet50 import ResNet50
    from keras.layers import Dense, GlobalAveragePooling2D
    
    base_model = ResNet50(weights='imagenet', include_top=False)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model
    
    
# TODO add more parameters    
def load_data(batch_size=32, target_size= (96,96), only_use_subset=False):
    df_data = pd.read_csv('./data/train_labels.csv')
    df_data['id'] = df_data['id'].astype(str) + '.tif'

    df_data_test = pd.read_csv('./data/sample_submission.csv')
    df_data_test['id'] = df_data_test['id'].astype(str) + '.tif'
    
        # Load data into memory
    df_data_copy = df_data
    if only_use_subset:
        df_data_copy = df_data_copy[0:10000]

    x_train = []
    y_train = np.array(df_data_copy['label'])
    print('Loading data')
    for file_path in tqdm(df_data_copy['id']):
        x_train.append(np.array(Image.open('data/train/{}'.format(file_path))))
    x_train = np.array(x_train)
    
    # Keras-inbuilt generator
    train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.1
    # Specify other augmentations here.
    )
    
    train_generator = train_datagen.flow(
    x=x_train,
    y=y_train,
    batch_size=batch_size,
    subset='training')

    validation_generator = train_datagen.flow(
    x=x_train,
    y=y_train,
    batch_size=batch_size,
    subset='validation')
    
    test_datagen = ImageDataGenerator(
        rescale=1./255
    )

    test_generator = test_datagen.flow_from_dataframe(
        dataframe=df_data_test,
        directory='data/test/',
        target_size=target_size,
        batch_size=batch_size,
        x_col='id',
        y_col='label',
        class_mode='other',
        seed=0,
        shuffle=False
    )

    return train_generator, validation_generator, test_generator
