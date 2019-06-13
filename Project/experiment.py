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
from keras import backend as K

# Suppress GPU if needed
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

from keras.layers import Input, Dense, Flatten
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, Callback, ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam

from skimage import exposure

from sacred import Experiment
from sacred.observers import MongoObserver
import pymongo

from constants import MONGO_URI, RANDOM_SEED
from data.data import load_data
from models.resnet import build_resnet
from models.densenet import build_dense
from models.nasnet import build_nasnet
from models.convnet import build_convnet
from models.convnet_custom import build_custom_convnet
from models.capsnet import build_capsnet
from models.capsnet_bn import build_capsnet_bn
from models.convnet_reg import build_convnet_reg
from models.recnn import build_recnn

from generators.augment import augmentor
from keras.preprocessing.image import ImageDataGenerator, NumpyArrayIterator


class NumpyArrayIteratorDouble(NumpyArrayIterator):
    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros(tuple([len(index_array)] + list(self.x.shape)[1:]),
                           dtype=self.dtype)
        for i, j in enumerate(index_array):
            x = self.x[j]
            params = self.image_data_generator.get_random_transform(x.shape)
            x = self.image_data_generator.apply_transform(
                x.astype(self.dtype), params)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x

        if self.save_to_dir:
            for i, j in enumerate(index_array):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(
                    prefix=self.save_prefix,
                    index=j,
                    hash=np.random.randint(1e4),
                    format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        batch_x_miscs = [xx[index_array] for xx in self.x_misc]
        x = batch_x if batch_x_miscs == [] else [batch_x] + batch_x_miscs
        
        output = ([x, self.y[index_array]],)
        if self.y is None:
            return output[0]
        output += ([self.y[index_array], x],)
        if self.sample_weight is not None:
            output += (self.sample_weight[index_array],)
        return output


class ImageDataGeneratorDouble(ImageDataGenerator):
    def flow(self,
             x,
             y=None,
             batch_size=32,
             shuffle=True,
             sample_weight=None,
             seed=None,
             save_to_dir=None,
             save_prefix='',
             save_format='png',
             subset=None):
        """Takes data & label arrays, generates batches of augmented data.
        # Arguments
            x: Input data. Numpy array of rank 4 or a tuple.
                If tuple, the first element
                should contain the images and the second element
                another numpy array or a list of numpy arrays
                that gets passed to the output
                without any modifications.
                Can be used to feed the model miscellaneous data
                along with the images.
                In case of grayscale data, the channels axis of the image array
                should have value 1, in case
                of RGB data, it should have value 3, and in case
                of RGBA data, it should have value 4.
            y: Labels.
            batch_size: Int (default: 32).
            shuffle: Boolean (default: True).
            sample_weight: Sample weights.
            seed: Int (default: None).
            save_to_dir: None or str (default: None).
                This allows you to optionally specify a directory
                to which to save the augmented pictures being generated
                (useful for visualizing what you are doing).
            save_prefix: Str (default: `''`).
                Prefix to use for filenames of saved pictures
                (only relevant if `save_to_dir` is set).
            save_format: one of "png", "jpeg"
                (only relevant if `save_to_dir` is set). Default: "png".
            subset: Subset of data (`"training"` or `"validation"`) if
                `validation_split` is set in `ImageDataGenerator`.
        # Returns
            An `Iterator` yielding tuples of `(x, y)`
                where `x` is a numpy array of image data
                (in the case of a single image input) or a list
                of numpy arrays (in the case with
                additional inputs) and `y` is a numpy array
                of corresponding labels. If 'sample_weight' is not None,
                the yielded tuples are of the form `(x, y, sample_weight)`.
                If `y` is None, only the numpy array `x` is returned.
        """       
        
        return NumpyArrayIteratorDouble(
            x,
            y,
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            sample_weight=sample_weight,
            seed=seed,
            data_format=self.data_format,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            subset=subset
        )    

# TODO add more parameters    
def build_generators(batch_size=32, target_size= (96,96), only_use_subset=False, use_augment=False, use_capsnet=False):
    (x_train, y_train, meta_train), (x_valid, y_valid, meta_valid), (x_test, y_test, meta_test) = load_data()
    
    if only_use_subset:
        x_train = x_train[0:200]
        y_train = y_train[0:200]
        x_valid = x_valid[0:200]
        y_valid = y_valid[0:200]
        
        x_test = x_test[0:100]
        y_test = y_test[0:100]
    
    if use_capsnet:
        train_datagen = ImageDataGeneratorDouble(
            rescale=1./255,
            # Specify other augmentations here.
        )
        validation_datagen = ImageDataGeneratorDouble(
            rescale=1./255,
        )
        if use_augment:
            train_datagen = ImageDataGeneratorDouble(
                rescale=1./255,
                preprocessing_function = augmentor
            )
        test_datagen = ImageDataGeneratorDouble(
            rescale=1./255,
            # Specify other augmentations here.
        )
    else:
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            # Specify other augmentations here.
        )

        validation_datagen = ImageDataGenerator(
            rescale=1./255,
        )
        if use_augment:
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                preprocessing_function = augmentor
            )
        test_datagen = ImageDataGenerator(
            rescale=1./255,
        )
    
    print("[!] Creating validation_generator")
    validation_generator = validation_datagen.flow(
        x=x_valid,
        y=np.ravel(y_valid),
        batch_size=batch_size,
        shuffle=False,
        seed=0)

    print("[!] Creating train_generator")
    train_generator = train_datagen.flow(
        x=x_train,
        y=np.ravel(y_train),
        batch_size=batch_size)

    print("[!] Creating test_generator")
    test_generator = test_datagen.flow(
        x=x_test,
        y=np.ravel(y_test),
        batch_size=32,
        seed=0,
        shuffle=False
    )

    return train_generator, validation_generator, test_generator

def run_experiment(config, predict_test = False, predict_val = False, reproduce_result=None):

    ex = Experiment('ISMI', interactive=True)
    # log to the mongoDB instance living in the cloud
    client = pymongo.MongoClient(MONGO_URI)
    ex.observers.append(MongoObserver.create(client=client))
    
    ex.add_config(config)
    ex.add_config({'seed': RANDOM_SEED})
    
    use_capsnet = config.get('use_capsnet')

    @ex.capture
    def my_metrics(_run, logs):
        if not config.get('use_capsnet'):
            _run.log_scalar("loss", float(logs.get('loss')))
            _run.log_scalar("acc", float(logs.get('acc')))
            _run.log_scalar("val_loss", float(logs.get('val_loss')))
            _run.log_scalar("val_acc", float(logs.get('val_acc')))
            _run.result = float(logs.get('val_acc'))
        else:
            _run.log_scalar("loss", float(logs.get('capsnet_loss')))
            _run.log_scalar("acc", float(logs.get('capsnet_acc')))
            _run.log_scalar("val_loss", float(logs.get('val_capsnet_loss')))
            _run.log_scalar("val_acc", float(logs.get('val_capsnet_acc')))
            _run.result = float(logs.get('val_capsnet_acc'))

    # a callback to log to Sacred, found here: https://www.hhllcks.de/blog/2018    /5/4/version-your-machine--models-with-sacred
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
        train_generator, validation_generator, test_generator =            build_generators(batch_size=batch_size,target_size=target_size,only_use_subset=only_use_subset,use_augment=use_augment,use_capsnet=use_capsnet)
        
        print('[!] Building model')
        
        # The function for building the model
        model_func = model_dict[config.get('model')]
        # Actually invoke the function
        model_params = config.get('model_params',{})
        lr = config.get('lr',0.001)
        if config.get('model') == 'dense' and 'target_size' not in model_params:
            model_params['target_size'] = target_size

        # TODO add kwargs
        model = model_func(**model_params)

        # TODO: parametrize optimizer and learning rate here
        model.compile(optimizer=Adam(lr=lr), loss='binary_crossentropy', metrics=['accuracy'])
        
        # print(model.summary())

        tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

        epochs = config.get('epochs')
        
        print('[!] Training model')
        # Reduce learning rate on plateau
        if config.get('reduce_lr_on_plateau'):
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, min_lr=10e-5)
        else: 
            reduce_lr = Callback()
        
        # Save best models to file
        modelcheckpoint_name = "checkpoints/model-{}.hdf5".format(time())
        if reproduce_result:
            modelcheckpoint_name = "checkpoints/model-{}.hdf5".format(reproduce_result)
        
        modelcheckpoint = ModelCheckpoint(modelcheckpoint_name, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)
        
        # Stop early when val_loss does not increase anymore
        earlystopping = EarlyStopping(monitor='val_loss', patience=10)
        model.fit_generator(train_generator, 
                            steps_per_epoch=len(train_generator),
                            epochs=epochs, 
                            validation_data=validation_generator, 
                            validation_steps=len(validation_generator),
                            callbacks=[tensorboard, LogMetrics(), modelcheckpoint, earlystopping])     
        print(model.evaluate_generator(validation_generator, steps=len(validation_generator)))
            
        if predict_val: 
            print('[!] Predicting validation set')
            model.load_weights(modelcheckpoint_name)
            prediction = model.predict_generator(validation_generator, steps=len(validation_generator), verbose=1)
            if config.get('use_capsnet'):
                prediction = prediction[0]
            data = {'case': np.arange(len(prediction)), 'prediction': np.ravel(prediction)}
            submission = pd.DataFrame(data=data)
            submission.to_csv('./predictions/validation_{}.csv'.format(predict_val), index=False)
            
        if predict_test:
            print('[!] Predicting test set')
            model.load_weights(modelcheckpoint_name)
            prediction = model.predict_generator(test_generator, steps=len(test_generator), verbose=1)
            if config.get('use_capsnet'):
                prediction = prediction[0]
            data = {'case': np.arange(len(prediction)), 'prediction': np.ravel(prediction)}
            submission = pd.DataFrame(data=data)
            submission.to_csv('./predictions/test_{}.csv'.format(predict_test), index=False)
            
            display(HTML('Prediction saved to file: <a target="_blank" href="./predictions/test_{}.csv">data/{}.csv</a>'.format(predict_test, predict_test)))
        
            # add the submissions as an artifact
            _run.add_artifact('./predictions/validation_{}.csv'.format(predict_val))
            _run.add_artifact('./predictions/test_{}.csv'.format(predict_test))

            
    run = ex.run()
    
    return run

model_dict = {
    'dense' : build_dense,
    'resnet': build_resnet, 
    'nasnet': build_nasnet,
    'convnet': build_convnet,
    'convnet_reg': build_convnet_reg,
    'capsnet': build_capsnet,
    'capsnet_bn': build_capsnet_bn,
    'recnn': build_recnn,
    'custom_convnet':build_custom_convnet
}
    

