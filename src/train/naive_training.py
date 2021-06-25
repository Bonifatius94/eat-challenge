
from datetime import datetime
import numpy as np
import tensorflow as tf

from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

from dataset import load_cached_tfrecord_datasets
from model import NaiveEatModel


class NaiveTrainingSession:

    def __init__(self, params: dict):
        super(NaiveTrainingSession, self).__init__()

        self.params = params

        # load the training / test datasets and prepare the model to be trained
        self.train_data = load_cached_tfrecord_datasets(params)
        self.model = self.load_model(params)

        # create model checkpoint manager
        self.model_ckpt_callback = ModelCheckpoint(
            filepath='./trained_models/naive', save_weights_only=False,
            monitor='val_accuracy', mode='max', save_best_only=True)

        # create tensorboard logger
        logdir = './logs/naive/naive' + datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tb_callback = TensorBoard(log_dir=logdir)


    def load_model(self, params: dict):

        # create the optimizer and loss function
        loss_func = SparseCategoricalCrossentropy()
        optimizer = Adam()

        # prepare the model for training
        model = NaiveEatModel(params['num_classes'])
        model.compile(optimizer=optimizer, loss=loss_func, metrics=['accuracy'])

        melspec_shape = params['melspec_shape']
        model.build((None, melspec_shape[0], melspec_shape[1], 1))

        return model


    def run_training(self):

        # split the dataset into train / eval / test portions (60/20/20)
        num_batches = int(np.ceil(self.params['num_train_samples'] / self.params['batch_size']))
        num_train_batches = int(num_batches * 0.6)
        num_eval_batches = int(num_batches * 0.2)
        train_data = self.train_data.take(num_train_batches)
        eval_data = self.train_data.skip(num_train_batches).take(num_eval_batches)
        test_data = self.train_data.skip(num_train_batches + num_eval_batches)

        # evaluate the untrained model
        self.model.evaluate(x=test_data)

        # do the training by fitting the model to the training dataset
        history = self.model.fit(x=train_data, epochs=self.params['num_epochs'],
                                 validation_data=eval_data,
                                 callbacks=[self.tb_callback, self.model_ckpt_callback])

        # evaluate the trained model
        # TODO: load the best model from checkpoint, first
        self.model.evaluate(x=test_data)

        # TODO: write the history to tensorboard (or at least some matplotlib diagrams)
