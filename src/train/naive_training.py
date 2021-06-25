
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
        self.dataset = load_cached_tfrecord_datasets(params, shuffle=False)
        self.model = self.create_model(params)

        # create model checkpoint manager
        self.ckpt_dir = './trained_models/naive'
        ckpt_path = self.ckpt_dir + "/naive-{epoch:04d}.ckpt"
        self.model_ckpt_callback = ModelCheckpoint(
            filepath=ckpt_path, save_weights_only=False,
            monitor='val_accuracy', mode='max', save_best_only=True)

        # create tensorboard logger
        logdir = './logs/naive/naive' + datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tb_callback = TensorBoard(log_dir=logdir)


    def create_model(self, params: dict):

        # create the optimizer and loss function
        loss_func = SparseCategoricalCrossentropy()
        optimizer = Adam()

        # prepare the model for training
        model = NaiveEatModel(params['num_classes'])
        model.compile(optimizer=optimizer, loss=loss_func, metrics=['accuracy'])

        melspec_shape = params['melspec_shape']
        model.build((None, melspec_shape[0], melspec_shape[1], 1))

        return model


    def load_best_model(self):

        # load the best model from checkpoint cache
        best_ckpt = tf.train.latest_checkpoint(self.ckpt_dir)
        best_model = self.create_model(self.params)
        best_model.load_weights(best_ckpt)
        self.model = best_model


    def run_training(self):

        # split the dataset into train / eval / test portions (60/20/20)
        num_batches = int(np.ceil(self.params['num_train_samples'] / self.params['batch_size']))
        num_train_batches = int(num_batches * 0.6)
        num_eval_batches = int(num_batches * 0.2)
        train_data = self.dataset.take(num_train_batches).shuffle(20)
        eval_data = self.dataset.skip(num_train_batches).take(num_eval_batches)
        test_data = self.dataset.skip(num_train_batches + num_eval_batches)

        # evaluate the untrained model on the test dataset
        self.model.evaluate(x=test_data)

        # do the training by fitting the model to the training dataset
        self.model.fit(x=train_data, epochs=self.params['num_epochs'],
                       validation_data=eval_data,
                       callbacks=[self.tb_callback, self.model_ckpt_callback])

        # evaluate the 'best' model checkpoint on the test dataset
        self.load_best_model()
        self.model.evaluate(x=test_data)
