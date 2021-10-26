
from datetime import datetime
import numpy as np
import tensorflow as tf

from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

from dataset import DatasetFactory
from dataset.core import partition_dataset
from model import NaiveEatModel
from model.custom_callbacks import SaveBestModelCallbackFactory


class NaiveTrainingSession:

    def __init__(self, params: dict):
        super(NaiveTrainingSession, self).__init__()

        self.params = params

        # load the train / eval / test datasets
        self.train_data, self.eval_data, self.test_data = self.load_dataset(params)

        # prepare the model to be trained
        self.model = self.create_model(params)

        # create the model checkpoint manager
        self.ckpt_dir = './trained_models/naive'
        ckpt_path = self.ckpt_dir + "/naive.ckpt"
        self.model_ckpt_callback = SaveBestModelCallbackFactory().get_callback(ckpt_path)

        # create the tensorboard logger
        logdir = './logs/naive/naive' + datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tb_callback = TensorBoard(log_dir=logdir)


    def load_dataset(self, params: dict):

        # load the tfrecord melspec training dataset
        factory = DatasetFactory()
        train_ds, test_ds = factory.load_dataset(params['dataset_specifier'], params, train=True)

        # batch the dataset properly and tune the performance by prefetching
        train_ds = train_ds.batch(params['batch_size'])
        train_ds = train_ds.prefetch(5)
        test_ds = test_ds.batch(params['batch_size'])
        test_ds = test_ds.prefetch(5)

        # partition the dataset into train / eval / test splits
        train_ds, eval_ds = self.partition_dataset(train_ds, params['dataset_splits'])
        train_ds = train_ds.shuffle(50)

        return train_ds, eval_ds, test_ds


    def partition_dataset(self, dataset: tf.data.Dataset, splits: list):

        # make sure the splits are valid
        if np.sum(np.array(splits)) != 1.0:
            raise ValueError('Invalid dataset splits! The sum of all splits needs to be 1!')

        # determine the amount of batches on the dataset
        num_samples = int(np.sum([1 for _ in iter(dataset)]))

        # determine the amount of train / eval / test batches (test batches = rest)
        num_train_batches = int(num_samples * splits[0])
        # num_eval_batches = int(num_samples * splits[1])

        # partition the train / eval / test datasets
        train_data = dataset.take(num_train_batches)
        eval_data = dataset.skip(num_train_batches)

        return train_data, eval_data


    def create_model(self, params: dict):

        # create the optimizer and loss function
        loss_func = SparseCategoricalCrossentropy()
        optimizer = Adam()

        # prepare the model for training
        model = NaiveEatModel(params['num_classes'])
        model.compile(optimizer=optimizer, loss=loss_func, metrics=['accuracy'])

        # set the input data shape
        input_shape = [None] + params['inputs_shape']
        model.build(input_shape)

        return model


    def load_best_model(self):

        # load the best model from checkpoint cache
        best_ckpt = tf.train.latest_checkpoint(self.ckpt_dir)
        best_model = self.create_model(self.params)
        best_model.load_weights(best_ckpt)
        self.model = best_model


    def run_training(self):

        # print a summary of the model layers to be trained
        print(self.model.summary())

        # check if training should be skipped
        if not self.params['skip_training']:

            # evaluate the untrained model on the test dataset
            self.model.evaluate(x=self.test_data)

            # do the training by fitting the model to the training dataset
            self.model.fit(x=self.train_data, epochs=self.params['num_epochs'],
                           validation_data=self.eval_data,
                           callbacks=[self.tb_callback, self.model_ckpt_callback])

        # evaluate the 'best' model checkpoint on the test dataset
        self.load_best_model()
        self.model.evaluate(x=self.test_data)
