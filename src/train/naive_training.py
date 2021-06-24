
import tensorflow as tf
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

from dataset import load_cached_tfrecord_datasets
from model import NaiveEatModel


class NaiveTrainingSession:

    def __init__(self, params: dict):
        super(NaiveTrainingSession, self).__init__()

        self.params = params

        # load the training / test datasets and prepare the model to be trained
        self.train_data, self.test_data = load_cached_tfrecord_datasets(params)
        self.model = self.load_model(params)


    def load_model(self, params: dict):

        # create the optimizer and loss function
        loss_func = SparseCategoricalCrossentropy()
        optimizer = Adam()

        # prepare the model for training
        model = NaiveEatModel(params['num_classes'])
        model.compile(optimizer=optimizer, loss=loss_func, metrics=['accuracy'])
        model.build((None, 128, 126, 1))

        return model


    def run_training(self):

        # make the datasets iterable
        train_data_iter = iter(self.train_data)
        test_data_iter = iter(self.test_data)

        # print first training batch
        print(train_data_iter.next())

        # do the training by fitting the model to the training dataset
        history = self.model.fit(x=train_data_iter, epochs=self.params['num_epochs'],
                                 validation_data=test_data_iter)

        # TODO: write the history to tensorboard (or at least some matplotlib diagrams)
