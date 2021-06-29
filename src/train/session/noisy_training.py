
from datetime import datetime
import numpy as np
import tensorflow as tf

from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

from dataset import DatasetFactory
from model import NoisyEatModel


class NoisyTrainingSession:

    def __init__(self, params: dict):
        super(NoisyTrainingSession, self).__init__()

        # TODO: rework training to fit the new config file parameters

        self.params = params

        # load the training / test datasets and prepare the model to be trained
        self.dataset = self.load_dataset(params)
        self.model = self.create_model(params)

        # create model checkpoint manager
        self.ckpt_dir = './trained_models/naive'
        ckpt_path = self.ckpt_dir + "/naive.ckpt"
        # ckpt_path = self.ckpt_dir + "/naive-{epoch:04d}.ckpt"
        self.model_ckpt_callback = ModelCheckpoint(
            filepath=ckpt_path, save_weights_only=False,
            monitor='val_accuracy', mode='max', save_best_only=True)

        # create tensorboard logger
        logdir = './logs/naive/naive' + datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tb_callback = TensorBoard(log_dir=logdir)


    def load_dataset(self, params: dict):

        # load the tfrecord melspec training dataset
        factory = DatasetFactory()
        dataset = factory.load_dataset(params['dataset_specifier'], params, train=True)

        # batch the dataset properly and tune the performance by prefetching
        dataset = dataset.batch(params['batch_size'])
        dataset = dataset.prefetch(5)

        return dataset


    def create_model(self, params: dict):

        # create the optimizer and loss function
        loss_func = SparseCategoricalCrossentropy()
        optimizer = Adam()

        # prepare the model for training
        model = NoisyEatModel(params['num_classes'])
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

        # split the dataset into train / eval / test portions (60/20/20)
        num_batches = int(np.ceil(self.params['num_train_samples'] / self.params['batch_size']))
        num_train_batches = int(num_batches * 0.6)
        num_eval_batches = int(num_batches * 0.2)
        train_data = self.dataset.take(num_train_batches).shuffle(20)
        eval_data = self.dataset.skip(num_train_batches).take(num_eval_batches)
        test_data = self.dataset.skip(num_train_batches + num_eval_batches)

        # check if training should be skipped
        if not self.params['skip_training']:

            # evaluate the untrained model on the test dataset
            self.model.evaluate(x=test_data)

            # do the training by fitting the model to the training dataset
            self.model.fit(x=train_data, epochs=self.params['num_epochs'],
                        validation_data=eval_data,
                        callbacks=[self.tb_callback, self.model_ckpt_callback])

        # evaluate the 'best' model checkpoint on the test dataset
        self.load_best_model()
        self.model.evaluate(x=test_data)
