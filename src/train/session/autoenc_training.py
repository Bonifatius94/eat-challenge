
from datetime import datetime
from matplotlib import pyplot as plt
import tensorflow as tf

from tensorflow.keras.losses import SparseCategoricalCrossentropy, MSE
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

from dataset import DatasetFactory
from dataset.core import partition_dataset
from model import AutoEncEatModel, AutoEncClassifModel


class AutoEncTrainingSession:

    def __init__(self, params: dict):
        super(AutoEncTrainingSession, self).__init__()

        self.params = params

        # load the train / eval / test datasets
        self.train_data, self.eval_data, self.test_data = self.load_dataset(params)

        # prepare the models to be trained
        self.autoenc_model, self.model = self.create_model(params)

        # create the model checkpoint manager
        self.ckpt_dir = './trained_models/autoenc'

        ckpt_path = self.ckpt_dir + "/autoenc.ckpt"
        self.model_ckpt_callback = ModelCheckpoint(
            filepath=ckpt_path, save_weights_only=False,
            monitor='val_accuracy', mode='max', save_best_only=True)

        # create the tensorboard logger
        logdir = './logs/autoenc/autoenc' + datetime.now().strftime("%Y%m%d-%H%M%S")
        self.autoenc_tb_callback = TensorBoard(log_dir=logdir)
        logdir = './logs/autoenc/classif' + datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tb_callback = TensorBoard(log_dir=logdir)


    def load_dataset(self, params: dict):

        # load the tfrecord melspec training dataset
        factory = DatasetFactory()
        dataset = factory.load_dataset(params['dataset_specifier'], params, train=True)

        # batch the dataset properly and tune the performance by prefetching
        dataset = dataset.batch(params['batch_size'])
        dataset = dataset.prefetch(5)

        # partition the dataset into train / eval / test splits
        train, eval, test = partition_dataset(dataset, params['dataset_splits'])
        train = train.shuffle(50)

        return train, eval, test


    def create_model(self, params: dict):

        # create the optimizer and loss function
        autoenc_optimizer = Adam()
        classif_optimizer = Adam()

        # prepare the model for training
        autoenc_model = AutoEncEatModel(out_channels=params['encoder_out_channels'])
        autoenc_model.compile(optimizer=autoenc_optimizer, loss=MSE)

        # prepare the model for training
        classif_model = AutoEncClassifModel(encoder=autoenc_model.nn_encoder,
            num_classes=params['num_classes'])
        classif_model.compile(optimizer=classif_optimizer,
                              loss=SparseCategoricalCrossentropy(),
                              metrics=['accuracy'])

        # set the input data shape
        input_shape = [None] + params['inputs_shape']
        autoenc_model.build(input_shape)
        classif_model.build(input_shape)

        return autoenc_model, classif_model


    def load_best_model(self):

        # load the best model from checkpoint cache
        best_ckpt = tf.train.latest_checkpoint(self.ckpt_dir)
        _, best_model = self.create_model(self.params)
        best_model.load_weights(best_ckpt)
        self.model = best_model


    def run_training(self):

        # print a summary of the model layers to be trained
        print(self.model.summary())

        # check if training should be skipped
        if not self.params['skip_training']:

            # evaluate the untrained model on the test dataset
            self.model.evaluate(x=self.test_data)

            # minimize the reconstruction error of the auto encoder model
            autoenc_train_data = self.train_data.map(lambda x, _: (x, x))
            autoenc_eval_data = self.eval_data.map(lambda x, _: (x, x))
            self.autoenc_model.fit(x=autoenc_train_data, epochs=self.params['num_autoenc_epochs'],
                           validation_data=autoenc_eval_data, callbacks=[self.autoenc_tb_callback])

            # sample some melspec reconstructions
            # self.print_auto_enc_samples()

            # train the classifier on the training dataset split (freezed encoder)
            self.model.nn_encoder.trainable = False
            self.model.fit(x=self.train_data, epochs=int(self.params['num_epochs']),
                           validation_data=self.eval_data,
                           callbacks=[self.tb_callback, self.model_ckpt_callback])

        # evaluate the 'best' model checkpoint on the test dataset
        self.load_best_model()
        self.model.evaluate(x=self.test_data)


    def print_auto_enc_samples(self):
        n_examples = 6
        examples, _ = iter(self.eval_data).next()
        predictions = self.autoenc_model.predict(examples)
        fig, axes = plt.subplots(nrows=n_examples, ncols=2, figsize=(15, 30))

        for i in range(n_examples):
            axes[i][0].imshow(examples[i])
            axes[i][1].imshow(predictions[i])
        plt.show()
