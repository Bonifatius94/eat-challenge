
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Activation
from tensorflow.keras.layers import BatchNormalization, Dropout, Flatten


class NaiveEatModel(tf.keras.Model):

    def __init__(self, num_classes: int):
        super(NaiveEatModel, self).__init__()

        # first convolution layer with batch normalization and max pooling
        self.nn_conv_1 = Conv2D(64, (3, 3), strides=1, padding='same')
        self.nn_bnorm_1 = BatchNormalization()
        self.act_1 = Activation('relu')
        self.maxpool_1 = MaxPooling2D()

        # second convolution layer only with max pooling
        self.nn_conv_2 = Conv2D(32, (3, 3), strides=1, padding='same')
        self.act_2 = Activation('relu')
        self.maxpool_2 = MaxPooling2D()

        # add a dropout layer to filter conv signal noise
        self.dropout = Dropout(rate=0.5)

        # output logits layer
        self.flatten = Flatten()
        self.nn_dense_out = Dense(num_classes, activation='softmax')


    def call(self, inputs, training=False):

        x = inputs

        # process the first convolution layer
        x = self.nn_conv_1(x)
        x = self.nn_bnorm_1(x, training)
        x = self.act_1(x)
        x = self.maxpool_1(x)

        # process the second convolution layer
        x = self.nn_conv_2(x)
        x = self.maxpool_2(x)

        # add a dropout layer to filter conv signal noise
        x = self.dropout(x, training)

        # process the logits output layer
        x = self.flatten(x)
        x = self.nn_dense_out(x)

        return x