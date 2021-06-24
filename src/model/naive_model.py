
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Activation
from tensorflow.keras.layers import BatchNormalization, Dropout, Flatten


class NaiveEatModel(tf.keras.Model):

    def __init__(self, num_classes: int):
        super(NaiveEatModel, self).__init__()

        # first convolution layer with batch normalization and max pooling
        self.nn_conv_1 = Conv2D(8, (3, 3), strides=1, padding='same')
        self.norm_1 = BatchNormalization()
        self.nn_act_1 = Activation('relu')
        self.nn_pool_1 = MaxPooling2D()

        # second convolution layer only with max pooling, no padding, no activation
        self.nn_conv_2 = Conv2D(16, (3, 3), strides=1, padding='valid')
        self.nn_pool_2 = MaxPooling2D()

        # add a dropout layer to filter conv signal noise
        self.dropout_2 = Dropout(rate=0.5)

        # output logits layer
        self.flatten = Flatten()
        self.dense_out = Dense(num_classes, activation='softmax')


    def call(self, inputs, training=False):

        x = inputs

        # process the first convolution layer
        x = self.nn_conv_1(x)
        x = self.norm_1(x)
        x = self.nn_act_1(x)
        x = self.nn_pool_1(x)

        # process the second convolution layer
        x = self.nn_conv_2(x)
        x = self.nn_pool_2(x)

        # add a dropout layer to filter conv signal noise
        if training: x = self.dropout_2(x)

        # process the logits output layer
        x = self.flatten(x)
        x = self.dense_out(x)

        return x